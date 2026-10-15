// src/glm/engine_prefill.cu - GpuEngine's prompt path: T tokens per chunk (engine.hpp).
//
// The same computation as eval() token by token, batched: each product runs once for the chunk (gpu_gemm.hpp),
// the KDA recurrence walks the chunk's tokens in one kernel, and MLA attends with two cuBLAS products over the
// f16 latent cache. Every routed expert the chunk uses is computed once for all of its tokens on the GPU: from
// the VRAM cache when it is there, otherwise streamed in from RAM through two pinned buffers while the previous
// one computes (and kept in a free cache slot when there is one).
//
// While the indexer still selects every pool (the context below top_k + kpool - 1 tokens) attention is plain
// causal over the cache. Past that each query attends its own selection, made as the decode path makes it: the
// chunk's pool scores with one product per indexer head, a top-k per query on the GPU (ties to the lower pool,
// like decode's stable sort), then the query's cells gathered side by side and the same two f16 products as the
// causal case, batched per query.
#include "engine_impl.hpp"

#include "strata/glm/gpu_kernels.hpp"
#include "strata/glm/gpu_prefill.hpp"

#include "ggml.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strata::glm {

using namespace detail;

namespace {

/// Carves buffers out of a device arena, 256-byte aligned.
struct Bump {
    uint8_t* base;
    size_t cap, used = 0;
    template <class T>
    T* take(size_t n) {
        const size_t b = (n * sizeof(T) + 255) / 256 * 256;
        if (used + b > cap) throw std::runtime_error("prompt chunk arena too small");
        T* r = reinterpret_cast<T*>(base + used);
        used += b;
        return r;
    }
};

void cb(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("cuBLAS ") + what + " failed");
}

size_t align256(size_t b) { return (b + 255) / 256 * 256; }

}  // namespace

void GpuEngine::setup_prefill() {
    Impl& I = *impl_;
    const Config& c = I.c;
    const int T = I.opt.prefill_chunk;
    if (T <= 1) return;
    const int64_t E = c.n_embd, hc = c.hc, D = c.kda_head_dim, KL = c.kv_lora_rank, HK = c.head_k_mla,
                  HV = c.head_v_mla, QL = c.q_lora_rank, ID = c.indexer_head_size,
                  P = c.indexer_kpool, NE = c.n_expert, KU = c.n_expert_used;
    int64_t maxH = 0, maxDI = 0, maxF = 0, maxFF = c.n_ff;
    for (int64_t il = 0; il < c.n_layer; ++il) {
        maxH = std::max(maxH, c.n_head[(size_t) il]);
        if (c.is_kda(il)) maxDI = std::max(maxDI, c.n_head[(size_t) il] * D);
        if (!c.is_dense(il)) {
            maxF = std::max(maxF, c.n_ff_exp[(size_t) il]);
            maxFF = std::max(maxFF, c.n_ff_exp[(size_t) il] * std::max<int64_t>(c.n_expert_shared, 1));
        }
    }
    // the three phases share one arena: the KDA mixer, the MLA mixer, the FFN
    const size_t f = sizeof(float);
    const int64_t MS = I.max_sel, IH = c.indexer_n_head;
    const size_t kda = align256(8 * T * maxDI * f) + align256(T * D * f) + align256(T * maxH * f) + 8 * 256;
    // MLA: the projections and the indexer's keys, then the attention: over every cell (the dense range), or past
    // it the indexer's queries and pool scores, and per batch of queries their gathered cells, scores and
    // probabilities (sparse_batch() queries at a time, at least min(T, 32))
    const size_t mla_common = align256(T * QL * f) + align256(T * maxH * HK * f) + align256(T * KL * f) +
                              2 * align256(T * maxH * KL * f) + align256(T * maxH * KL * 2) +
                              align256(T * maxH * HV * f) + align256(T * ID * f) + 2 * align256((P + T) * ID * f) +
                              8 * 256;
    const size_t dense_att = align256(T * maxH * MS * f) + align256(T * maxH * MS * 2);
    const int64_t nb = std::min<int64_t>(T, 32);
    const size_t sparse_att = align256(T * IH * ID * f) + align256(T * IH * f) + 2 * align256(T * I.max_pools * f) +
                              align256(nb * MS * KL * 2) + align256(nb * maxH * MS * f) + align256(nb * maxH * MS * 2) +
                              8 * 256;
    const size_t mla = mla_common + std::max(dense_att, sparse_att);
    const size_t ffn = 3 * align256(T * maxFF * f) + align256(T * NE * f) + align256(T * KU * f) +
                       align256(T * KU * sizeof(int32_t)) + align256(T * KU * E * f) + 3 * align256(T * E * f) + 10 * 256;
    const size_t arena = std::max({kda, mla, ffn});

    // the experts' blob, for the streaming buffers
    for (int64_t il = 0; il < c.n_layer; ++il) {
        if (c.is_dense(il)) continue;
        const LayerWeights& w = I.m.layer(il);
        const size_t blob = w.gate_exps.row_bytes() * (size_t) w.gate_exps.ne[1] +
                            w.up_exps.row_bytes() * (size_t) w.up_exps.ne[1] +
                            w.down_exps.row_bytes() * (size_t) w.down_exps.ne[1];
        I.blob_max = std::max(I.blob_max, align256(blob));
    }

    for (size_t di = 0; di < I.ctx.size(); ++di) {
        DevCtx& d = I.ctx[di];
        ok(cudaSetDevice(d.id), "cudaSetDevice");
        size_t* b = &d.bytes;
        d.T = T;
        d.pR = dalloc<float>((size_t) (T * hc * E), b);
        d.pR2 = dalloc<float>((size_t) (T * hc * E), b);
        d.pflat = dalloc<float>((size_t) (T * hc * E), b);
        d.pmixes = dalloc<float>((size_t) (T * c.hc_mix_dim()), b);
        d.ppc = dalloc<float>((size_t) (T * (2 * hc + hc * hc)), b);
        d.px = dalloc<float>((size_t) (T * E), b);
        d.pxn = dalloc<float>((size_t) (T * E), b);
        d.py = dalloc<float>((size_t) (T * E), b);
        d.arena = dalloc<uint8_t>(arena, b);
        d.arena_bytes = arena;
        d.d_idx = dalloc<int32_t>((size_t) (T * KU), b);
        d.d_slot = dalloc<int32_t>((size_t) (T * KU), b);
        d.sel_rows = dalloc<int32_t>((size_t) (T * MS), b);
        d.n_sel = dalloc<int32_t>((size_t) T, b);
        // the largest weight a product dequantizes on this device, and the largest activation it converts
        size_t welems = (size_t) (E * maxF);
        for (int64_t il = 0; il < c.n_layer; ++il) {
            if (I.ctx_of_layer[(size_t) il] != (int) di) continue;
            const DevLayer& l = I.L[(size_t) il];
            for (const DevMatrix* m : {&l.hc_attn_fn, &l.hc_ffn_fn, &l.wo, &l.wq, &l.wk, &l.wv, &l.f_a, &l.f_b,
                                       &l.beta, &l.g_a, &l.g_b, &l.q_a, &l.q_b, &l.kv_a, &l.k_b, &l.v_b, &l.idx_q_b,
                                       &l.idx_k, &l.idx_gate, &l.idx_proj, &l.ffn_gate, &l.ffn_up, &l.ffn_down,
                                       &l.gate_inp, &l.gate_shexp, &l.up_shexp, &l.down_shexp})
                if (m->ok()) welems = std::max(welems, (size_t) (m->n_in * m->n_out));
        }
        const size_t act = (size_t) T * std::max({hc * E, maxDI, maxH * KL, maxH * HK, maxH * HV, maxFF, E, QL});
        d.gemm = std::make_unique<gpu::GemmCtx>(d.id);
        d.gemm->reserve(welems, act);
        d.bytes += welems * 2 + act * 2;
        if (I.blob_max) {
            for (int k = 0; k < 2; ++k) {
                d.dstage[k] = dalloc<uint8_t>(I.blob_max, b);
                ok(cudaEventCreateWithFlags(&d.ev_copied[k], cudaEventDisableTiming), "cudaEventCreate");
                ok(cudaEventCreateWithFlags(&d.ev_free[k], cudaEventDisableTiming), "cudaEventCreate");
                ok(cudaEventRecord(d.ev_free[k], d.stream), "cudaEventRecord");
                ok(cudaEventRecord(d.ev_copied[k], d.stream), "cudaEventRecord");
            }
            ok(cudaStreamCreateWithFlags(&d.copy_stream, cudaStreamNonBlocking), "cudaStreamCreate");
        }
    }
    if (I.blob_max)
        for (auto& h : I.hstage) h = halloc<uint8_t>(I.blob_max);
    I.h_sel_rows = halloc<int32_t>((size_t) (T * MS));
    I.h_n_sel = halloc<int32_t>((size_t) T);
}

const std::vector<float>& GpuEngine::prefill(const int32_t* ids, int64_t n,
                                             const std::function<bool(int64_t)>& progress) {
    Impl& I = *impl_;
    const Config& c = I.c;
    if (n <= 0) throw std::runtime_error("prefill of no tokens");
    // Positions whose query still sees every cell, (t + 1) / P <= top_k / P with the tail selected, attend over
    // all of them (the dense chunk path). Past those each query attends its own selection (sparse chunks), which
    // a shared indexer layer takes from the full one before it: without one, the one-token path.
    bool any_mla = false, full_seen = false, sparse_ok = true;
    for (int64_t il = 0; il < c.n_layer; ++il) {
        if (c.is_kda(il)) continue;
        any_mla = true;
        if (c.indexer_full[(size_t) il]) full_seen = true;
        else if (!full_seen) sparse_ok = false;
    }
    const int64_t dense_until = !any_mla ? INT64_MAX
                                : c.indexer_select_tail ? (c.indexer_top_k / c.indexer_kpool + 1) * c.indexer_kpool - 1
                                                        : 0;
    int64_t i = 0;
    while (i < n) {
        const int64_t room = std::min<int64_t>(I.opt.prefill_chunk, n - i);
        const bool sparse = pos_ >= dense_until;
        const int64_t T = !sparse ? std::min<int64_t>(room, dense_until - pos_) : sparse_ok ? room : 0;
        if (T >= 2) {
            run_chunk(ids + i, (int) T, sparse);
            i += T;
        } else {
            eval(ids[i]);
            ++i;
        }
        if (progress && !progress(i)) break;
    }
    return I.logits;
}

bool GpuEngine::run_chunk(const int32_t* ids, int T, bool sparse) {
    Impl& I = *impl_;
    const Config& c = I.c;
    const Model& m = I.m;
    const int64_t E = c.n_embd, hc = c.hc, t0 = pos_;
    if (t0 + T > I.opt.max_ctx) throw std::runtime_error("the sequence is longer than the context (--ctx)");

    // ---- embeddings, dequantized on the host, copied into every stream
    {
        std::vector<float> emb((size_t) (T * hc * E));
        const Tensor& te = m.tok_embd();
        for (int t = 0; t < T; ++t) {
            if (ids[t] < 0 || ids[t] >= c.n_vocab) throw std::runtime_error("token id outside the vocabulary");
            float* row = emb.data() + (size_t) t * hc * E;
            if (te.type == GGML_TYPE_F32) std::memcpy(row, te.row(ids[t]), (size_t) E * sizeof(float));
            else ggml_get_type_traits((ggml_type) te.type)->to_float(te.row(ids[t]), row, E);
            for (int64_t h = 1; h < hc; ++h) std::memcpy(row + h * E, row, (size_t) E * sizeof(float));
        }
        DevCtx& d0 = I.ctx[(size_t) I.ctx_of_layer[0]];
        ok(cudaSetDevice(d0.id), "cudaSetDevice");
        ok(cudaMemcpy(d0.pR, emb.data(), emb.size() * sizeof(float), cudaMemcpyHostToDevice), "embeddings");
    }

    int cur = I.ctx_of_layer[0];
    for (int64_t il = 0; il < c.n_layer; ++il) {
        const int ci = I.ctx_of_layer[(size_t) il];
        if (ci != cur) {
            DevCtx& from = I.ctx[(size_t) cur];
            DevCtx& to = I.ctx[(size_t) ci];
            ok(cudaSetDevice(from.id), "cudaSetDevice");
            ok(cudaStreamSynchronize(from.stream), "sync");
            ok(cudaMemcpyPeer(to.pR, to.id, from.pR, from.id, (size_t) (T * hc * E) * sizeof(float)), "peer copy");
            cur = ci;
        }
        DevCtx& d = I.ctx[(size_t) ci];
        DevLayer& l = I.L[(size_t) il];
        ok(cudaSetDevice(d.id), "cudaSetDevice");
        void* s = d.stream;
        gpu::GemmCtx& G = *d.gemm;
        const int64_t H = c.n_head[(size_t) il];

        // ---- attention half
        gpu::rms_norm_rows(d.pR, d.pflat, T, (int) (hc * E), c.rms_eps, nullptr, s);
        G.gemm(l.hc_attn_fn, d.pflat, T, d.pmixes, s);
        gpu::hc_weights_rows(d.pmixes, l.hc_attn_scale, l.hc_attn_base, d.ppc, T, (int) hc, (int) c.hc_sinkhorn_iters,
                             c.hc_eps, s);
        gpu::hc_mix_rows(d.pR, d.ppc, d.px, T, (int) hc, (int) E, s);
        gpu::rms_norm_rows(d.px, d.pxn, T, (int) E, c.rms_eps, l.attn_norm, s);

        Bump A{d.arena, d.arena_bytes};
        if (c.is_kda(il)) {
            const int64_t D = c.kda_head_dim, DI = H * D, K = c.ssm_d_conv;
            float* raw = A.take<float>((size_t) (T * DI));
            float* q = A.take<float>((size_t) (T * DI));
            float* k = A.take<float>((size_t) (T * DI));
            float* v = A.take<float>((size_t) (T * DI));
            float* g = A.take<float>((size_t) (T * DI));
            float* g2 = A.take<float>((size_t) (T * DI));
            float* o = A.take<float>((size_t) (T * DI));
            float* gated = A.take<float>((size_t) (T * DI));
            float* fa = A.take<float>((size_t) (T * D));
            float* bta = A.take<float>((size_t) (T * H));
            float* outs[3] = {q, k, v};
            const DevMatrix* proj[3] = {&l.wq, &l.wk, &l.wv};
            for (int j = 0; j < 3; ++j) {
                G.gemm(*proj[j], d.pxn, T, raw, s);
                gpu::kda_conv_rows(raw, l.conv_st[j], l.conv_w[j], outs[j], T, (int) DI, (int) K, s);
            }
            G.gemm(l.f_a, d.pxn, T, fa, s);
            G.gemm(l.f_b, fa, T, g, s);
            gpu::kda_gate_rows(g, l.dt_b, l.a, T, (int) H, (int) D, c.has_gate_lower_bound, c.kda_gate_lower_bound, s);
            G.gemm(l.beta, d.pxn, T, bta, s);
            gpu::sigmoid_inplace(bta, (int) (T * H), s);
            gpu::l2_norm_heads(q, (int) (T * H), (int) D, s);
            gpu::l2_norm_heads(k, (int) (T * H), (int) D, s);
            gpu::kda_scan(l.S, q, k, v, g, bta, o, T, (int) H, (int) D, s);
            G.gemm(l.g_a, d.pxn, T, fa, s);
            G.gemm(l.g_b, fa, T, g2, s);
            gpu::kda_out_gate(o, l.o_norm, g2, gated, (int) (T * H), (int) D, c.rms_eps, s);
            G.gemm(l.wo, gated, T, d.py, s);
        } else {
            const int64_t QL = c.q_lora_rank, KL = c.kv_lora_rank, HK = c.head_k_mla, HV = c.head_v_mla;
            float* qr = A.take<float>((size_t) (T * QL));
            float* qm = A.take<float>((size_t) (T * H * HK));
            float* kv = A.take<float>((size_t) (T * KL));
            float* q_abs = A.take<float>((size_t) (T * H * KL));
            uint16_t* q16 = A.take<uint16_t>((size_t) (T * H * KL));
            float* lat = A.take<float>((size_t) (T * H * KL));
            float* heads = A.take<float>((size_t) (T * H * HV));
            G.gemm(l.q_a, d.pxn, T, qr, s);
            gpu::rms_norm_rows(qr, qr, T, (int) QL, c.rms_eps, l.q_a_norm, s);
            G.gemm(l.q_b, qr, T, qm, s);
            G.gemm(l.kv_a, d.pxn, T, kv, s);
            gpu::rms_norm_rows(kv, kv, T, (int) KL, c.rms_eps, l.kv_a_norm, s);
            gpu::store_f16_rows(kv, l.cache, t0, T, (int) KL, s);
            G.gemm_heads(l.k_b, (int) H, qm, (int) (H * HK), HK, T, q_abs, (int) (H * KL), KL, s);
            gpu::to_f16(q_abs, q16, T * H * KL, s);

            if (c.indexer_full[(size_t) il]) {
                // the pooled keys later tokens' selections read: the pool being filled carries over chunks
                const int64_t ID = c.indexer_head_size, P = c.indexer_kpool, r = t0 % P;
                float* ik = A.take<float>((size_t) (T * ID));
                float* kb = A.take<float>((size_t) ((P + T) * ID));
                float* gb = A.take<float>((size_t) ((P + T) * ID));
                if (r) {
                    ok(cudaMemcpyAsync(kb, l.pend_k, (size_t) (r * ID) * sizeof(float), cudaMemcpyDeviceToDevice,
                                       d.stream), "pending keys");
                    ok(cudaMemcpyAsync(gb, l.pend_g, (size_t) (r * ID) * sizeof(float), cudaMemcpyDeviceToDevice,
                                       d.stream), "pending gates");
                }
                G.gemm(l.idx_k, d.pxn, T, ik, s);
                gpu::layer_norm_rows(ik, kb + r * ID, T, (int) ID, c.norm_eps, l.idx_k_norm, l.idx_k_norm_b, s);
                G.gemm(l.idx_gate, d.pxn, T, gb + r * ID, s);
                const int64_t n_done = (r + T) / P, rem = (r + T) % P;
                gpu::kpool_rows(kb, gb, l.idx_ape, l.pooled, (t0 - r) / P, (int) n_done, 0, (int) P, (int) ID, s);
                if (rem) {
                    ok(cudaMemcpyAsync(l.pend_k, kb + n_done * P * ID, (size_t) (rem * ID) * sizeof(float),
                                       cudaMemcpyDeviceToDevice, d.stream), "pending keys");
                    ok(cudaMemcpyAsync(l.pend_g, gb + n_done * P * ID, (size_t) (rem * ID) * sizeof(float),
                                       cudaMemcpyDeviceToDevice, d.stream), "pending gates");
                }
            }

            cublasHandle_t hb = (cublasHandle_t) G.cublas();
            const float scale = 1.0f / std::sqrt((float) HK), one = 1.0f, zero = 0.0f;
            if (!sparse) {
                // causal attention over the latent cache: S = Q K^T * scale, f16 softmax, O = P K
                const int n_ctx = (int) (t0 + T);
                float* Sc = A.take<float>((size_t) (T * H) * (size_t) n_ctx);
                uint16_t* P16 = A.take<uint16_t>((size_t) (T * H) * (size_t) n_ctx);
                cb(cublasSetStream(hb, d.stream), "cublasSetStream");
                cb(cublasGemmEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, n_ctx, (int) (T * H), (int) KL, &scale, l.cache,
                                CUDA_R_16F, (int) KL, q16, CUDA_R_16F, (int) KL, &zero, Sc, CUDA_R_32F, n_ctx,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "attention scores");
                gpu::causal_softmax_f16(Sc, P16, T, (int) H, n_ctx, t0, s);
                cb(cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, (int) KL, (int) (T * H), n_ctx, &one, l.cache,
                                CUDA_R_16F, (int) KL, P16, CUDA_R_16F, n_ctx, &zero, lat, CUDA_R_32F, (int) KL,
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "attention values");
            } else {
                const int64_t IH = c.indexer_n_head, ID = c.indexer_head_size, P = c.indexer_kpool;
                const int MS = (int) I.max_sel;
                if (c.indexer_full[(size_t) il]) {
                    // the selection (build_kpool_select), every query of the chunk at once: the pool scores
                    // sum_h w[h] relu(q_h . pooled), head by head, then each query's top pools
                    const int64_t NP = (t0 + T) / P;                // pools complete by the chunk's last token
                    float* iq = A.take<float>((size_t) (T * IH * ID));
                    float* wts = A.take<float>((size_t) (T * IH));
                    float* acc = A.take<float>((size_t) (T * NP));
                    float* dots = A.take<float>((size_t) (T * NP));
                    G.gemm(l.idx_q_b, qr, T, iq, s);
                    G.gemm(l.idx_proj, d.pxn, T, wts, s);
                    gpu::scale(wts, (int) (T * IH), 1.0f / std::sqrt((float) (ID * IH)), s);
                    cb(cublasSetStream(hb, d.stream), "cublasSetStream");
                    for (int64_t h = 0; h < IH; ++h) {
                        // dots [T][NP] = iq[:, h, :] pooled^T
                        cb(cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, (int) NP, T, (int) ID, &one, l.pooled, (int) ID,
                                       iq + h * ID, (int) (IH * ID), &zero, dots, (int) NP), "pool scores");
                        gpu::pool_scores_accum(dots, wts, (int) h, (int) IH, acc, T, NP, h == 0, s);
                    }
                    gpu::topk_pools_rows(acc, NP, T, t0, (int) P, (int) (c.indexer_top_k / P), c.indexer_select_tail,
                                         MS, d.sel_rows, d.n_sel, s);
                    // a copy for the shared indexer layers on the other devices
                    ok(cudaMemcpyAsync(I.h_sel_rows, d.sel_rows, (size_t) T * MS * sizeof(int32_t),
                                       cudaMemcpyDeviceToHost, d.stream), "selection");
                    ok(cudaMemcpyAsync(I.h_n_sel, d.n_sel, (size_t) T * sizeof(int32_t), cudaMemcpyDeviceToHost,
                                       d.stream), "selection");
                    ok(cudaStreamSynchronize(d.stream), "sync");
                    d.sel_have = ++I.sel_version;
                } else if (d.sel_have != I.sel_version) {
                    // a shared indexer layer on another device than the full one: its selection from the host
                    // (prefill() runs sparse chunks only when a full layer comes first)
                    ok(cudaMemcpyAsync(d.sel_rows, I.h_sel_rows, (size_t) T * MS * sizeof(int32_t),
                                       cudaMemcpyHostToDevice, d.stream), "selection");
                    ok(cudaMemcpyAsync(d.n_sel, I.h_n_sel, (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice,
                                       d.stream), "selection");
                    ok(cudaStreamSynchronize(d.stream), "sync");    // before the next full layer refills them
                    d.sel_have = I.sel_version;
                }
                // attention over each query's own cells, nb queries at a time: gathered K, then the dense path's
                // products batched per query
                const size_t per_q = align256((size_t) MS * KL * 2) + align256((size_t) H * MS * sizeof(float)) +
                                     align256((size_t) H * MS * 2);
                const size_t room = A.cap > A.used + 3 * 256 ? A.cap - A.used - 3 * 256 : 0;
                const int nb = (int) std::min<size_t>((size_t) T, room / per_q);
                if (nb < 1) throw std::runtime_error("prompt chunk arena too small for sparse attention");
                uint16_t* Kg = A.take<uint16_t>((size_t) nb * MS * KL);
                float* Sc = A.take<float>((size_t) nb * H * MS);
                uint16_t* P16 = A.take<uint16_t>((size_t) nb * H * MS);
                cb(cublasSetStream(hb, d.stream), "cublasSetStream");
                for (int b0 = 0; b0 < T; b0 += nb) {
                    const int bn = std::min(nb, T - b0);
                    gpu::gather_cells(l.cache, d.sel_rows + (size_t) b0 * MS, d.n_sel + b0, bn, MS, (int) KL, Kg, s);
                    // S_b [H][MS] = q_b [H][KL] Kg_b^T * scale
                    cb(cublasGemmStridedBatchedEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, MS, (int) H, (int) KL, &scale, Kg,
                                                  CUDA_R_16F, (int) KL, (long long) MS * KL,
                                                  q16 + (size_t) b0 * H * KL, CUDA_R_16F, (int) KL, (long long) H * KL,
                                                  &zero, Sc, CUDA_R_32F, MS, (long long) H * MS, bn,
                                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                       "sparse attention scores");
                    gpu::sparse_softmax_f16(Sc, P16, d.n_sel + b0, bn, (int) H, MS, s);
                    // O_b [H][KL] = P_b [H][MS] Kg_b
                    cb(cublasGemmStridedBatchedEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, (int) KL, (int) H, MS, &one, Kg,
                                                  CUDA_R_16F, (int) KL, (long long) MS * KL, P16, CUDA_R_16F, MS,
                                                  (long long) H * MS, &zero, lat + (size_t) b0 * H * KL, CUDA_R_32F,
                                                  (int) KL, (long long) H * KL, bn, CUBLAS_COMPUTE_32F,
                                                  CUBLAS_GEMM_DEFAULT),
                       "sparse attention values");
                }
            }
            G.gemm_heads(l.v_b, (int) H, lat, (int) (H * KL), KL, T, heads, (int) (H * HV), HV, s);
            G.gemm(l.wo, heads, T, d.py, s);
        }
        gpu::hc_post_rows(d.py, d.pR, d.ppc, d.pR2, T, (int) hc, (int) E, s);
        std::swap(d.pR, d.pR2);

        // ---- FFN half
        gpu::rms_norm_rows(d.pR, d.pflat, T, (int) (hc * E), c.rms_eps, nullptr, s);
        G.gemm(l.hc_ffn_fn, d.pflat, T, d.pmixes, s);
        gpu::hc_weights_rows(d.pmixes, l.hc_ffn_scale, l.hc_ffn_base, d.ppc, T, (int) hc, (int) c.hc_sinkhorn_iters,
                             c.hc_eps, s);
        gpu::hc_mix_rows(d.pR, d.ppc, d.px, T, (int) hc, (int) E, s);
        gpu::rms_norm_rows(d.px, d.pxn, T, (int) E, c.rms_eps, l.ffn_norm, s);

        Bump B{d.arena, d.arena_bytes};
        if (c.is_dense(il)) {
            const int64_t F = c.n_ff;
            float* fg = B.take<float>((size_t) (T * F));
            float* fu = B.take<float>((size_t) (T * F));
            float* fact = B.take<float>((size_t) (T * F));
            G.gemm(l.ffn_gate, d.pxn, T, fg, s);
            G.gemm(l.ffn_up, d.pxn, T, fu, s);
            gpu::swiglu(fg, fu, fact, (int) (T * F), c.clamp_shexp[(size_t) il], s);
            G.gemm(l.ffn_down, fact, T, d.py, s);
        } else {
            const int KU = (int) c.n_expert_used;
            const int64_t F = c.n_ff_exp[(size_t) il], NE = c.n_expert;
            const int64_t FS = F * std::max<int64_t>(c.n_expert_shared, 1);
            const int64_t FM = std::max(F, FS);
            float* fg = B.take<float>((size_t) (T * FM));
            float* fu = B.take<float>((size_t) (T * FM));
            float* fact = B.take<float>((size_t) (T * FM));
            float* logits_e = B.take<float>((size_t) (T * NE));
            float* w_d = B.take<float>((size_t) (T * KU));
            int32_t* ids_d = B.take<int32_t>((size_t) (T * KU));
            float* O = B.take<float>((size_t) (T * KU * E));
            float* ysh = B.take<float>((size_t) (T * E));
            float* Xg = B.take<float>((size_t) (T * E));
            float* ey = B.take<float>((size_t) (T * E));

            G.gemm(l.gate_inp, d.pxn, T, logits_e, s);
            gpu::route_rows(logits_e, l.exp_probs_b, T, (int) NE, KU, c.expert_gating_func, c.expert_weights_norm,
                            c.expert_weights_scale, ids_d, w_d, s);
            I.h_ids_rows.resize((size_t) (T * KU));
            ok(cudaMemcpyAsync(I.h_ids_rows.data(), ids_d, (size_t) (T * KU) * sizeof(int32_t), cudaMemcpyDeviceToHost,
                               d.stream), "routing");
            if (c.n_expert_shared > 0) {
                G.gemm(l.gate_shexp, d.pxn, T, fg, s);
                G.gemm(l.up_shexp, d.pxn, T, fu, s);
                gpu::swiglu(fg, fu, fact, (int) (T * FS), c.clamp_shexp[(size_t) il], s);
                G.gemm(l.down_shexp, fact, T, ysh, s);
            }
            ok(cudaStreamSynchronize(d.stream), "sync");
            for (int t = 0; t < T; ++t) I.cache->record(il, I.h_ids_rows.data() + (size_t) t * KU, KU);

            // the tokens of each expert, and where each result goes (token * KU + its place in the routing)
            std::vector<std::vector<int32_t>> tok_of((size_t) NE), slot_of((size_t) NE);
            for (int t = 0; t < T; ++t)
                for (int j = 0; j < KU; ++j) {
                    const int32_t e = I.h_ids_rows[(size_t) (t * KU + j)];
                    tok_of[(size_t) e].push_back(t);
                    slot_of[(size_t) e].push_back(t * KU + j);
                }
            std::vector<int32_t> idx_all, slot_all;
            std::vector<int64_t> off((size_t) NE + 1, 0);
            for (int64_t e = 0; e < NE; ++e) {
                off[(size_t) e] = (int64_t) idx_all.size();
                idx_all.insert(idx_all.end(), tok_of[(size_t) e].begin(), tok_of[(size_t) e].end());
                slot_all.insert(slot_all.end(), slot_of[(size_t) e].begin(), slot_of[(size_t) e].end());
            }
            off[(size_t) NE] = (int64_t) idx_all.size();
            ok(cudaMemcpy(d.d_idx, idx_all.data(), idx_all.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "idx");
            ok(cudaMemcpy(d.d_slot, slot_all.data(), slot_all.size() * sizeof(int32_t), cudaMemcpyHostToDevice),
               "slots");

            const ExpertSlots& es = I.cache->layer(il);
            if (!es.cacheable) {
                // no GPU kernel for these experts: the CPU computes them, token by token
                std::vector<float> xh((size_t) (T * E)), wh((size_t) (T * KU)), out((size_t) (T * E));
                ok(cudaMemcpy(xh.data(), d.pxn, xh.size() * sizeof(float), cudaMemcpyDeviceToHost), "x");
                ok(cudaMemcpy(wh.data(), w_d, wh.size() * sizeof(float), cudaMemcpyDeviceToHost), "w");
                for (int t = 0; t < T; ++t)
                    I.experts->run(il, xh.data() + (size_t) t * E, I.h_ids_rows.data() + (size_t) t * KU,
                                   wh.data() + (size_t) t * KU, KU, out.data() + (size_t) t * E);
                ok(cudaMemcpy(d.py, out.data(), out.size() * sizeof(float), cudaMemcpyHostToDevice), "experts");
                if (c.n_expert_shared > 0) gpu::add(d.py, ysh, (int) (T * E), s);
            } else {
                int b = 0;
                const LayerWeights& lw = m.layer(il);
                for (int64_t e = 0; e < NE; ++e) {
                    const int mcount = (int) (off[(size_t) e + 1] - off[(size_t) e]);
                    if (mcount == 0) continue;
                    uint8_t* base = nullptr;
                    int slot = I.cache->slot(il, (int) e);
                    bool staged = false, streamed = false;
                    if (slot >= 0) {
                        base = I.cache->slot_ptr(il, slot);
                    } else {
                        // stream it in: into a free cache slot when there is one, else a staging buffer
                        slot = I.cache->claim_free(il, (int) e);
                        staged = slot < 0;
                        base = staged ? d.dstage[b] : I.cache->slot_ptr(il, slot);
                        if (staged) ok(cudaEventSynchronize(d.ev_free[b]), "staging free");
                        ok(cudaEventSynchronize(d.ev_copied[b]), "pinned free");
                        uint8_t* h = I.hstage[b];
                        const Tensor g = lw.gate_exps.slice2(e), u = lw.up_exps.slice2(e), dn = lw.down_exps.slice2(e);
                        std::memcpy(h, g.data, es.up_off);
                        std::memcpy(h + es.up_off, u.data, es.down_off - es.up_off);
                        std::memcpy(h + es.down_off, dn.data, es.down.row_bytes * (size_t) es.down.n_out);
                        ok(cudaMemcpyAsync(base, h, es.down_off + es.down.row_bytes * (size_t) es.down.n_out,
                                           cudaMemcpyHostToDevice, d.copy_stream), "expert upload");
                        ok(cudaEventRecord(d.ev_copied[b], d.copy_stream), "cudaEventRecord");
                        ok(cudaStreamWaitEvent(d.stream, d.ev_copied[b], 0), "cudaStreamWaitEvent");
                        streamed = true;
                    }
                    DevMatrix Gm = es.gate, Um = es.up, Dm = es.down;
                    Gm.w = base;
                    Um.w = base + es.up_off;
                    Dm.w = base + es.down_off;
                    gpu::gather_rows(d.pxn, d.d_idx + off[(size_t) e], Xg, mcount, (int) E, s);
                    G.gemm(Gm, Xg, mcount, fg, s);
                    G.gemm(Um, Xg, mcount, fu, s);
                    gpu::swiglu(fg, fu, fact, (int) (mcount * F), c.clamp_exp[(size_t) il], s);
                    G.gemm(Dm, fact, mcount, ey, s);
                    gpu::scatter_rows(ey, d.d_slot + off[(size_t) e], O, mcount, (int) E, s);
                    if (staged) ok(cudaEventRecord(d.ev_free[b], d.stream), "cudaEventRecord");
                    if (streamed) b ^= 1;     // the other pair of buffers for the next streamed expert
                }
                gpu::combine_experts(O, w_d, c.n_expert_shared > 0 ? ysh : nullptr, d.py, T, KU, (int) E, s);
            }
        }
        gpu::hc_post_rows(d.py, d.pR, d.ppc, d.pR2, T, (int) hc, (int) E, s);
        std::swap(d.pR, d.pR2);
    }

    // ---- the last token's logits
    DevCtx& d = I.ctx[(size_t) I.out_ctx];
    if (cur != I.out_ctx) throw std::runtime_error("the output head is not on the last layer's device");
    ok(cudaSetDevice(d.id), "cudaSetDevice");
    gpu::hc_mean(d.pR + (size_t) (T - 1) * hc * E, d.x, (int) hc, (int) E, d.stream);
    gpu::rms_norm(d.x, d.xn, (int) E, c.rms_eps, I.output_norm, d.stream);
    gpu::gemv(I.output, d.xn, d.out_logits, d.q8, d.stream);
    ok(cudaMemcpyAsync(I.logits.data(), d.out_logits, (size_t) c.n_vocab * sizeof(float), cudaMemcpyDeviceToHost,
                       d.stream), "logits");
    for (auto& dc : I.ctx) {
        ok(cudaSetDevice(dc.id), "cudaSetDevice");
        ok(cudaStreamSynchronize(dc.stream), "sync");
    }
    I.cache->maintain();
    pos_ += T;
    return true;
}

}  // namespace strata::glm
