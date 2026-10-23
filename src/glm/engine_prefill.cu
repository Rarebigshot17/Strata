// src/glm/engine_prefill.cu - GpuEngine's prompt path: T tokens per chunk (engine.hpp).
//
// The same computation as eval() token by token, batched: each product runs once for the chunk (gpu_gemm.hpp),
// the KDA recurrence walks the chunk's tokens in one kernel, and MLA attends with two cuBLAS products over the
// f16 latent cache. Every routed expert the chunk uses is computed once for all of its tokens on the GPU: from
// the VRAM cache when it is there, otherwise streamed in from RAM while the previous one computes (and kept in a
// free cache slot when there is one). Mapped experts go through two pinned buffers, a CPU thread copying each one
// out of the file first; page-locked ones (--pin-experts) are copied by DMA straight from RAM, the host only
// queueing the copies.
//
// While the indexer still selects every pool (the context below top_k + kpool - 1 tokens) attention is plain
// causal over the cache. Past that each query attends its own selection, made as the decode path makes it: the
// chunk's pool scores with one product per indexer head, a top-k per query on the GPU (ties to the lower pool,
// like decode's stable sort), then the query's cells gathered side by side and the same two f16 products as the
// causal case, batched per query.
#include "engine_impl.hpp"

#include "strata/glm/gpu_kernels.hpp"
#include "strata/glm/gpu_prefill.hpp"
#include "strata/glm/nvtx.hpp"

#include "ggml.h"

#include <cublas_v2.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// The attention half's tokens at a time: at most kRowsMax, fewer while its arena would pass kAttnArena (a long
// context's pool scores take 8 bytes per query and pool: 1 MiB a query at 524,288 tokens).
constexpr int kRowsMax = 512;
constexpr size_t kAttnArena = (size_t) 512 << 20;
// --kv-in-ram: the sparse attention's queries a batch, whose cells are read from host memory once (at most
// kKvBatch * max_sel rows, 64 MiB of f16 at 2,051 cells of 512)
constexpr int kKvBatch = 32;
// --prefill-spread: an expert's fixed cost in rows (its weights copied in and dequantized whatever its rows), when
// the streamed experts are shared out by load
constexpr int64_t kExpertRows = 64;

}  // namespace

void GpuEngine::setup_prefill() {
    Impl& I = *impl_;
    const Config& c = I.c;
    const int T = I.opt.prefill_chunk;
    if (T <= 1) return;
    const int64_t E = c.n_embd, hc = c.hc, D = c.kda_head_dim, KL = c.kv_lora_rank, HK = c.head_k_mla,
                  HV = c.head_v_mla, QL = c.q_lora_rank, ID = c.indexer_head_size,
                  P = c.indexer_kpool, NE = c.n_expert, KU = c.n_expert_used;
    int64_t maxH = 0, maxDI = 0, maxF = 0;
    for (int64_t il = 0; il < c.n_layer; ++il) {
        maxH = std::max(maxH, c.n_head[(size_t) il]);
        if (c.is_kda(il)) maxDI = std::max(maxDI, c.n_head[(size_t) il] * D);
        if (!c.is_dense(il)) maxF = std::max(maxF, c.n_ff_exp[(size_t) il]);
    }
    // Everything but the routed experts runs Ta tokens at a time (the attention half, the mixing, a dense FFN):
    // their buffers grow with the context (the dense range's scores, every query's pool scores), the routed experts'
    // with T only, and a larger T spreads the copying of the streamed experts over more tokens. The phases share one
    // arena: the KDA mixer, the MLA mixer, the FFN.
    const size_t f = sizeof(float);
    const int64_t MS = I.max_sel, IH = c.indexer_n_head;
    int64_t FM = 0;                                  // the widest routed or shared expert
    for (int64_t il = 0; il < c.n_layer; ++il)
        if (!c.is_dense(il)) FM = std::max(FM, c.n_ff_exp[(size_t) il] * std::max<int64_t>(c.n_expert_shared, 1));
    FM = std::max(FM, maxF);
    const int64_t n_ff_dense = c.n_layer_dense_lead > 0 ? c.n_ff : 0;
    const auto attn_arena = [&](int64_t r) {
        const size_t kda = align256(8 * r * maxDI * f) + align256(r * D * f) + align256(r * maxH * f) + 8 * 256;
        // MLA: the projections and the indexer's keys, then the attention: over every cell (the dense range), or
        // past it the indexer's queries and pool scores, and per batch of queries their gathered cells, scores and
        // probabilities (at least min(r, 32) queries a batch)
        const size_t mla_common = align256(r * QL * f) + align256(r * maxH * HK * f) + align256(r * KL * f) +
                                  2 * align256(r * maxH * KL * f) + align256(r * maxH * KL * 2) +
                                  align256(r * maxH * HV * f) + align256(r * ID * f) + 2 * align256((P + r) * ID * f) +
                                  8 * 256;
        const size_t dense_att = align256(r * maxH * MS * f) + align256(r * maxH * MS * 2);
        const int64_t nb = std::min<int64_t>(r, 32);
        const size_t sparse_att = align256(r * IH * ID * f) + align256(r * IH * f) + 2 * align256(r * I.max_pools * f) +
                                  align256(nb * MS * KL * 2) + align256(nb * maxH * MS * f) +
                                  align256(nb * maxH * MS * 2) + 8 * 256;
        return std::max(kda, mla_common + std::max(dense_att, sparse_att));
    };
    int Ta = I.opt.prefill_rows > 0 ? std::min(I.opt.prefill_rows, T) : std::min(T, kRowsMax);
    if (I.opt.prefill_rows <= 0)
        while (Ta > 16 && attn_arena(Ta) > kAttnArena) Ta -= 16;
    I.prefill_rows = Ta;
    const size_t dense_ffn = 3 * align256(Ta * n_ff_dense * f) + 3 * 256;
    const size_t moe = 3 * align256(T * FM * f) + align256(T * NE * f) + align256(T * KU * f) +
                       align256(T * KU * sizeof(int32_t)) + align256(T * KU * E * f) + 3 * align256(T * E * f) + 10 * 256;
    const size_t arena = std::max({attn_arena(Ta), dense_ffn, moe});

    // the experts' blob, for the streaming buffers; which layers' experts are page-locked (--pin-experts)
    I.experts_pinned.assign((size_t) c.n_layer, 0);
    for (int64_t il = 0; il < c.n_layer; ++il) {
        if (c.is_dense(il)) continue;
        const LayerWeights& w = I.m.layer(il);
        bool pinned = true;
        for (const Tensor* t : {&w.gate_exps, &w.up_exps, &w.down_exps}) {
            cudaPointerAttributes at{};
            pinned = pinned && cudaPointerGetAttributes(&at, t->data) == cudaSuccess && at.type == cudaMemoryTypeHost;
        }
        (void) cudaGetLastError();   // a pointer CUDA does not know is no error to keep
        I.experts_pinned[(size_t) il] = pinned ? 1 : 0;
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
        d.pflat = dalloc<float>((size_t) (Ta * hc * E), b);
        d.pmixes = dalloc<float>((size_t) (Ta * c.hc_mix_dim()), b);
        d.ppc = dalloc<float>((size_t) (T * (2 * hc + hc * hc)), b);
        d.px = dalloc<float>((size_t) (Ta * E), b);
        d.pxn = dalloc<float>((size_t) (T * E), b);
        d.py = dalloc<float>((size_t) (T * E), b);
        d.arena = dalloc<uint8_t>(arena, b);
        d.arena_bytes = arena;
        d.d_idx = dalloc<int32_t>((size_t) (T * KU), b);
        d.d_slot = dalloc<int32_t>((size_t) (T * KU), b);
        d.sel_rows = dalloc<int32_t>((size_t) (T * MS), b);
        d.n_sel = dalloc<int32_t>((size_t) T, b);
        if (I.opt.kv_in_ram) {
            // a batch of kKvBatch queries' cells, each read from host memory once (cell_union)
            const int64_t cells = I.opt.max_ctx;
            d.kv_flags = dalloc<int32_t>((size_t) cells, b);
            d.kv_pos = dalloc<int32_t>((size_t) cells, b);
            d.kv_U = dalloc<int32_t>((size_t) (kKvBatch * MS), b);
            d.kv_local = dalloc<int32_t>((size_t) (kKvBatch * MS), b);
            d.kv_count = dalloc<int32_t>(1, b);
            d.kv_rows = dalloc<uint16_t>((size_t) (kKvBatch * MS * KL), b);
            d.kv_tmp_bytes = gpu::cell_union_scratch(cells);
            d.kv_tmp = dalloc<uint8_t>(d.kv_tmp_bytes, b);
        }
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
        const size_t act = std::max((size_t) Ta * std::max({hc * E, maxDI, maxH * KL, maxH * HK, maxH * HV, n_ff_dense,
                                                             E, QL}),
                                    (size_t) T * std::max(E, FM));
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
    // --prefill-spread: every card can take a share of a layer's experts; the cards reach each other directly
    // where they can (peer access), otherwise the driver copies through RAM
    I.spread = I.opt.prefill_spread && I.ctx.size() > 1;
    if (I.spread) {
        const size_t nd = I.ctx.size();
        for (size_t di = 0; di < nd; ++di) {
            DevCtx& d = I.ctx[di];
            ok(cudaSetDevice(d.id), "cudaSetDevice");
            size_t* b = &d.bytes;
            cudaDeviceProp pr{};
            ok(cudaGetDeviceProperties(&pr, d.id), "cudaGetDeviceProperties");
            d.sms = pr.multiProcessorCount;
            d.y_part = dalloc<float>((size_t) (T * E), b);
            d.y_recv.assign(nd, nullptr);
            for (size_t k = 0; k < nd; ++k)
                if (k != di) d.y_recv[k] = dalloc<float>((size_t) (T * E), b);
            d.d_wrow = dalloc<float>((size_t) (T * KU), b);
            d.h_idx = halloc<int32_t>((size_t) (T * KU));
            d.h_wrow = halloc<float>((size_t) (T * KU));
            ok(cudaEventCreateWithFlags(&d.ev_x, cudaEventDisableTiming), "cudaEventCreate");
            ok(cudaEventCreateWithFlags(&d.ev_y, cudaEventDisableTiming), "cudaEventCreate");
            for (size_t k = 0; k < nd; ++k) {
                int can = 0;
                if (k == di || cudaDeviceCanAccessPeer(&can, d.id, I.ctx[k].id) != cudaSuccess || !can) continue;
                const cudaError_t e = cudaDeviceEnablePeerAccess(I.ctx[k].id, 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled && I.opt.verbose)
                    std::fprintf(stderr, "prefill spread: peer access %d -> %d: %s\n", d.id, I.ctx[k].id,
                                 cudaGetErrorString(e));
            }
            (void) cudaGetLastError();
        }
    }
    I.h_sel_rows = halloc<int32_t>((size_t) (T * MS));
    I.h_n_sel = halloc<int32_t>((size_t) T);
    if (const char* pv = std::getenv("STRATA_GLM_PROFILE")) {
        long long from = -1;
        int n = 0;
        if (std::sscanf(pv, "%lld,%d", &from, &n) == 2 && from >= 0 && n > 0) {
            I.prof_from = from;
            I.prof_left = n;
            std::fprintf(stderr, "STRATA_GLM_PROFILE: recording %d prompt chunk(s) from position %lld\n", n, from);
        } else {
            std::fprintf(stderr, "STRATA_GLM_PROFILE=%s: expected FROM,N (a position and a number of chunks)\n", pv);
        }
    }
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
    if (I.prof_left > 0 && !I.prof_on && t0 >= I.prof_from) {
        ok(cudaProfilerStart(), "cudaProfilerStart");
        I.prof_on = true;
    }
    char rname[64];
    std::snprintf(rname, sizeof rname, "chunk %lld+%d", (long long) t0, T);
    NvtxRange r_chunk(rname);
    I.cache->drain();   // the decode's queued expert copies land before this chunk claims free slots

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
        ticks_.fetch_add(1, std::memory_order_relaxed);
        const int ci = I.ctx_of_layer[(size_t) il];
        if (ci != cur) {
            NvtxRange r_peer("to the next card");
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

        // ---- attention half, Ta tokens at a time (the KDA state, the latent cache and the pooled keys carry over
        // from one to the next as from chunk to chunk)
        const int64_t PCW = 2 * hc + hc * hc;           // a token's mHC weights (ppc)
        const int Ta = I.prefill_rows;
        const bool mla = !c.is_kda(il);
        std::snprintf(rname, sizeof rname, "L%lld %s", (long long) il, mla ? "MLA" : "KDA");
        nvtx_push(rname);                                   // the attention half, popped before the FFN half
        if (mla && sparse && !c.indexer_full[(size_t) il] && d.sel_have != I.sel_version) {
            // a shared indexer layer on another device than the full one: the chunk's selection from the host
            // (prefill() runs sparse chunks only when a full layer comes first)
            const int MS = (int) I.max_sel;
            ok(cudaMemcpyAsync(d.sel_rows, I.h_sel_rows, (size_t) T * MS * sizeof(int32_t), cudaMemcpyHostToDevice,
                               d.stream), "selection");
            ok(cudaMemcpyAsync(d.n_sel, I.h_n_sel, (size_t) T * sizeof(int32_t), cudaMemcpyHostToDevice, d.stream),
               "selection");
            ok(cudaStreamSynchronize(d.stream), "sync");    // before the next full layer refills them
            d.sel_have = I.sel_version;
        }
        for (int a0 = 0; a0 < T; a0 += Ta) {
            const int ta = std::min(Ta, T - a0);
            const int64_t tp = t0 + a0;                     // the rows' first position
            const float* R = d.pR + (size_t) a0 * hc * E;
            float* pc = d.ppc + (size_t) a0 * PCW;
            float* xn = d.pxn + (size_t) a0 * E;
            float* y = d.py + (size_t) a0 * E;
            gpu::rms_norm_rows(R, d.pflat, ta, (int) (hc * E), c.rms_eps, nullptr, s);
            G.gemm(l.hc_attn_fn, d.pflat, ta, d.pmixes, s);
            gpu::hc_weights_rows(d.pmixes, l.hc_attn_scale, l.hc_attn_base, pc, ta, (int) hc,
                                 (int) c.hc_sinkhorn_iters, c.hc_eps, s);
            gpu::hc_mix_rows(R, pc, d.px, ta, (int) hc, (int) E, s);
            gpu::rms_norm_rows(d.px, xn, ta, (int) E, c.rms_eps, l.attn_norm, s);

            Bump A{d.arena, d.arena_bytes};
            if (c.is_kda(il)) {
                const int64_t D = c.kda_head_dim, DI = H * D, K = c.ssm_d_conv;
                float* raw = A.take<float>((size_t) (ta * DI));
                float* q = A.take<float>((size_t) (ta * DI));
                float* k = A.take<float>((size_t) (ta * DI));
                float* v = A.take<float>((size_t) (ta * DI));
                float* g = A.take<float>((size_t) (ta * DI));
                float* g2 = A.take<float>((size_t) (ta * DI));
                float* o = A.take<float>((size_t) (ta * DI));
                float* gated = A.take<float>((size_t) (ta * DI));
                float* fa = A.take<float>((size_t) (ta * D));
                float* bta = A.take<float>((size_t) (ta * H));
                float* outs[3] = {q, k, v};
                const DevMatrix* proj[3] = {&l.wq, &l.wk, &l.wv};
                for (int j = 0; j < 3; ++j) {
                    G.gemm(*proj[j], xn, ta, raw, s);
                    gpu::kda_conv_rows(raw, l.conv_st[j], l.conv_w[j], outs[j], ta, (int) DI, (int) K, s);
                }
                G.gemm(l.f_a, xn, ta, fa, s);
                G.gemm(l.f_b, fa, ta, g, s);
                gpu::kda_gate_rows(g, l.dt_b, l.a, ta, (int) H, (int) D, c.has_gate_lower_bound, c.kda_gate_lower_bound,
                                   s);
                G.gemm(l.beta, xn, ta, bta, s);
                gpu::sigmoid_inplace(bta, (int) (ta * H), s);
                gpu::l2_norm_heads(q, (int) (ta * H), (int) D, s);
                gpu::l2_norm_heads(k, (int) (ta * H), (int) D, s);
                gpu::kda_scan(l.S, q, k, v, g, bta, o, ta, (int) H, (int) D, s);
                G.gemm(l.g_a, xn, ta, fa, s);
                G.gemm(l.g_b, fa, ta, g2, s);
                gpu::kda_out_gate(o, l.o_norm, g2, gated, (int) (ta * H), (int) D, c.rms_eps, s);
                G.gemm(l.wo, gated, ta, y, s);
            } else {
                const int64_t QL = c.q_lora_rank, KL = c.kv_lora_rank, HK = c.head_k_mla, HV = c.head_v_mla;
                float* qr = A.take<float>((size_t) (ta * QL));
                float* qm = A.take<float>((size_t) (ta * H * HK));
                float* kv = A.take<float>((size_t) (ta * KL));
                float* q_abs = A.take<float>((size_t) (ta * H * KL));
                uint16_t* q16 = A.take<uint16_t>((size_t) (ta * H * KL));
                float* lat = A.take<float>((size_t) (ta * H * KL));
                float* heads = A.take<float>((size_t) (ta * H * HV));
                G.gemm(l.q_a, xn, ta, qr, s);
                gpu::rms_norm_rows(qr, qr, ta, (int) QL, c.rms_eps, l.q_a_norm, s);
                G.gemm(l.q_b, qr, ta, qm, s);
                G.gemm(l.kv_a, xn, ta, kv, s);
                gpu::rms_norm_rows(kv, kv, ta, (int) KL, c.rms_eps, l.kv_a_norm, s);
                gpu::store_f16_rows(kv, l.cache, tp, ta, (int) KL, s);
                if (l.head && tp < I.max_sel)                // --kv-in-ram: the dense range's rows in VRAM too
                    gpu::store_f16_rows(kv, l.head, tp, (int) std::min<int64_t>(ta, I.max_sel - tp), (int) KL, s);
                G.gemm_heads(l.k_b, (int) H, qm, (int) (H * HK), HK, ta, q_abs, (int) (H * KL), KL, s);
                gpu::to_f16(q_abs, q16, ta * H * KL, s);

                if (c.indexer_full[(size_t) il]) {
                    // the pooled keys later tokens' selections read: the pool being filled carries over chunks
                    const int64_t ID = c.indexer_head_size, P = c.indexer_kpool, r = tp % P;
                    float* ik = A.take<float>((size_t) (ta * ID));
                    float* kb = A.take<float>((size_t) ((P + ta) * ID));
                    float* gb = A.take<float>((size_t) ((P + ta) * ID));
                    if (r) {
                        ok(cudaMemcpyAsync(kb, l.pend_k, (size_t) (r * ID) * sizeof(float), cudaMemcpyDeviceToDevice,
                                           d.stream), "pending keys");
                        ok(cudaMemcpyAsync(gb, l.pend_g, (size_t) (r * ID) * sizeof(float), cudaMemcpyDeviceToDevice,
                                           d.stream), "pending gates");
                    }
                    G.gemm(l.idx_k, xn, ta, ik, s);
                    gpu::layer_norm_rows(ik, kb + r * ID, ta, (int) ID, c.norm_eps, l.idx_k_norm, l.idx_k_norm_b, s);
                    G.gemm(l.idx_gate, xn, ta, gb + r * ID, s);
                    const int64_t n_done = (r + ta) / P, rem = (r + ta) % P;
                    gpu::kpool_rows(kb, gb, l.idx_ape, l.pooled, (tp - r) / P, (int) n_done, 0, (int) P, (int) ID, s);
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
                    const int n_ctx = (int) (tp + ta);
                    // --kv-in-ram: the dense range is in the VRAM head (prefill() ends dense chunks at max_sel)
                    const uint16_t* kc = l.head ? l.head : l.cache;
                    if (l.head && n_ctx > I.max_sel) throw std::runtime_error("a dense prompt chunk past the head");
                    float* Sc = A.take<float>((size_t) (ta * H) * (size_t) n_ctx);
                    uint16_t* P16 = A.take<uint16_t>((size_t) (ta * H) * (size_t) n_ctx);
                    cb(cublasSetStream(hb, d.stream), "cublasSetStream");
                    cb(cublasGemmEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, n_ctx, (int) (ta * H), (int) KL, &scale, kc,
                                    CUDA_R_16F, (int) KL, q16, CUDA_R_16F, (int) KL, &zero, Sc, CUDA_R_32F, n_ctx,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "attention scores");
                    gpu::causal_softmax_f16(Sc, P16, ta, (int) H, n_ctx, tp, s);
                    cb(cublasGemmEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, (int) KL, (int) (ta * H), n_ctx, &one, kc,
                                    CUDA_R_16F, (int) KL, P16, CUDA_R_16F, n_ctx, &zero, lat, CUDA_R_32F, (int) KL,
                                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "attention values");
                } else {
                    const int64_t IH = c.indexer_n_head, ID = c.indexer_head_size, P = c.indexer_kpool;
                    const int MS = (int) I.max_sel;
                    if (c.indexer_full[(size_t) il]) {
                        // the selection (build_kpool_select), every query of the rows at once: the pool scores
                        // sum_h w[h] relu(q_h . pooled), head by head, then each query's top pools
                        const int64_t NP = (tp + ta) / P;                // pools complete by the rows' last token
                        float* iq = A.take<float>((size_t) (ta * IH * ID));
                        float* wts = A.take<float>((size_t) (ta * IH));
                        float* acc = A.take<float>((size_t) (ta * NP));
                        float* dots = A.take<float>((size_t) (ta * NP));
                        G.gemm(l.idx_q_b, qr, ta, iq, s);
                        G.gemm(l.idx_proj, xn, ta, wts, s);
                        gpu::scale(wts, (int) (ta * IH), 1.0f / std::sqrt((float) (ID * IH)), s);
                        cb(cublasSetStream(hb, d.stream), "cublasSetStream");
                        for (int64_t h = 0; h < IH; ++h) {
                            // dots [ta][NP] = iq[:, h, :] pooled^T
                            cb(cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, (int) NP, ta, (int) ID, &one, l.pooled,
                                           (int) ID, iq + h * ID, (int) (IH * ID), &zero, dots, (int) NP),
                               "pool scores");
                            gpu::pool_scores_accum(dots, wts, (int) h, (int) IH, acc, ta, NP, h == 0, s);
                        }
                        gpu::topk_pools_rows(acc, NP, ta, tp, (int) P, (int) (c.indexer_top_k / P),
                                             c.indexer_select_tail, MS, d.sel_rows + (size_t) a0 * MS, d.n_sel + a0, s);
                    }
                    // attention over each query's own cells, nb queries at a time: gathered K, then the dense path's
                    // products batched per query
                    const size_t per_q = align256((size_t) MS * KL * 2) + align256((size_t) H * MS * sizeof(float)) +
                                         align256((size_t) H * MS * 2);
                    const size_t room = A.cap > A.used + 3 * 256 ? A.cap - A.used - 3 * 256 : 0;
                    const int nb = (int) std::min<size_t>({(size_t) ta, room / per_q,
                                                           l.head ? (size_t) kKvBatch : (size_t) ta});
                    if (nb < 1) throw std::runtime_error("prompt chunk arena too small for sparse attention");
                    uint16_t* Kg = A.take<uint16_t>((size_t) nb * MS * KL);
                    float* Sc = A.take<float>((size_t) nb * H * MS);
                    uint16_t* P16 = A.take<uint16_t>((size_t) nb * H * MS);
                    cb(cublasSetStream(hb, d.stream), "cublasSetStream");
                    for (int b0 = 0; b0 < ta; b0 += nb) {
                        const int bn = std::min(nb, ta - b0);
                        const int32_t* qsel = d.sel_rows + (size_t) (a0 + b0) * MS;
                        if (l.head) {
                            // --kv-in-ram: the batch's cells, each once, over PCIe into kv_rows, then each query's
                            // from there
                            gpu::cell_union(qsel, d.n_sel + a0 + b0, bn, MS, tp + ta, d.kv_flags, d.kv_pos, d.kv_tmp,
                                            d.kv_tmp_bytes, d.kv_U, d.kv_count, d.kv_local, s);
                            gpu::gather_host_rows(l.cache, d.kv_U, d.kv_count, bn * MS, (int) KL, d.kv_rows, s);
                            gpu::gather_cells(d.kv_rows, d.kv_local, d.n_sel + a0 + b0, bn, MS, (int) KL, Kg, s);
                        } else {
                            gpu::gather_cells(l.cache, qsel, d.n_sel + a0 + b0, bn, MS, (int) KL, Kg, s);
                        }
                        // S_b [H][MS] = q_b [H][KL] Kg_b^T * scale
                        cb(cublasGemmStridedBatchedEx(hb, CUBLAS_OP_T, CUBLAS_OP_N, MS, (int) H, (int) KL, &scale, Kg,
                                                      CUDA_R_16F, (int) KL, (long long) MS * KL,
                                                      q16 + (size_t) b0 * H * KL, CUDA_R_16F, (int) KL,
                                                      (long long) H * KL,
                                                      &zero, Sc, CUDA_R_32F, MS, (long long) H * MS, bn,
                                                      CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
                           "sparse attention scores");
                        gpu::sparse_softmax_f16(Sc, P16, d.n_sel + a0 + b0, bn, (int) H, MS, s);
                        // O_b [H][KL] = P_b [H][MS] Kg_b
                        cb(cublasGemmStridedBatchedEx(hb, CUBLAS_OP_N, CUBLAS_OP_N, (int) KL, (int) H, MS, &one, Kg,
                                                      CUDA_R_16F, (int) KL, (long long) MS * KL, P16, CUDA_R_16F, MS,
                                                      (long long) H * MS, &zero, lat + (size_t) b0 * H * KL,
                                                      CUDA_R_32F,
                                                      (int) KL, (long long) H * KL, bn, CUBLAS_COMPUTE_32F,
                                                      CUBLAS_GEMM_DEFAULT),
                           "sparse attention values");
                    }
                }
                G.gemm_heads(l.v_b, (int) H, lat, (int) (H * KL), KL, ta, heads, (int) (H * HV), HV, s);
                G.gemm(l.wo, heads, ta, y, s);
            }
        }
        if (mla && sparse && c.indexer_full[(size_t) il]) {
            // the chunk's selection, for the shared indexer layers on the other devices
            const int MS = (int) I.max_sel;
            ok(cudaMemcpyAsync(I.h_sel_rows, d.sel_rows, (size_t) T * MS * sizeof(int32_t), cudaMemcpyDeviceToHost,
                               d.stream), "selection");
            ok(cudaMemcpyAsync(I.h_n_sel, d.n_sel, (size_t) T * sizeof(int32_t), cudaMemcpyDeviceToHost, d.stream),
               "selection");
            ok(cudaStreamSynchronize(d.stream), "sync");
            d.sel_have = ++I.sel_version;
        }
        gpu::hc_post_rows(d.py, d.pR, d.ppc, d.pR2, T, (int) hc, (int) E, s);
        std::swap(d.pR, d.pR2);
        nvtx_pop();

        // ---- FFN half: the mixing and a dense FFN Ta tokens at a time, the routed experts for the whole chunk
        std::snprintf(rname, sizeof rname, "L%lld FFN", (long long) il);
        NvtxRange r_ffn(rname);
        for (int a0 = 0; a0 < T; a0 += Ta) {
            const int ta = std::min(Ta, T - a0);
            const float* R = d.pR + (size_t) a0 * hc * E;
            float* pc = d.ppc + (size_t) a0 * PCW;
            float* xn = d.pxn + (size_t) a0 * E;
            gpu::rms_norm_rows(R, d.pflat, ta, (int) (hc * E), c.rms_eps, nullptr, s);
            G.gemm(l.hc_ffn_fn, d.pflat, ta, d.pmixes, s);
            gpu::hc_weights_rows(d.pmixes, l.hc_ffn_scale, l.hc_ffn_base, pc, ta, (int) hc, (int) c.hc_sinkhorn_iters,
                                 c.hc_eps, s);
            gpu::hc_mix_rows(R, pc, d.px, ta, (int) hc, (int) E, s);
            gpu::rms_norm_rows(d.px, xn, ta, (int) E, c.rms_eps, l.ffn_norm, s);
            if (c.is_dense(il)) {
                const int64_t F = c.n_ff;
                Bump D{d.arena, d.arena_bytes};
                float* fg = D.take<float>((size_t) (ta * F));
                float* fu = D.take<float>((size_t) (ta * F));
                float* fact = D.take<float>((size_t) (ta * F));
                G.gemm(l.ffn_gate, xn, ta, fg, s);
                G.gemm(l.ffn_up, xn, ta, fu, s);
                gpu::swiglu(fg, fu, fact, (int) (ta * F), c.clamp_shexp[(size_t) il], s);
                G.gemm(l.ffn_down, fact, ta, d.py + (size_t) a0 * E, s);
            }
        }

        Bump B{d.arena, d.arena_bytes};
        if (!c.is_dense(il)) {
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
            if (I.spread) {                                 // the helpers' rows carry their routing weights
                I.h_w_rows.resize((size_t) (T * KU));
                ok(cudaMemcpyAsync(I.h_w_rows.data(), w_d, (size_t) (T * KU) * sizeof(float), cudaMemcpyDeviceToHost,
                                   d.stream), "routing weights");
            }
            if (c.n_expert_shared > 0) {
                G.gemm(l.gate_shexp, d.pxn, T, fg, s);
                G.gemm(l.up_shexp, d.pxn, T, fu, s);
                gpu::swiglu(fg, fu, fact, (int) (T * FS), c.clamp_shexp[(size_t) il], s);
                G.gemm(l.down_shexp, fact, T, ysh, s);
            }
            {
                NvtxRange r_route("routing: wait for the GPU");
                ok(cudaStreamSynchronize(d.stream), "sync");
            }
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
                NvtxRange r_exp("routed experts: queue");
                int b = 0;
                const LayerWeights& lw = m.layer(il);
                // --prefill-spread (page-locked experts only): this card computes the experts its cache holds; each
                // streamed one goes to the card least loaded so far (rows plus kExpertRows, over its multiprocessors),
                // the biggest first. The other cards stream theirs in from RAM themselves and each send back the
                // weighted sum of their experts' outputs, which is added after this card's own.
                std::vector<int> dev_of((size_t) NE, ci);
                std::vector<int> helpers;
                if (I.spread && I.experts_pinned[(size_t) il]) {
                    std::vector<double> load(I.ctx.size(), 0.0);
                    std::vector<int64_t> streamed;
                    for (int64_t e = 0; e < NE; ++e) {
                        const int64_t mc = off[(size_t) e + 1] - off[(size_t) e];
                        if (mc == 0) continue;
                        if (I.cache->slot(il, (int) e) >= 0) load[(size_t) ci] += (double) (mc + kExpertRows);
                        else streamed.push_back(e);
                    }
                    std::stable_sort(streamed.begin(), streamed.end(), [&](int64_t x, int64_t y) {
                        return off[(size_t) x + 1] - off[(size_t) x] > off[(size_t) y + 1] - off[(size_t) y];
                    });
                    for (int64_t e : streamed) {
                        const double w = (double) (off[(size_t) e + 1] - off[(size_t) e] + kExpertRows);
                        size_t best = (size_t) ci;
                        for (size_t k = 0; k < I.ctx.size(); ++k)
                            if ((load[k] + w) / I.ctx[k].sms < (load[best] + w) / I.ctx[best].sms) best = k;
                        dev_of[(size_t) e] = (int) best;
                        load[best] += w;
                    }
                    for (size_t k = 0; k < I.ctx.size(); ++k)
                        if ((int) k != ci && load[k] > 0) helpers.push_back((int) k);
                }
                if (!helpers.empty()) {
                    NvtxRange r_help("routed experts: the other cards' shares");
                    // the chunk's rows to each helper (card to card where they have peer access, else through RAM)
                    for (int k : helpers)
                        ok(cudaMemcpyPeerAsync(I.ctx[(size_t) k].pxn, I.ctx[(size_t) k].id, d.pxn, d.id,
                                               (size_t) (T * E) * sizeof(float), d.stream), "rows to a helper");
                    ok(cudaEventRecord(d.ev_x, d.stream), "cudaEventRecord");
                    for (int k : helpers) {
                        DevCtx& h = I.ctx[(size_t) k];
                        ok(cudaSetDevice(h.id), "cudaSetDevice");
                        void* hs = h.stream;
                        // the share's rows grouped by expert: their tokens and routing weights, and each expert's start
                        std::vector<int64_t> mine, hoff;
                        int64_t n = 0;
                        for (int64_t e = 0; e < NE; ++e) {
                            if (dev_of[(size_t) e] != k) continue;
                            mine.push_back(e);
                            hoff.push_back(n);
                            for (int64_t r = off[(size_t) e]; r < off[(size_t) e + 1]; ++r, ++n) {
                                h.h_idx[n] = idx_all[(size_t) r];
                                h.h_wrow[n] = I.h_w_rows[(size_t) slot_all[(size_t) r]];
                            }
                        }
                        ok(cudaStreamWaitEvent(h.stream, d.ev_x, 0), "cudaStreamWaitEvent");
                        ok(cudaMemcpyAsync(h.d_idx, h.h_idx, (size_t) n * sizeof(int32_t), cudaMemcpyHostToDevice,
                                           h.stream), "share rows");
                        ok(cudaMemcpyAsync(h.d_wrow, h.h_wrow, (size_t) n * sizeof(float), cudaMemcpyHostToDevice,
                                           h.stream), "share weights");
                        ok(cudaMemsetAsync(h.y_part, 0, (size_t) (T * E) * sizeof(float), h.stream), "memset");
                        Bump H{h.arena, h.arena_bytes};
                        float* hfg = H.take<float>((size_t) (T * F));
                        float* hfu = H.take<float>((size_t) (T * F));
                        float* hfact = H.take<float>((size_t) (T * F));
                        float* hXg = H.take<float>((size_t) (T * E));
                        float* hey = H.take<float>((size_t) (T * E));
                        gpu::GemmCtx& HG = *h.gemm;
                        int hb = 0;
                        for (size_t q = 0; q < mine.size(); ++q) {
                            const int64_t e = mine[q];
                            const int mc = (int) (off[(size_t) e + 1] - off[(size_t) e]);
                            uint8_t* base = h.dstage[hb];
                            const Tensor g = lw.gate_exps.slice2(e), u = lw.up_exps.slice2(e),
                                         dn = lw.down_exps.slice2(e);
                            const size_t db = es.down.row_bytes * (size_t) es.down.n_out;
                            ok(cudaStreamWaitEvent(h.copy_stream, h.ev_free[hb], 0), "cudaStreamWaitEvent");
                            ok(cudaMemcpyAsync(base, g.data, es.up_off, cudaMemcpyHostToDevice, h.copy_stream),
                               "expert upload");
                            ok(cudaMemcpyAsync(base + es.up_off, u.data, es.down_off - es.up_off,
                                               cudaMemcpyHostToDevice, h.copy_stream), "expert upload");
                            ok(cudaMemcpyAsync(base + es.down_off, dn.data, db, cudaMemcpyHostToDevice, h.copy_stream),
                               "expert upload");
                            ok(cudaEventRecord(h.ev_copied[hb], h.copy_stream), "cudaEventRecord");
                            ok(cudaStreamWaitEvent(h.stream, h.ev_copied[hb], 0), "cudaStreamWaitEvent");
                            DevMatrix Gm = es.gate, Um = es.up, Dm = es.down;
                            Gm.w = base;
                            Um.w = base + es.up_off;
                            Dm.w = base + es.down_off;
                            gpu::gather_rows(h.pxn, h.d_idx + hoff[q], hXg, mc, (int) E, hs);
                            HG.gemm(Gm, hXg, mc, hfg, hs);
                            HG.gemm(Um, hXg, mc, hfu, hs);
                            gpu::swiglu(hfg, hfu, hfact, (int) (mc * F), c.clamp_exp[(size_t) il], hs);
                            HG.gemm(Dm, hfact, mc, hey, hs);
                            gpu::scatter_add_w_rows(hey, h.d_idx + hoff[q], h.d_wrow + hoff[q], h.y_part, mc, (int) E,
                                                    hs);
                            ok(cudaEventRecord(h.ev_free[hb], h.stream), "cudaEventRecord");
                            hb ^= 1;
                        }
                        ok(cudaMemcpyPeerAsync(d.y_recv[(size_t) k], d.id, h.y_part, h.id,
                                               (size_t) (T * E) * sizeof(float), h.stream), "partial sum back");
                        ok(cudaEventRecord(h.ev_y, h.stream), "cudaEventRecord");
                    }
                    ok(cudaSetDevice(d.id), "cudaSetDevice");
                    // the helpers' slots stay zero in O (the arena holds other phases' bytes)
                    ok(cudaMemsetAsync(O, 0, (size_t) (T * KU * E) * sizeof(float), d.stream), "memset");
                }
                for (int64_t e = 0; e < NE; ++e) {
                    const int mcount = (int) (off[(size_t) e + 1] - off[(size_t) e]);
                    if (mcount == 0 || dev_of[(size_t) e] != ci) continue;
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
                        const Tensor g = lw.gate_exps.slice2(e), u = lw.up_exps.slice2(e), dn = lw.down_exps.slice2(e);
                        const size_t db = es.down.row_bytes * (size_t) es.down.n_out;
                        if (I.experts_pinned[(size_t) il]) {
                            // page-locked: straight from RAM, after the expert that last used this buffer
                            if (staged) ok(cudaStreamWaitEvent(d.copy_stream, d.ev_free[b], 0), "cudaStreamWaitEvent");
                            ok(cudaMemcpyAsync(base, g.data, es.up_off, cudaMemcpyHostToDevice, d.copy_stream),
                               "expert upload");
                            ok(cudaMemcpyAsync(base + es.up_off, u.data, es.down_off - es.up_off,
                                               cudaMemcpyHostToDevice, d.copy_stream), "expert upload");
                            ok(cudaMemcpyAsync(base + es.down_off, dn.data, db, cudaMemcpyHostToDevice, d.copy_stream),
                               "expert upload");
                        } else {
                            NvtxRange r_stage("expert: wait, copy out of the file");
                            if (staged) ok(cudaEventSynchronize(d.ev_free[b]), "staging free");
                            ok(cudaEventSynchronize(d.ev_copied[b]), "pinned free");
                            uint8_t* h = I.hstage[b];
                            std::memcpy(h, g.data, es.up_off);
                            std::memcpy(h + es.up_off, u.data, es.down_off - es.up_off);
                            std::memcpy(h + es.down_off, dn.data, db);
                            ok(cudaMemcpyAsync(base, h, es.down_off + db, cudaMemcpyHostToDevice, d.copy_stream),
                               "expert upload");
                        }
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
                for (int k : helpers) {
                    ok(cudaStreamWaitEvent(d.stream, I.ctx[(size_t) k].ev_y, 0), "cudaStreamWaitEvent");
                    gpu::add(d.py, d.y_recv[(size_t) k], (int) (T * E), s);
                }
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
    {
        NvtxRange r_end("chunk end: wait for every card");
        for (auto& dc : I.ctx) {
            ok(cudaSetDevice(dc.id), "cudaSetDevice");
            ok(cudaStreamSynchronize(dc.stream), "sync");
        }
    }
    I.cache->maintain();
    pos_ += T;
    if (I.prof_on && --I.prof_left == 0) {
        ok(cudaProfilerStop(), "cudaProfilerStop");
        I.prof_on = false;
        std::fprintf(stderr, "STRATA_GLM_PROFILE: stopped after the chunk at %lld\n", (long long) t0);
    }
    return true;
}

}  // namespace strata::glm
