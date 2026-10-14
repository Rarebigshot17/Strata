// src/glm/cpu_ref.cpp - GLM-5.3-Flash on the CPU, one token at a time (cpu_ref.hpp).
//
// Each step names the llama.cpp graph node it reproduces (src/models/glm5-next.cpp), so a mismatch found with
// `llama-debug --tensor-filter` points at one block of code here.
#include "strata/glm/cpu_ref.hpp"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>

namespace strata::glm {

// ================================ thread pool ================================

struct ThreadPool::Impl {
    std::vector<std::thread> workers;
    std::mutex mu;
    std::condition_variable cv_start, cv_done;
    const std::function<void(int64_t, int64_t)>* fn = nullptr;
    int64_t n = 0;
    uint64_t generation = 0;
    int pending = 0;
    bool stop = false;
};

ThreadPool::ThreadPool(int n_threads) : impl_(new Impl), n_(std::max(1, n_threads)) {
    for (int t = 1; t < n_; ++t) {
        impl_->workers.emplace_back([this, t] {
            uint64_t seen = 0;
            for (;;) {
                const std::function<void(int64_t, int64_t)>* fn;
                int64_t n;
                {
                    std::unique_lock<std::mutex> lk(impl_->mu);
                    impl_->cv_start.wait(lk, [&] { return impl_->stop || impl_->generation != seen; });
                    if (impl_->stop) return;
                    seen = impl_->generation;
                    fn = impl_->fn;
                    n = impl_->n;
                }
                const int64_t chunk = (n + n_ - 1) / n_;
                const int64_t b = std::min(n, chunk * t), e = std::min(n, b + chunk);
                if (b < e) (*fn)(b, e);
                std::lock_guard<std::mutex> lk(impl_->mu);
                if (--impl_->pending == 0) impl_->cv_done.notify_one();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->stop = true;
    }
    impl_->cv_start.notify_all();
    for (auto& w : impl_->workers) w.join();
}

void ThreadPool::parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn) {
    if (n <= 0) return;
    if (n_ == 1 || n < 2 * n_) {
        fn(0, n);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(impl_->mu);
        impl_->fn = &fn;
        impl_->n = n;
        impl_->pending = n_ - 1;
        ++impl_->generation;
    }
    impl_->cv_start.notify_all();
    const int64_t chunk = (n + n_ - 1) / n_;
    fn(0, std::min(n, chunk));
    std::unique_lock<std::mutex> lk(impl_->mu);
    impl_->cv_done.wait(lk, [&] { return impl_->pending == 0; });
}

// ================================ small helpers ================================

namespace {

float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
float siluf(float x) { return x / (1.0f + std::exp(-x)); }
float softplusf(float x) { return x > 20.0f ? x : std::log1p(std::exp(x)); }   // ggml_softplus' threshold

/// ggml_rms_norm: x / sqrt(mean(x^2) + eps), then an optional elementwise weight.
void rms_norm(const float* x, float* y, int64_t n, float eps, const float* w = nullptr) {
    double sum = 0;
    for (int64_t i = 0; i < n; ++i) sum += (double) x[i] * x[i];
    const float scale = 1.0f / std::sqrt((float) (sum / (double) n) + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = x[i] * scale * (w ? w[i] : 1.0f);
}

/// ggml_norm (LayerNorm without its affine part), then weight and bias.
void layer_norm(const float* x, float* y, int64_t n, float eps, const float* w, const float* b) {
    double mean = 0;
    for (int64_t i = 0; i < n; ++i) mean += x[i];
    mean /= (double) n;
    double var = 0;
    for (int64_t i = 0; i < n; ++i) var += ((double) x[i] - mean) * ((double) x[i] - mean);
    var /= (double) n;
    const float scale = 1.0f / std::sqrt((float) var + eps);
    for (int64_t i = 0; i < n; ++i) y[i] = (float) (x[i] - mean) * scale * w[i] + b[i];
}

/// A small tensor (a norm weight, a bias, a gate scale) as floats.
std::vector<float> to_f32(const Tensor& t) {
    std::vector<float> out((size_t) t.elements());
    if (t.type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t.data, out.size() * sizeof(float));
        return out;
    }
    const auto* tr = ggml_get_type_traits((ggml_type) t.type);
    if (!tr || !tr->to_float) throw std::runtime_error("cannot dequantize tensor " + t.name);
    for (int64_t r = 0; r < t.rows(); ++r) tr->to_float(t.row(r), out.data() + r * t.ne[0], t.ne[0]);
    return out;
}

float round_f16(float x) { return ggml_fp16_to_fp32(ggml_fp32_to_fp16(x)); }

}  // namespace

// ================================ state ================================

struct CpuRef::LayerState {
    // KDA: the recurrent state S[h][i][j] (i = key channel, j = value channel) and the three conv windows
    std::vector<float> S;
    std::vector<float> conv[3];          ///< (d_conv - 1) x d_inner, oldest row first
    // MLA: the normalized latent per cell (rounded to f16, as llama.cpp's default K cache stores it)
    std::vector<float> latent;
    // indexer: per-cell LayerNormed key and pool gate, and one pooled key per complete pool
    std::vector<float> idx_key, idx_gate, pooled;
    // dequantized small weights
    std::vector<float> attn_norm, ffn_norm, hc_attn_base, hc_attn_scale, hc_ffn_base, hc_ffn_scale;
    std::vector<float> a, dt_b, o_norm, conv_w[3];
    std::vector<float> q_a_norm, kv_a_norm, idx_k_norm, idx_k_norm_b, idx_ape;
    std::vector<float> exp_probs_b;
};

CpuRef::CpuRef(const Model& model, int64_t max_ctx, int n_threads)
    : m_(model), c_(model.config()), max_ctx_(max_ctx), pool_(new ThreadPool(n_threads)) {
    ggml_cpu_init();
    for (int64_t il = 0; il < c_.n_layer; ++il) {
        auto s = std::make_unique<LayerState>();
        const LayerWeights& w = m_.layer(il);
        s->attn_norm = to_f32(w.attn_norm);
        s->ffn_norm = to_f32(w.ffn_norm);
        s->hc_attn_base = to_f32(w.hc_attn_base);
        s->hc_attn_scale = to_f32(w.hc_attn_scale);
        s->hc_ffn_base = to_f32(w.hc_ffn_base);
        s->hc_ffn_scale = to_f32(w.hc_ffn_scale);
        if (c_.is_kda(il)) {
            s->a = to_f32(w.kda.a);
            s->dt_b = to_f32(w.kda.dt_b);
            s->o_norm = to_f32(w.kda.o_norm);
            s->conv_w[0] = to_f32(w.kda.conv_q);
            s->conv_w[1] = to_f32(w.kda.conv_k);
            s->conv_w[2] = to_f32(w.kda.conv_v);
        } else {
            s->q_a_norm = to_f32(w.mla.q_a_norm);
            s->kv_a_norm = to_f32(w.mla.kv_a_norm);
            if (c_.indexer_full[(size_t) il]) {
                s->idx_k_norm = to_f32(w.mla.idx_k_norm);
                s->idx_k_norm_b = to_f32(w.mla.idx_k_norm_b);
                s->idx_ape = to_f32(w.mla.idx_kpool_ape);
            }
        }
        if (!c_.is_dense(il)) s->exp_probs_b = to_f32(w.exp_probs_b);
        st_.push_back(std::move(s));
    }
    reset();
}

CpuRef::~CpuRef() = default;

void CpuRef::reset() {
    pos_ = 0;
    for (int64_t il = 0; il < c_.n_layer; ++il) {
        LayerState& s = *st_[(size_t) il];
        if (c_.is_kda(il)) {
            const int64_t H = c_.n_head[(size_t) il], D = c_.kda_head_dim;
            s.S.assign((size_t) (H * D * D), 0.0f);
            for (auto& cv : s.conv) cv.assign((size_t) ((c_.ssm_d_conv - 1) * H * D), 0.0f);
        } else {
            s.latent.assign((size_t) (max_ctx_ * c_.kv_lora_rank), 0.0f);
            if (c_.indexer_full[(size_t) il]) {
                const int64_t ID = c_.indexer_head_size;
                s.idx_key.assign((size_t) (max_ctx_ * ID), 0.0f);
                s.idx_gate.assign((size_t) (max_ctx_ * ID), 0.0f);
                s.pooled.assign((size_t) ((max_ctx_ / c_.indexer_kpool + 1) * ID), 0.0f);
            }
        }
    }
}

// ================================ products ================================

void CpuRef::matvec(const Tensor& w, const float* x, float* y) {
    const int64_t n_in = w.ne[0], n_out = w.rows();
    const ggml_type wt = (ggml_type) w.type;
    if (wt == GGML_TYPE_F32) {
        pool_->parallel_for(n_out, [&](int64_t b, int64_t e) {
            for (int64_t r = b; r < e; ++r) {
                const float* row = reinterpret_cast<const float*>(w.row(r));
                float sum = 0.0f;
                ggml_get_type_traits_cpu(GGML_TYPE_F32)->vec_dot((int) n_in, &sum, 0, row, 0, x, 0, 1);
                y[r] = sum;
            }
        });
        return;
    }
    const auto* tr = ggml_get_type_traits_cpu(wt);
    if (!tr || !tr->vec_dot) throw std::runtime_error("ggml-cpu has no dot product for tensor " + w.name);
    const ggml_type at = tr->vec_dot_type;
    std::vector<uint8_t> xq(ggml_row_size(at, n_in));
    if (at == GGML_TYPE_F32) std::memcpy(xq.data(), x, (size_t) n_in * sizeof(float));
    else ggml_get_type_traits_cpu(at)->from_float(x, xq.data(), n_in);
    pool_->parallel_for(n_out, [&](int64_t b, int64_t e) {
        for (int64_t r = b; r < e; ++r) {
            float sum = 0.0f;
            tr->vec_dot((int) n_in, &sum, 0, w.row(r), 0, xq.data(), 0, 1);
            y[r] = sum;
        }
    });
}

// ================================ mHC ================================

// build_hc_pre (the 6-argument one): the streams' mixing weights and the mixer's input.
void CpuRef::hc_pre(const Tensor& fn, const Tensor& scale_t, const Tensor& base_t, const float* R, float* x,
                    float* post, float* comb) {
    const int64_t hc = c_.hc, E = c_.n_embd;
    std::vector<float> flat((size_t) (hc * E)), mixes((size_t) c_.hc_mix_dim());
    rms_norm(R, flat.data(), hc * E, c_.rms_eps);
    matvec(fn, flat.data(), mixes.data());                                    // hc_mixes
    const std::vector<float> scale = to_f32(scale_t), base = to_f32(base_t);
    std::vector<float> pre((size_t) hc);
    for (int64_t h = 0; h < hc; ++h) {
        pre[(size_t) h] = sigmoidf(mixes[(size_t) h] * scale[0] + base[(size_t) h]) + c_.hc_eps;          // hc_pre
        post[h] = 2.0f * sigmoidf(mixes[(size_t) (hc + h)] * scale[1] + base[(size_t) (hc + h)]);         // hc_post
    }
    // comb[src * hc + dst]: softmax over dst, then Sinkhorn (build_hc_sinkhorn)
    for (int64_t k = 0; k < hc * hc; ++k) comb[k] = mixes[(size_t) (2 * hc + k)] * scale[2] + base[(size_t) (2 * hc + k)];
    for (int64_t src = 0; src < hc; ++src) {
        float* row = comb + src * hc;
        float mx = row[0];
        for (int64_t d = 1; d < hc; ++d) mx = std::max(mx, row[d]);
        float sum = 0.0f;
        for (int64_t d = 0; d < hc; ++d) sum += (row[d] = std::exp(row[d] - mx));
        for (int64_t d = 0; d < hc; ++d) row[d] = row[d] / sum + c_.hc_eps;
    }
    auto norm_cols = [&] {
        for (int64_t d = 0; d < hc; ++d) {
            float s = 0.0f;
            for (int64_t src = 0; src < hc; ++src) s += comb[src * hc + d];
            s += c_.hc_eps;
            for (int64_t src = 0; src < hc; ++src) comb[src * hc + d] /= s;
        }
    };
    auto norm_rows = [&] {
        for (int64_t src = 0; src < hc; ++src) {
            float s = 0.0f;
            for (int64_t d = 0; d < hc; ++d) s += comb[src * hc + d];
            s += c_.hc_eps;
            for (int64_t d = 0; d < hc; ++d) comb[src * hc + d] /= s;
        }
    };
    norm_cols();
    for (int64_t i = 1; i < c_.hc_sinkhorn_iters; ++i) {
        norm_rows();
        norm_cols();
    }
    // the mixer's input: the streams weighted by pre (the 3-argument build_hc_pre)
    for (int64_t e = 0; e < E; ++e) {
        float acc = R[e] * pre[0];
        for (int64_t h = 1; h < hc; ++h) acc += R[h * E + e] * pre[(size_t) h];
        x[e] = acc;
    }
}

// build_hc_post: R[dst] = y * post[dst] + sum_src R_old[src] * comb[src][dst]
void CpuRef::hc_post(const float* y, float* R, const float* post, const float* comb) {
    const int64_t hc = c_.hc, E = c_.n_embd;
    std::vector<float> out((size_t) (hc * E));
    for (int64_t d = 0; d < hc; ++d)
        for (int64_t e = 0; e < E; ++e) {
            float acc = y[e] * post[d];
            for (int64_t src = 0; src < hc; ++src) acc += R[src * E + e] * comb[src * hc + d];
            out[(size_t) (d * E + e)] = acc;
        }
    std::memcpy(R, out.data(), out.size() * sizeof(float));
}

// ================================ KDA (build_kda_layer) ================================

void CpuRef::kda_layer(int64_t il, const float* x, float* y) {
    const LayerWeights& w = m_.layer(il);
    LayerState& s = *st_[(size_t) il];
    const int64_t H = c_.n_head[(size_t) il], D = c_.kda_head_dim, DI = H * D, K = c_.ssm_d_conv;

    // q, k, v: projection, causal conv over the last K inputs, SiLU (glm5_conv1d)
    const Tensor* proj[3] = {&w.kda.wq, &w.kda.wk, &w.kda.wv};
    std::vector<float> qkv[3];
    for (int j = 0; j < 3; ++j) {
        std::vector<float> raw((size_t) DI);
        matvec(*proj[j], x, raw.data());
        std::vector<float>& cs = s.conv[j];
        const std::vector<float>& cw = s.conv_w[j];
        qkv[j].resize((size_t) DI);
        for (int64_t ch = 0; ch < DI; ++ch) {
            float acc = 0.0f;
            for (int64_t k = 0; k < K - 1; ++k) acc += cs[(size_t) (k * DI + ch)] * cw[(size_t) (ch * K + k)];
            acc += raw[(size_t) ch] * cw[(size_t) (ch * K + K - 1)];
            qkv[j][(size_t) ch] = siluf(acc);
        }
        // the window moves one row: drop the oldest, append this token's projection
        if (K > 1) {
            std::memmove(cs.data(), cs.data() + DI, (size_t) ((K - 2) * DI) * sizeof(float));
            std::memcpy(cs.data() + (K - 2) * DI, raw.data(), (size_t) DI * sizeof(float));
        }
    }
    float* q = qkv[0].data();
    float* k = qkv[1].data();
    const float* v = qkv[2].data();

    // decay gate (kda_g1): low rank, + dt bias, then the safe-gate form or softplus
    std::vector<float> fa((size_t) D), g((size_t) DI);
    matvec(w.kda.f_a, x, fa.data());
    matvec(w.kda.f_b, fa.data(), g.data());
    for (int64_t h = 0; h < H; ++h)
        for (int64_t i = 0; i < D; ++i) {
            float& gi = g[(size_t) (h * D + i)];
            gi += s.dt_b[(size_t) (h * D + i)];
            if (c_.has_gate_lower_bound) gi = c_.kda_gate_lower_bound * sigmoidf(-(gi * s.a[(size_t) h]));
            else gi = softplusf(gi) * s.a[(size_t) h];
        }

    std::vector<float> beta((size_t) H);
    matvec(w.kda.beta, x, beta.data());
    for (auto& b : beta) b = sigmoidf(b);

    // q and k: L2 norm per head, eps 1e-6 (build_gdn_l2_norm)
    for (int64_t h = 0; h < H; ++h)
        for (float* t : {q + h * D, k + h * D}) {
            double ss = 0;
            for (int64_t i = 0; i < D; ++i) ss += (double) t[i] * t[i];
            const float inv = 1.0f / std::sqrt((float) ss + 1e-6f);
            for (int64_t i = 0; i < D; ++i) t[i] *= inv;
        }

    // the recurrence, as ggml's fused gated_delta_net: S[i][:] *= exp(g[i]); delta = beta (v - S^T k);
    // S += k delta^T; o = S^T q / sqrt(D)
    std::vector<float> o((size_t) DI);
    const float scale = 1.0f / std::sqrt((float) D);
    pool_->parallel_for(H, [&](int64_t b, int64_t e) {
        std::vector<float> delta((size_t) D);
        for (int64_t h = b; h < e; ++h) {
            float* S = s.S.data() + h * D * D;
            const float* gh = g.data() + h * D;
            const float* kh = k + h * D;
            const float* qh = q + h * D;
            const float* vh = v + h * D;
            for (int64_t i = 0; i < D; ++i) {
                const float d = std::exp(gh[i]);
                for (int64_t j = 0; j < D; ++j) S[i * D + j] *= d;
            }
            for (int64_t j = 0; j < D; ++j) {
                float sum = 0.0f;
                for (int64_t i = 0; i < D; ++i) sum += S[i * D + j] * kh[i];
                delta[(size_t) j] = (vh[j] - sum) * beta[(size_t) h];
            }
            for (int64_t i = 0; i < D; ++i)
                for (int64_t j = 0; j < D; ++j) S[i * D + j] += kh[i] * delta[(size_t) j];
            for (int64_t j = 0; j < D; ++j) {
                float sum = 0.0f;
                for (int64_t i = 0; i < D; ++i) sum += S[i * D + j] * qh[i];
                o[(size_t) (h * D + j)] = sum * scale;
            }
        }
    });

    // output gate: RMSNorm(o) * o_norm * sigmoid(g2), per head
    std::vector<float> ga((size_t) D), g2((size_t) DI), gated((size_t) DI);
    matvec(w.kda.g_a, x, ga.data());
    matvec(w.kda.g_b, ga.data(), g2.data());
    for (int64_t h = 0; h < H; ++h) {
        rms_norm(o.data() + h * D, gated.data() + h * D, D, c_.rms_eps, s.o_norm.data());
        for (int64_t i = 0; i < D; ++i) gated[(size_t) (h * D + i)] *= sigmoidf(g2[(size_t) (h * D + i)]);
    }
    matvec(w.wo, gated.data(), y);
}

// ================================ MLA with the k-pool indexer (build_dsa_layer) ================================

void CpuRef::mla_layer(int64_t il, const float* x, float* y) {
    const LayerWeights& w = m_.layer(il);
    LayerState& s = *st_[(size_t) il];
    const int64_t H = c_.n_head[(size_t) il], QL = c_.q_lora_rank, KL = c_.kv_lora_rank;
    const int64_t HK = c_.head_k_mla, HV = c_.head_v_mla, t = pos_;

    std::vector<float> qr((size_t) QL), q((size_t) (H * HK)), kv((size_t) KL);
    matvec(w.mla.q_a, x, qr.data());
    rms_norm(qr.data(), qr.data(), QL, c_.rms_eps, s.q_a_norm.data());          // q_resid
    matvec(w.mla.q_b, qr.data(), q.data());
    matvec(w.mla.kv_a, x, kv.data());
    rms_norm(kv.data(), kv.data(), KL, c_.rms_eps, s.kv_a_norm.data());         // kv_cmpr
    float* lat = s.latent.data() + t * KL;
    for (int64_t i = 0; i < KL; ++i) lat[i] = round_f16(kv[(size_t) i]);

    // q absorbed through k_b: q_abs[h][c] = sum_d k_b[h][c][d] q[h][d]
    std::vector<float> q_abs((size_t) (H * KL));
    for (int64_t h = 0; h < H; ++h) matvec(w.mla.k_b.slice2(h), q.data() + h * HK, q_abs.data() + h * KL);

    // the indexer (build_kpool_select), or the previous full layer's selection
    if (c_.indexer_full[(size_t) il]) {
        const int64_t IH = c_.indexer_n_head, ID = c_.indexer_head_size, P = c_.indexer_kpool;
        std::vector<float> iq((size_t) (IH * ID)), ik((size_t) ID), wts((size_t) IH);
        matvec(w.mla.idx_attn_q_b, qr.data(), iq.data());
        matvec(w.mla.idx_attn_k, x, ik.data());
        layer_norm(ik.data(), s.idx_key.data() + t * ID, ID, c_.norm_eps, s.idx_k_norm.data(),
                   s.idx_k_norm_b.data());
        matvec(w.mla.idx_kpool_gate, x, s.idx_gate.data() + t * ID);
        if ((t + 1) % P == 0) {                      // this token completes pool t / P: pool its members
            const int64_t p = t / P, m0 = p * P;
            float* pk = s.pooled.data() + p * ID;
            for (int64_t d = 0; d < ID; ++d) {
                float mx = -INFINITY;
                std::vector<float> lg((size_t) P);
                for (int64_t m = 0; m < P; ++m) {
                    lg[(size_t) m] = s.idx_gate[(size_t) ((m0 + m) * ID + d)] + s.idx_ape[(size_t) (m * ID + d)];
                    mx = std::max(mx, lg[(size_t) m]);
                }
                float sum = 0.0f;
                for (auto& l : lg) sum += (l = std::exp(l - mx));
                float acc = 0.0f;
                for (int64_t m = 0; m < P; ++m) acc += lg[(size_t) m] / sum * s.idx_key[(size_t) ((m0 + m) * ID + d)];
                pk[d] = acc;
            }
        }
        matvec(w.mla.idx_proj, x, wts.data());
        const float wscale = 1.0f / std::sqrt((float) (ID * IH));
        for (auto& wv : wts) wv *= wscale;

        const int64_t nv = (t + 1) / P;                            // complete pools visible to this token
        const int64_t n_top = c_.indexer_top_k / P;
        std::vector<int64_t> pools((size_t) nv);
        std::iota(pools.begin(), pools.end(), 0);
        if (nv > n_top) {
            std::vector<float> score((size_t) nv);
            for (int64_t p = 0; p < nv; ++p) {
                float sc = 0.0f;
                for (int64_t h = 0; h < IH; ++h) {
                    float dot = 0.0f;
                    for (int64_t d = 0; d < ID; ++d) dot += iq[(size_t) (h * ID + d)] * s.pooled[(size_t) (p * ID + d)];
                    sc += std::max(dot, 0.0f) * wts[(size_t) h];
                }
                score[(size_t) p] = sc;
            }
            std::stable_sort(pools.begin(), pools.end(),
                             [&](int64_t a, int64_t b) { return score[(size_t) a] > score[(size_t) b]; });
            pools.resize((size_t) n_top);
        }
        sel_.clear();
        for (int64_t p : pools)
            for (int64_t m = 0; m < P; ++m) sel_.push_back((int32_t) (p * P + m));
        if (c_.indexer_select_tail)
            for (int64_t k = 0; k < (t + 1) % P; ++k) sel_.push_back((int32_t) (t - k));
    }

    // sparse attention over the selected cells, in the latent space. The K cache is f16, so llama.cpp rounds the
    // query and the probabilities to f16 for its two products; this does the same.
    const float kq_scale = 1.0f / std::sqrt((float) HK);
    std::vector<float> lat_out((size_t) (H * KL));
    pool_->parallel_for(H, [&](int64_t b, int64_t e) {
        std::vector<float> sc(sel_.size()), qa((size_t) KL);
        for (int64_t h = b; h < e; ++h) {
            for (int64_t c = 0; c < KL; ++c) qa[(size_t) c] = round_f16(q_abs[(size_t) (h * KL + c)]);
            float mx = -INFINITY;
            for (size_t i = 0; i < sel_.size(); ++i) {
                const float* kc = s.latent.data() + (int64_t) sel_[i] * KL;
                float dot = 0.0f;
                for (int64_t c = 0; c < KL; ++c) dot += qa[(size_t) c] * kc[c];
                sc[i] = dot * kq_scale;
                mx = std::max(mx, sc[i]);
            }
            float sum = 0.0f;
            for (auto& v : sc) sum += (v = std::exp(v - mx));
            float* lo = lat_out.data() + h * KL;
            std::fill(lo, lo + KL, 0.0f);
            for (size_t i = 0; i < sel_.size(); ++i) {
                const float p = round_f16(sc[i] / sum);
                const float* kc = s.latent.data() + (int64_t) sel_[i] * KL;
                for (int64_t c = 0; c < KL; ++c) lo[c] += p * kc[c];
            }
        }
    });
    // v_b per head ([KL, HV] each), then the output projection
    std::vector<float> heads((size_t) (H * HV));
    for (int64_t h = 0; h < H; ++h) matvec(w.mla.v_b.slice2(h), lat_out.data() + h * KL, heads.data() + h * HV);
    matvec(w.wo, heads.data(), y);
}

// ================================ FFN ================================

void CpuRef::swiglu_ffn(const Tensor& gate, const Tensor& up, const Tensor& down, float limit, const float* x,
                        float* y) {
    const int64_t F = gate.ne[1];
    std::vector<float> g((size_t) F), u((size_t) F);
    matvec(gate, x, g.data());
    matvec(up, x, u.data());
    for (int64_t i = 0; i < F; ++i) {
        float gi = g[(size_t) i], ui = u[(size_t) i];
        if (limit > 1e-6f) {                                       // ggml_swiglu_clamp
            gi = std::min(gi, limit);
            ui = std::clamp(ui, -limit, limit);
        }
        g[(size_t) i] = siluf(gi) * ui;
    }
    matvec(down, g.data(), y);
}

void CpuRef::ffn(int64_t il, const float* x, float* y) {
    const LayerWeights& w = m_.layer(il);
    const LayerState& s = *st_[(size_t) il];
    const int64_t E = c_.n_embd, NE = c_.n_expert, KU = c_.n_expert_used;
    if (c_.is_dense(il)) {
        swiglu_ffn(w.ffn_gate, w.ffn_up, w.ffn_down, c_.clamp_shexp[(size_t) il], x, y);
        return;
    }
    // routing (build_moe_ffn): probs, biased selection, unbiased weights, normalize, scale
    std::vector<float> logits((size_t) NE), probs((size_t) NE), selp((size_t) NE);
    matvec(w.gate_inp, x, logits.data());
    if (c_.expert_gating_func == 2) {
        for (int64_t e = 0; e < NE; ++e) probs[(size_t) e] = sigmoidf(logits[(size_t) e]);
    } else {
        const float mx = *std::max_element(logits.begin(), logits.end());
        float sum = 0.0f;
        for (int64_t e = 0; e < NE; ++e) sum += (probs[(size_t) e] = std::exp(logits[(size_t) e] - mx));
        for (auto& p : probs) p /= sum;
    }
    for (int64_t e = 0; e < NE; ++e) selp[(size_t) e] = probs[(size_t) e] + s.exp_probs_b[(size_t) e];
    std::vector<int64_t> ids((size_t) NE);
    std::iota(ids.begin(), ids.end(), 0);
    std::partial_sort(ids.begin(), ids.begin() + KU, ids.end(), [&](int64_t a, int64_t b) {
        return selp[(size_t) a] > selp[(size_t) b] || (selp[(size_t) a] == selp[(size_t) b] && a < b);
    });
    std::vector<float> wts((size_t) KU);
    for (int64_t k = 0; k < KU; ++k) wts[(size_t) k] = probs[(size_t) ids[(size_t) k]];
    if (c_.expert_weights_norm) {
        float sum = 0.0f;
        for (float wv : wts) sum += wv;
        sum = std::max(sum, 6.103515625e-5f);
        for (auto& wv : wts) wv /= sum;
    }
    if (c_.expert_weights_scale != 0.0f && c_.expert_weights_scale != 1.0f)
        for (auto& wv : wts) wv *= c_.expert_weights_scale;

    std::fill(y, y + E, 0.0f);
    std::vector<float> ye((size_t) E);
    for (int64_t k = 0; k < KU; ++k) {
        const int64_t e = ids[(size_t) k];
        swiglu_ffn(w.gate_exps.slice2(e), w.up_exps.slice2(e), w.down_exps.slice2(e), c_.clamp_exp[(size_t) il], x,
                   ye.data());
        for (int64_t i = 0; i < E; ++i) y[i] += wts[(size_t) k] * ye[(size_t) i];
    }
    if (c_.n_expert_shared > 0) {
        swiglu_ffn(w.gate_shexp, w.up_shexp, w.down_shexp, c_.clamp_shexp[(size_t) il], x, ye.data());
        for (int64_t i = 0; i < E; ++i) y[i] += ye[(size_t) i];
    }
}

// ================================ one token ================================

const std::vector<float>& CpuRef::eval(int32_t token, Trace* trace) {
    if (pos_ >= max_ctx_) throw std::runtime_error("CpuRef: the sequence is longer than max_ctx");
    if (token < 0 || token >= c_.n_vocab) throw std::runtime_error("CpuRef: token id outside the vocabulary");
    const int64_t E = c_.n_embd, hc = c_.hc;

    // embedding, copied into every stream (hc_init)
    std::vector<float> emb((size_t) E);
    {
        const Tensor& te = m_.tok_embd();
        if (te.type == GGML_TYPE_F32) std::memcpy(emb.data(), te.row(token), (size_t) E * sizeof(float));
        else ggml_get_type_traits((ggml_type) te.type)->to_float(te.row(token), emb.data(), E);
    }
    std::vector<float> R((size_t) (hc * E));
    for (int64_t h = 0; h < hc; ++h) std::memcpy(R.data() + h * E, emb.data(), (size_t) E * sizeof(float));

    std::vector<float> x((size_t) E), xn((size_t) E), yv((size_t) E), post((size_t) hc), comb((size_t) (hc * hc));
    if (trace) trace->layer_out.clear();
    for (int64_t il = 0; il < c_.n_layer; ++il) {
        const LayerWeights& w = m_.layer(il);
        const LayerState& s = *st_[(size_t) il];
        hc_pre(w.hc_attn_fn, w.hc_attn_scale, w.hc_attn_base, R.data(), x.data(), post.data(), comb.data());
        rms_norm(x.data(), xn.data(), E, c_.rms_eps, s.attn_norm.data());
        if (c_.is_kda(il)) kda_layer(il, xn.data(), yv.data());
        else mla_layer(il, xn.data(), yv.data());
        hc_post(yv.data(), R.data(), post.data(), comb.data());

        hc_pre(w.hc_ffn_fn, w.hc_ffn_scale, w.hc_ffn_base, R.data(), x.data(), post.data(), comb.data());
        rms_norm(x.data(), xn.data(), E, c_.rms_eps, s.ffn_norm.data());
        ffn(il, xn.data(), yv.data());
        hc_post(yv.data(), R.data(), post.data(), comb.data());
        if (trace) trace->layer_out.push_back(R);
    }

    // hc_head: the mean of the streams, then the output norm and head
    for (int64_t e = 0; e < E; ++e) {
        float acc = R[(size_t) e];
        for (int64_t h = 1; h < hc; ++h) acc += R[(size_t) (h * E + e)];
        x[(size_t) e] = acc * (1.0f / (float) hc);
    }
    const std::vector<float> on = to_f32(m_.output_norm());
    rms_norm(x.data(), xn.data(), E, c_.rms_eps, on.data());
    logits_.resize((size_t) c_.n_vocab);
    matvec(m_.output(), xn.data(), logits_.data());
    ++pos_;
    return logits_;
}

}  // namespace strata::glm
