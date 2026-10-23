// src/glm/engine_impl.hpp - GpuEngine's internals, shared by engine.cu (one token) and engine_prefill.cu (a chunk
// of prompt tokens). Not a public header.
#pragma once

#include "strata/glm/cpu_experts.hpp"
#include "strata/glm/engine.hpp"
#include "strata/glm/expert_cache.hpp"
#include "strata/glm/gpu_gemm.hpp"
#include "strata/glm/gpu_linear.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::glm {

using gpu::DevMatrix;

namespace detail {

inline void ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(e));
}

template <class T>
T* dalloc(size_t n, size_t* bytes = nullptr) {
    T* p = nullptr;
    if (n == 0) n = 1;
    ok(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    ok(cudaMemset(p, 0, n * sizeof(T)), "cudaMemset");
    if (bytes) *bytes += n * sizeof(T);
    return p;
}

template <class T>
T* halloc(size_t n) {
    T* p = nullptr;
    ok(cudaMallocHost(&p, std::max<size_t>(n, 1) * sizeof(T)), "cudaMallocHost");
    return p;
}

}  // namespace detail

// ================================================================ per-layer weights and state

struct DevLayer {
    int dev = 0;
    // mHC and norms
    DevMatrix hc_attn_fn, hc_ffn_fn;
    float *hc_attn_scale = nullptr, *hc_attn_base = nullptr, *hc_ffn_scale = nullptr, *hc_ffn_base = nullptr;
    float *attn_norm = nullptr, *ffn_norm = nullptr;
    DevMatrix wo;
    // KDA
    DevMatrix wq, wk, wv, f_a, f_b, beta, g_a, g_b;
    float *conv_w[3] = {nullptr, nullptr, nullptr}, *a = nullptr, *dt_b = nullptr, *o_norm = nullptr;
    float* S = nullptr;                      ///< H x D x D
    float* conv_st[3] = {nullptr, nullptr, nullptr};
    // MLA
    DevMatrix q_a, q_b, kv_a, k_b, v_b;
    float *q_a_norm = nullptr, *kv_a_norm = nullptr;
    uint16_t* cache = nullptr;               ///< max_ctx x KL, f16
    // indexer
    DevMatrix idx_q_b, idx_k, idx_gate, idx_proj;
    float *idx_k_norm = nullptr, *idx_k_norm_b = nullptr, *idx_ape = nullptr;
    float *pend_k = nullptr, *pend_g = nullptr;   ///< P x ID: the keys and gates of the pool being filled
    float* pooled = nullptr;                 ///< (max_ctx / P + 1) x ID
    // FFN
    DevMatrix ffn_gate, ffn_up, ffn_down, gate_inp, gate_shexp, up_shexp, down_shexp;
    float* exp_probs_b = nullptr;
    size_t bytes = 0;
};

/// Scratch on one device, sized for the largest layer it may run.
struct DevCtx {
    int id = 0;
    cudaStream_t stream = nullptr;
    float *R = nullptr, *R2 = nullptr, *flat = nullptr, *mixes = nullptr, *pc = nullptr;
    float *x = nullptr, *xn = nullptr, *y = nullptr;
    float *raw = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *fa = nullptr, *g = nullptr, *g2 = nullptr,
          *bta = nullptr, *o = nullptr, *gated = nullptr;
    float *qr = nullptr, *qm = nullptr, *kv = nullptr, *q_abs = nullptr, *lat = nullptr, *heads = nullptr,
          *scores = nullptr, *iq = nullptr, *ik = nullptr, *wts = nullptr, *pscore = nullptr;
    int32_t* sel = nullptr;
    float *fg = nullptr, *fu = nullptr, *fact = nullptr, *logits_e = nullptr, *wexp = nullptr, *ysh = nullptr,
          *yr = nullptr, *yexp = nullptr, *ygpu = nullptr, *out_logits = nullptr;
    int32_t* ids = nullptr;
    void* q8 = nullptr;
    void* q8x = nullptr;   ///< xn quantized once for every cached expert's gate and up
    size_t bytes = 0;
    // ---- prompt chunks (engine_prefill.cu): T tokens at once
    int T = 0;
    float *pR = nullptr, *pR2 = nullptr, *pflat = nullptr, *pmixes = nullptr, *ppc = nullptr;
    float *px = nullptr, *pxn = nullptr, *py = nullptr;
    uint8_t* arena = nullptr;                ///< the mixer's or the FFN's buffers, one at a time
    size_t arena_bytes = 0;
    int32_t *d_idx = nullptr, *d_slot = nullptr;
    // past the indexer's dense range: each query's cells [T][max_sel] and their count, and which full indexer
    // layer's selection they are (Impl::sel_version)
    int32_t *sel_rows = nullptr, *n_sel = nullptr;
    uint64_t sel_have = 0;
    std::unique_ptr<gpu::GemmCtx> gemm;
    // streamed experts: two device buffers, filled on a copy stream while the other is computed
    uint8_t* dstage[2] = {nullptr, nullptr};
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t ev_copied[2] = {nullptr, nullptr}, ev_free[2] = {nullptr, nullptr};
};

struct GpuEngine::Impl {
    const Model& m;
    const Config& c;
    EngineOptions opt;
    std::vector<DevLayer> L;
    std::vector<DevCtx> ctx;                 ///< one per device used
    std::vector<int> ctx_of_layer;           ///< index into ctx
    DevMatrix output;
    float* output_norm = nullptr;
    int out_ctx = 0;
    std::unique_ptr<ThreadPool> pool;
    std::unique_ptr<CpuExperts> experts;
    std::unique_ptr<ExpertCache> cache;
    std::vector<int32_t> miss_ids;
    std::vector<float> miss_w;
    // STRATA_GLM_TIMING=1: where eval's time goes, printed when the engine ends (seconds, tokens)
    bool timing = false;
    double t_eval = 0, t_route_wait = 0, t_cpu = 0, t_logits_wait = 0, t_maintain = 0;
    int64_t t_tokens = 0;
    // host side
    float *h_x = nullptr, *h_w = nullptr, *h_yr = nullptr, *h_pscore = nullptr;
    int32_t* h_ids = nullptr;
    std::vector<int32_t> sel_host;
    std::vector<float> logits, emb;
    int64_t max_sel = 0, max_pools = 0;
    // prompt chunks
    uint8_t* hstage[2] = {nullptr, nullptr};     ///< pinned: an expert copied out of the mapped file
    std::vector<uint8_t> experts_pinned;         ///< per layer: its experts are page-locked (--pin-experts)
    size_t blob_max = 0;
    std::vector<int32_t> h_ids_rows;
    std::vector<float> h_w_rows;
    int32_t *h_sel_rows = nullptr, *h_n_sel = nullptr;   ///< pinned: the last selection, for other devices
    uint64_t sel_version = 0;                            ///< counts the full indexer layers' selections

    Impl(const Model& model, const EngineOptions& o) : m(model), c(model.config()), opt(o) {}
};

}  // namespace strata::glm
