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
#include <utility>
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
    float* S_bak = nullptr;                  ///< --mtp: S and conv_st after the first token of a pair (drop_last)
    float* conv_bak[3] = {nullptr, nullptr, nullptr};
    // MLA
    DevMatrix q_a, q_b, kv_a, k_b, v_b;
    float *q_a_norm = nullptr, *kv_a_norm = nullptr;
    uint16_t* cache = nullptr;               ///< max_ctx x KL, f16; page-locked host memory with --kv-in-ram
    uint16_t* head = nullptr;                ///< --kv-in-ram: rows 0 .. max_sel-1 in VRAM too (the dense range)
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

/// Tokens one decode pass takes at most: one, or a token and its draft (--mtp, GpuEngine::eval_pair). Every per-token
/// decode buffer holds this many, one after another.
constexpr size_t kNMax = 2;

/// --decode-graphs: one layer's fixed run of ops as an instantiated CUDA graph, and the residual buffers it reads and
/// leaves (eval's decode_segment).
struct DecodeGraph {
    cudaGraphExec_t exec = nullptr;
    float *R_in = nullptr, *R2_in = nullptr, *R_out = nullptr, *R2_out = nullptr;
    bool failed = false;                     ///< the capture failed: the layer runs op by op
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
    int32_t* n_sel1 = nullptr;               ///< the selection's length as the GPU counted it (topk_pools_rows)
    uint64_t dec_sel_have = 0;               ///< which full indexer layer's selection `sel` holds (dec_sel_version)
    uint16_t* kvsel = nullptr;               ///< --kv-in-ram: the selected rows, read from host memory
    int32_t* iota = nullptr;                 ///< 0, 1, ..., max_sel-1: kvsel's rows as a selection
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
    /// the large buffers only a prompt chunk uses (bytes each): with --lend-prompt-buffers they hold expert-cache
    /// slots between prompts (ExpertCache::add_loans)
    std::vector<std::pair<uint8_t*, size_t>> lendable;
    int32_t *d_idx = nullptr, *d_slot = nullptr;
    // past the indexer's dense range: each query's cells [T][max_sel] and their count, and which full indexer
    // layer's selection they are (Impl::sel_version)
    int32_t *sel_rows = nullptr, *n_sel = nullptr;
    uint64_t sel_have = 0;
    // --kv-in-ram: the cells a batch of queries selects, each read from host memory once (cell_union)
    int32_t *kv_flags = nullptr, *kv_pos = nullptr, *kv_U = nullptr, *kv_count = nullptr, *kv_local = nullptr;
    uint16_t* kv_rows = nullptr;
    void* kv_tmp = nullptr;
    size_t kv_tmp_bytes = 0;
    std::unique_ptr<gpu::GemmCtx> gemm;
    // streamed experts: two device buffers, filled on a copy stream while the other is computed
    uint8_t* dstage[2] = {nullptr, nullptr};
    cudaStream_t copy_stream = nullptr;
    cudaEvent_t ev_copied[2] = {nullptr, nullptr}, ev_free[2] = {nullptr, nullptr};
    cudaEvent_t ev_handoff = nullptr;        ///< the chunk's streams handed over to this card (copy_stream waits)
    // --prefill-spread: as a helper, this card's share of another card's layer (its rows' tokens and weights, its
    // partial sum); as the layer's card, the partial sums it receives (one per other card, by ctx index)
    int sms = 0;                             ///< multiprocessors, for sharing the experts out
    float* y_part = nullptr;
    std::vector<float*> y_recv;
    float* d_wrow = nullptr;
    int32_t* h_idx = nullptr;                ///< pinned: the share's tokens, uploaded on this card's stream
    float* h_wrow = nullptr;                 ///< pinned: their routing weights
    cudaEvent_t ev_x = nullptr;              ///< recorded here when this card has sent its xn to the helpers
    cudaEvent_t ev_y = nullptr;              ///< recorded here when this card's partial sum is on its way back
    cudaEvent_t ev_help = nullptr;           ///< --decode-help: this card's share of a token's layer is back in RAM
    cudaStream_t xfer = nullptr;             ///< --prefill-spread: sends the chunk's rows to the helpers
    cudaEvent_t ev_xready = nullptr;         ///< the rows are ready on this card's stream
    // copying from page-locked RAM to this card (GB/s): alone, and with every card copying at once
    double link_solo = 0, link_all = 0;
    // --prefill-spread: this card's share of a layer's streamed experts, copied in while the layer's attention half
    // runs (pf_slots experts of blob_max bytes), each slot's copy done event, and this card done with the pool
    uint8_t* pf_pool = nullptr;
    int pf_slots = 0;
    std::vector<cudaEvent_t> pf_ev;
    cudaEvent_t ev_pf_free = nullptr;
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
    // --decode-help: the misses the GPUs take, and each card's share's sum in RAM (pinned, by ctx index)
    std::vector<int32_t> help_ids;
    std::vector<float> help_w;
    std::vector<float*> h_yhelp;
    // STRATA_GLM_TIMING=1: where eval's time goes, printed when the engine ends (seconds, tokens)
    bool timing = false;
    double t_eval = 0, t_route_wait = 0, t_cpu = 0, t_logits_wait = 0, t_maintain = 0, t_help_wait = 0;
    int64_t t_tokens = 0, t_pairs = 0;
    // and how two tokens in a row route (what one pass over a token and its draft would compute): per layer, the
    // last token's experts; over the MoE layers seen, one token's experts and misses and the pair's
    std::vector<int32_t> prev_ids;
    int64_t r_layers = 0, r_union = 0, r_miss = 0, r_union_miss = 0;
    // host side
    float *h_x = nullptr, *h_w = nullptr, *h_yr = nullptr;
    int32_t* h_ids = nullptr;
    uint64_t dec_sel_version = 0;            ///< counts the full indexer layers' selections, one token at a time
    int dec_sel_ctx = 0;                     ///< the device of the last one
    std::vector<float> logits, emb;
    int64_t max_sel = 0, max_pools = 0;
    // prompt chunks
    uint8_t* hstage[2] = {nullptr, nullptr};     ///< pinned: an expert copied out of the mapped file
    std::vector<uint8_t> experts_pinned;         ///< per layer: its experts are page-locked (--pin-experts)
    size_t blob_max = 0;
    int prefill_rows = 0;                        ///< a chunk's tokens at a time outside the routed experts
    std::vector<void*> kv_host;                  ///< --kv-in-ram: the latent caches' page-locked blocks
    std::vector<int32_t> h_ids_rows;
    std::vector<float> h_w_rows;
    int32_t *h_sel_rows = nullptr, *h_n_sel = nullptr;   ///< pinned: the last selection, for other devices
    uint64_t sel_version = 0;                            ///< counts the full indexer layers' selections
    // STRATA_GLM_PROFILE=FROM,N (nvtx.hpp): the chunks recorded by a profiler started and stopped from here
    int64_t prof_from = -1;
    int prof_left = 0;
    bool prof_on = false;
    // STRATA_GLM_PROFILE_EVAL=FROM,N: the same for N tokens decoded from position FROM, and eval's ranges' names
    int64_t prof_eval_from = -1;
    int prof_eval_left = 0;
    std::vector<std::string> nv_attn, nv_ffn;
    // --decode-graphs: per layer, for passes of one token and of two (eval_pair)
    std::vector<DecodeGraph> dg, dg2;
    // --mtp: the logits of the last pair, and whether drop_last may take its second token back
    std::vector<float> plogits;
    bool pair_backup = false;
    int64_t graph_recaptures = 0;
    // --lend-prompt-buffers: the prompt path has its buffers back from the expert cache (from a prompt until the next
    // decoded token); prefill is running; how often a prompt took them back, and the short prompts that did not
    bool loans_out = false, in_prefill = false;
    int64_t loan_reclaims = 0, short_prompts = 0;
    bool spread = false;                         ///< --prefill-spread with more than one card
    std::vector<std::vector<char>> same_link;    ///< cards that share one link to RAM (measure_links)
    std::vector<std::vector<int>> help_order;    ///< per card: the cards that help it, its own first (--decode-help)
    int64_t pf_layer = -1;                       ///< the layer whose streamed experts were copied in ahead
    std::vector<int> pf_card, pf_slot;           ///< per expert of it: the card and pool slot, or -1

    Impl(const Model& model, const EngineOptions& o) : m(model), c(model.config()), opt(o) {}
};

}  // namespace strata::glm
