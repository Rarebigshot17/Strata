// src/glm/engine.cu - GLM-5.3-Flash on the GPUs, one token at a time (engine.hpp).
//
// Each block below names the cpu_ref.cpp function it reproduces. The routed experts run on the CPU (CpuExperts)
// while the GPU computes the shared expert; the indexer's top pools are chosen on the host from the GPU's
// scores, which is also where cpu_ref chooses them.
#include "strata/glm/engine.hpp"

#include "strata/glm/cpu_experts.hpp"
#include "strata/glm/expert_cache.hpp"
#include "strata/glm/gpu_kernels.hpp"
#include "strata/glm/gpu_linear.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>

namespace strata::glm {

using gpu::DevMatrix;

namespace {

void ok(cudaError_t e, const char* what) {
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

}  // namespace

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
    size_t bytes = 0;
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
    // host side
    float *h_x = nullptr, *h_w = nullptr, *h_yr = nullptr, *h_pscore = nullptr;
    int32_t* h_ids = nullptr;
    std::vector<int32_t> sel_host;
    std::vector<float> logits, emb;
    int64_t max_sel = 0, max_pools = 0;

    Impl(const Model& model, const EngineOptions& o) : m(model), c(model.config()), opt(o) {}
};

GpuEngine::GpuEngine(const Model& model, const EngineOptions& opt) : impl_(new Impl(model, opt)) {
    Impl& I = *impl_;
    const Config& c = I.c;
    int n_dev = 0;
    ok(cudaGetDeviceCount(&n_dev), "cudaGetDeviceCount");
    std::vector<int> devs = opt.devices;
    if (devs.empty())
        for (int d = 0; d < n_dev; ++d) devs.push_back(d);
    for (int d : devs)
        if (d < 0 || d >= n_dev) throw std::runtime_error("CUDA device " + std::to_string(d) + " does not exist");

    // ---- the layers' sizes, to split them over the devices
    std::vector<double> layer_bytes((size_t) c.n_layer, 0.0);
    auto tb = [](const Tensor& t) { return t.ok() ? (double) t.row_bytes() * (double) t.rows() : 0.0; };
    for (int64_t il = 0; il < c.n_layer; ++il) {
        const LayerWeights& w = model.layer(il);
        double b = tb(w.hc_attn_fn) + tb(w.hc_ffn_fn) + tb(w.wo);
        for (const Tensor* t : {&w.kda.wq, &w.kda.wk, &w.kda.wv, &w.kda.f_a, &w.kda.f_b, &w.kda.g_a, &w.kda.g_b,
                                &w.mla.q_a, &w.mla.q_b, &w.mla.kv_a, &w.mla.k_b, &w.mla.v_b, &w.mla.idx_attn_q_b,
                                &w.ffn_gate, &w.ffn_up, &w.ffn_down, &w.gate_shexp, &w.up_shexp, &w.down_shexp})
            b += tb(*t);
        if (!c.is_kda(il)) b += (double) opt.max_ctx * c.kv_lora_rank * 2.0;   // the f16 latent cache
        layer_bytes[(size_t) il] = b;
    }
    std::vector<double> share = opt.split;
    if (share.empty()) {
        for (int d : devs) {
            size_t fr = 0, tot = 0;
            ok(cudaSetDevice(d), "cudaSetDevice");
            ok(cudaMemGetInfo(&fr, &tot), "cudaMemGetInfo");
            share.push_back((double) fr);
        }
    }
    if (share.size() != devs.size()) throw std::runtime_error("--split needs one value per device");
    const double total = std::accumulate(layer_bytes.begin(), layer_bytes.end(), 0.0);
    const double share_sum = std::accumulate(share.begin(), share.end(), 0.0);
    I.ctx_of_layer.assign((size_t) c.n_layer, 0);
    {
        size_t di = 0;
        double acc = 0.0, limit = total * share[0] / share_sum;
        for (int64_t il = 0; il < c.n_layer; ++il) {
            while (di + 1 < devs.size() && acc + layer_bytes[(size_t) il] / 2 > limit) {
                ++di;
                limit += total * share[di] / share_sum;
            }
            I.ctx_of_layer[(size_t) il] = (int) di;
            acc += layer_bytes[(size_t) il];
        }
    }
    I.out_ctx = I.ctx_of_layer.back();

    // ---- scratch sizes
    int64_t maxH = 0, maxDI = 0, maxF = c.n_ff;
    for (int64_t il = 0; il < c.n_layer; ++il) {
        maxH = std::max(maxH, c.n_head[(size_t) il]);
        if (c.is_kda(il)) maxDI = std::max(maxDI, c.n_head[(size_t) il] * c.kda_head_dim);
        maxF = std::max(maxF, c.n_ff_exp[(size_t) il] * std::max<int64_t>(c.n_expert_shared, 1));
    }
    const int64_t E = c.n_embd, hc = c.hc, P = c.indexer_kpool;
    I.max_sel = std::min<int64_t>(opt.max_ctx, c.indexer_top_k + P - 1);
    I.max_pools = opt.max_ctx / P + 1;
    const int64_t max_in = std::max({hc * E, maxDI, maxF, maxH * c.head_v_mla, c.q_lora_rank, c.kv_lora_rank,
                                     c.kda_head_dim, maxH * c.head_k_mla});

    I.ctx.resize(devs.size());
    for (size_t di = 0; di < devs.size(); ++di) {
        DevCtx& d = I.ctx[di];
        d.id = devs[di];
        ok(cudaSetDevice(d.id), "cudaSetDevice");
        ok(cudaStreamCreateWithFlags(&d.stream, cudaStreamNonBlocking), "cudaStreamCreate");
        size_t* b = &d.bytes;
        d.R = dalloc<float>((size_t) (hc * E), b);
        d.R2 = dalloc<float>((size_t) (hc * E), b);
        d.flat = dalloc<float>((size_t) (hc * E), b);
        d.mixes = dalloc<float>((size_t) c.hc_mix_dim(), b);
        d.pc = dalloc<float>((size_t) (2 * hc + hc * hc), b);
        d.x = dalloc<float>((size_t) E, b);
        d.xn = dalloc<float>((size_t) E, b);
        d.y = dalloc<float>((size_t) E, b);
        d.raw = dalloc<float>((size_t) maxDI, b);
        d.q = dalloc<float>((size_t) maxDI, b);
        d.k = dalloc<float>((size_t) maxDI, b);
        d.v = dalloc<float>((size_t) maxDI, b);
        d.fa = dalloc<float>((size_t) c.kda_head_dim, b);
        d.g = dalloc<float>((size_t) maxDI, b);
        d.g2 = dalloc<float>((size_t) maxDI, b);
        d.bta = dalloc<float>((size_t) maxH, b);
        d.o = dalloc<float>((size_t) maxDI, b);
        d.gated = dalloc<float>((size_t) maxDI, b);
        d.qr = dalloc<float>((size_t) c.q_lora_rank, b);
        d.qm = dalloc<float>((size_t) (maxH * c.head_k_mla), b);
        d.kv = dalloc<float>((size_t) (c.kv_lora_rank + c.n_rot), b);
        d.q_abs = dalloc<float>((size_t) (maxH * c.kv_lora_rank), b);
        d.lat = dalloc<float>((size_t) (maxH * c.kv_lora_rank), b);
        d.heads = dalloc<float>((size_t) (maxH * c.head_v_mla), b);
        d.scores = dalloc<float>((size_t) (maxH * I.max_sel), b);
        d.sel = dalloc<int32_t>((size_t) I.max_sel, b);
        d.iq = dalloc<float>((size_t) (c.indexer_n_head * c.indexer_head_size), b);
        d.ik = dalloc<float>((size_t) c.indexer_head_size, b);
        d.wts = dalloc<float>((size_t) c.indexer_n_head, b);
        d.pscore = dalloc<float>((size_t) I.max_pools, b);
        d.fg = dalloc<float>((size_t) maxF, b);
        d.fu = dalloc<float>((size_t) maxF, b);
        d.fact = dalloc<float>((size_t) maxF, b);
        d.logits_e = dalloc<float>((size_t) c.n_expert, b);
        d.wexp = dalloc<float>((size_t) c.n_expert_used, b);
        d.ids = dalloc<int32_t>((size_t) c.n_expert_used, b);
        d.ysh = dalloc<float>((size_t) E, b);
        d.yr = dalloc<float>((size_t) E, b);
        d.yexp = dalloc<float>((size_t) E, b);
        d.ygpu = dalloc<float>((size_t) E, b);
        d.q8 = dalloc<uint8_t>(gpu::q8_scratch_bytes(max_in), b);
        if ((int) di == I.out_ctx) d.out_logits = dalloc<float>((size_t) c.n_vocab, b);
    }

    // ---- weights and per-layer state, each on its layer's device
    I.L.resize((size_t) c.n_layer);
    for (int64_t il = 0; il < c.n_layer; ++il) {
        DevLayer& l = I.L[(size_t) il];
        const LayerWeights& w = model.layer(il);
        const DevCtx& d = I.ctx[(size_t) I.ctx_of_layer[(size_t) il]];
        l.dev = d.id;
        ok(cudaSetDevice(d.id), "cudaSetDevice");
        size_t* b = &l.bytes;
        l.hc_attn_fn = gpu::upload(w.hc_attn_fn, b);
        l.hc_ffn_fn = gpu::upload(w.hc_ffn_fn, b);
        l.hc_attn_scale = gpu::upload_f32(w.hc_attn_scale, b);
        l.hc_attn_base = gpu::upload_f32(w.hc_attn_base, b);
        l.hc_ffn_scale = gpu::upload_f32(w.hc_ffn_scale, b);
        l.hc_ffn_base = gpu::upload_f32(w.hc_ffn_base, b);
        l.attn_norm = gpu::upload_f32(w.attn_norm, b);
        l.ffn_norm = gpu::upload_f32(w.ffn_norm, b);
        l.wo = gpu::upload(w.wo, b);
        const int64_t H = c.n_head[(size_t) il];
        if (c.is_kda(il)) {
            const int64_t D = c.kda_head_dim, DI = H * D;
            l.wq = gpu::upload(w.kda.wq, b);
            l.wk = gpu::upload(w.kda.wk, b);
            l.wv = gpu::upload(w.kda.wv, b);
            l.f_a = gpu::upload(w.kda.f_a, b);
            l.f_b = gpu::upload(w.kda.f_b, b);
            l.beta = gpu::upload(w.kda.beta, b);
            l.g_a = gpu::upload(w.kda.g_a, b);
            l.g_b = gpu::upload(w.kda.g_b, b);
            l.conv_w[0] = gpu::upload_f32(w.kda.conv_q, b);
            l.conv_w[1] = gpu::upload_f32(w.kda.conv_k, b);
            l.conv_w[2] = gpu::upload_f32(w.kda.conv_v, b);
            l.a = gpu::upload_f32(w.kda.a, b);
            l.dt_b = gpu::upload_f32(w.kda.dt_b, b);
            l.o_norm = gpu::upload_f32(w.kda.o_norm, b);
            l.S = dalloc<float>((size_t) (H * D * D), b);
            for (auto& cs : l.conv_st) cs = dalloc<float>((size_t) ((c.ssm_d_conv - 1) * DI), b);
        } else {
            l.q_a = gpu::upload(w.mla.q_a, b);
            l.q_b = gpu::upload(w.mla.q_b, b);
            l.kv_a = gpu::upload(w.mla.kv_a, b);
            l.k_b = gpu::upload(w.mla.k_b, b);
            l.v_b = gpu::upload(w.mla.v_b, b);
            l.q_a_norm = gpu::upload_f32(w.mla.q_a_norm, b);
            l.kv_a_norm = gpu::upload_f32(w.mla.kv_a_norm, b);
            l.cache = dalloc<uint16_t>((size_t) (opt.max_ctx * c.kv_lora_rank), b);
            if (c.indexer_full[(size_t) il]) {
                const int64_t ID = c.indexer_head_size;
                l.idx_q_b = gpu::upload(w.mla.idx_attn_q_b, b);
                l.idx_k = gpu::upload(w.mla.idx_attn_k, b);
                l.idx_gate = gpu::upload(w.mla.idx_kpool_gate, b);
                l.idx_proj = gpu::upload(w.mla.idx_proj, b);
                l.idx_k_norm = gpu::upload_f32(w.mla.idx_k_norm, b);
                l.idx_k_norm_b = gpu::upload_f32(w.mla.idx_k_norm_b, b);
                l.idx_ape = gpu::upload_f32(w.mla.idx_kpool_ape, b);
                l.pend_k = dalloc<float>((size_t) (P * ID), b);
                l.pend_g = dalloc<float>((size_t) (P * ID), b);
                l.pooled = dalloc<float>((size_t) (I.max_pools * ID), b);
            }
        }
        if (c.is_dense(il)) {
            l.ffn_gate = gpu::upload(w.ffn_gate, b);
            l.ffn_up = gpu::upload(w.ffn_up, b);
            l.ffn_down = gpu::upload(w.ffn_down, b);
        } else {
            l.gate_inp = gpu::upload(w.gate_inp, b);
            l.exp_probs_b = gpu::upload_f32(w.exp_probs_b, b);
            if (c.n_expert_shared > 0) {
                l.gate_shexp = gpu::upload(w.gate_shexp, b);
                l.up_shexp = gpu::upload(w.up_shexp, b);
                l.down_shexp = gpu::upload(w.down_shexp, b);
            }
        }
    }
    {
        DevCtx& d = I.ctx[(size_t) I.out_ctx];
        ok(cudaSetDevice(d.id), "cudaSetDevice");
        I.output = gpu::upload(model.output(), &d.bytes);
        I.output_norm = gpu::upload_f32(model.output_norm(), &d.bytes);
    }

    // ---- the VRAM expert cache: what is left on each device, minus the reserve
    I.cache = std::make_unique<ExpertCache>(model, opt.cache);
    std::vector<size_t> cache_bytes(I.ctx.size(), 0);
    for (size_t di = 0; di < I.ctx.size(); ++di) {
        std::vector<int64_t> layers;
        for (int64_t il = 0; il < c.n_layer; ++il)
            if (I.ctx_of_layer[(size_t) il] == (int) di) layers.push_back(il);
        cache_bytes[di] = I.cache->allocate(I.ctx[di].id, layers);
    }
    I.cache->warm_start();

    // ---- host side: the CPU experts and the pinned exchange buffers
    const int threads = opt.cpu_threads > 0 ? opt.cpu_threads : (int) std::max(1u, std::thread::hardware_concurrency());
    I.pool = std::make_unique<ThreadPool>(threads);
    I.experts = std::make_unique<CpuExperts>(model, *I.pool);
    I.h_x = halloc<float>((size_t) E);
    I.h_yr = halloc<float>((size_t) E);
    I.h_w = halloc<float>((size_t) c.n_expert_used);
    I.h_ids = halloc<int32_t>((size_t) c.n_expert_used);
    I.h_pscore = halloc<float>((size_t) I.max_pools);
    I.logits.resize((size_t) c.n_vocab);
    I.emb.resize((size_t) (hc * E));

    for (size_t di = 0; di < I.ctx.size(); ++di) {
        size_t bytes = I.ctx[di].bytes;
        int64_t first = -1, last = -1;
        for (int64_t il = 0; il < c.n_layer; ++il)
            if (I.ctx_of_layer[(size_t) il] == (int) di) {
                if (first < 0) first = il;
                last = il;
                bytes += I.L[(size_t) il].bytes;
            }
        char line[200];
        if (first < 0) std::snprintf(line, sizeof line, "GPU %d: no layers (%.2f GB)", I.ctx[di].id, bytes / 1e9);
        else std::snprintf(line, sizeof line, "GPU %d: layers %lld-%lld%s, %.2f GB weights and buffers, %.2f GB expert cache",
                           I.ctx[di].id, (long long) first, (long long) last, (int) di == I.out_ctx ? " + output head" : "",
                           bytes / 1e9, cache_bytes[di] / 1e9);
        placement_.push_back(line);
    }
    placement_.push_back(I.cache->summary());
    reset();
}

double GpuEngine::cache_hit_rate() const {
    const auto& c = *impl_->cache;
    return c.lookups() ? (double) c.hits() / (double) c.lookups() : 0.0;
}

std::string GpuEngine::cache_summary() const { return impl_->cache->summary(); }

void GpuEngine::save_profile() const { impl_->cache->save_profile(); }

GpuEngine::~GpuEngine() {
    // device memory is released with the process; streams and pinned buffers here
    if (!impl_) return;
    for (auto& d : impl_->ctx)
        if (d.stream) cudaStreamDestroy(d.stream);
    for (void* p : {(void*) impl_->h_x, (void*) impl_->h_yr, (void*) impl_->h_w, (void*) impl_->h_ids,
                    (void*) impl_->h_pscore})
        if (p) cudaFreeHost(p);
}

void GpuEngine::reset() {
    Impl& I = *impl_;
    const Config& c = I.c;
    pos_ = 0;
    for (int64_t il = 0; il < c.n_layer; ++il) {
        DevLayer& l = I.L[(size_t) il];
        ok(cudaSetDevice(l.dev), "cudaSetDevice");
        const int64_t H = c.n_head[(size_t) il];
        if (c.is_kda(il)) {
            const int64_t D = c.kda_head_dim;
            ok(cudaMemset(l.S, 0, (size_t) (H * D * D) * sizeof(float)), "cudaMemset");
            for (auto* cs : l.conv_st)
                ok(cudaMemset(cs, 0, (size_t) ((c.ssm_d_conv - 1) * H * D) * sizeof(float)), "cudaMemset");
        }
        // the MLA cache and the pooled keys are only read below the current position: nothing to clear
    }
}

const std::vector<float>& GpuEngine::eval(int32_t token, Trace* trace) {
    Impl& I = *impl_;
    const Config& c = I.c;
    const Model& m = I.m;
    const int64_t E = c.n_embd, hc = c.hc, t = pos_;
    if (t >= I.opt.max_ctx) throw std::runtime_error("the sequence is longer than the context (--ctx)");
    if (token < 0 || token >= c.n_vocab) throw std::runtime_error("token id outside the vocabulary");

    // ---- embedding (hc_init), dequantized on the host
    {
        const Tensor& te = m.tok_embd();
        if (te.type == GGML_TYPE_F32) std::memcpy(I.emb.data(), te.row(token), (size_t) E * sizeof(float));
        else ggml_get_type_traits((ggml_type) te.type)->to_float(te.row(token), I.emb.data(), E);
        for (int64_t h = 1; h < hc; ++h) std::memcpy(I.emb.data() + h * E, I.emb.data(), (size_t) E * sizeof(float));
        DevCtx& d0 = I.ctx[(size_t) I.ctx_of_layer[0]];
        ok(cudaSetDevice(d0.id), "cudaSetDevice");
        ok(cudaMemcpy(d0.R, I.emb.data(), (size_t) (hc * E) * sizeof(float), cudaMemcpyHostToDevice), "embedding");
    }
    if (trace) trace->layer_out.clear();

    int cur = I.ctx_of_layer[0];
    for (int64_t il = 0; il < c.n_layer; ++il) {
        const int ci = I.ctx_of_layer[(size_t) il];
        if (ci != cur) {                                  // the residual streams move to the next device
            DevCtx& from = I.ctx[(size_t) cur];
            DevCtx& to = I.ctx[(size_t) ci];
            ok(cudaSetDevice(from.id), "cudaSetDevice");
            ok(cudaStreamSynchronize(from.stream), "sync");
            ok(cudaMemcpyPeer(to.R, to.id, from.R, from.id, (size_t) (hc * E) * sizeof(float)), "cudaMemcpyPeer");
            cur = ci;
        }
        DevCtx& d = I.ctx[(size_t) ci];
        DevLayer& l = I.L[(size_t) il];
        ok(cudaSetDevice(d.id), "cudaSetDevice");
        void* s = d.stream;
        const int64_t H = c.n_head[(size_t) il];

        // ---- attention half: hc_pre, attn_norm, the mixer, hc_post
        gpu::rms_norm(d.R, d.flat, (int) (hc * E), c.rms_eps, nullptr, s);
        gpu::gemv(l.hc_attn_fn, d.flat, d.mixes, d.q8, s);
        gpu::hc_weights(d.mixes, l.hc_attn_scale, l.hc_attn_base, d.pc, (int) hc, (int) c.hc_sinkhorn_iters,
                        c.hc_eps, s);
        gpu::hc_mix(d.R, d.pc, d.x, (int) hc, (int) E, s);
        gpu::rms_norm(d.x, d.xn, (int) E, c.rms_eps, l.attn_norm, s);

        if (c.is_kda(il)) {
            // ---- kda_layer
            const int64_t D = c.kda_head_dim, DI = H * D, K = c.ssm_d_conv;
            float* outs[3] = {d.q, d.k, d.v};
            const DevMatrix* proj[3] = {&l.wq, &l.wk, &l.wv};
            for (int j = 0; j < 3; ++j) {
                gpu::gemv(*proj[j], d.xn, d.raw, d.q8, s);
                gpu::kda_conv(d.raw, l.conv_st[j], l.conv_w[j], outs[j], (int) DI, (int) K, s);
            }
            gpu::gemv(l.f_a, d.xn, d.fa, d.q8, s);
            gpu::gemv(l.f_b, d.fa, d.g, d.q8, s);
            gpu::kda_gate(d.g, l.dt_b, l.a, (int) H, (int) D, c.has_gate_lower_bound, c.kda_gate_lower_bound, s);
            gpu::gemv(l.beta, d.xn, d.bta, d.q8, s);
            gpu::sigmoid_inplace(d.bta, (int) H, s);
            gpu::l2_norm_heads(d.q, (int) H, (int) D, s);
            gpu::l2_norm_heads(d.k, (int) H, (int) D, s);
            gpu::kda_step(l.S, d.q, d.k, d.v, d.g, d.bta, d.o, (int) H, (int) D, s);
            gpu::gemv(l.g_a, d.xn, d.fa, d.q8, s);
            gpu::gemv(l.g_b, d.fa, d.g2, d.q8, s);
            gpu::kda_out_gate(d.o, l.o_norm, d.g2, d.gated, (int) H, (int) D, c.rms_eps, s);
            gpu::gemv(l.wo, d.gated, d.y, d.q8, s);
        } else {
            // ---- mla_layer
            const int64_t QL = c.q_lora_rank, KL = c.kv_lora_rank, HK = c.head_k_mla, HV = c.head_v_mla;
            gpu::gemv(l.q_a, d.xn, d.qr, d.q8, s);
            gpu::rms_norm(d.qr, d.qr, (int) QL, c.rms_eps, l.q_a_norm, s);
            gpu::gemv(l.q_b, d.qr, d.qm, d.q8, s);
            gpu::gemv(l.kv_a, d.xn, d.kv, d.q8, s);
            gpu::rms_norm(d.kv, d.kv, (int) KL, c.rms_eps, l.kv_a_norm, s);
            gpu::store_f16_row(d.kv, l.cache, t, (int) KL, s);
            for (int64_t h = 0; h < H; ++h)
                gpu::gemv(l.k_b.rows(h * KL, KL), d.qm + h * HK, d.q_abs + h * KL, d.q8, s);
            gpu::round_f16(d.q_abs, (int) (H * KL), s);

            if (c.indexer_full[(size_t) il]) {
                // ---- the k-pool indexer: this token's key and gate, the pool it completes, the selection
                const int64_t IH = c.indexer_n_head, ID = c.indexer_head_size, P = c.indexer_kpool;
                gpu::gemv(l.idx_q_b, d.qr, d.iq, d.q8, s);
                gpu::gemv(l.idx_k, d.xn, d.ik, d.q8, s);
                gpu::layer_norm(d.ik, l.pend_k + (t % P) * ID, (int) ID, c.norm_eps, l.idx_k_norm, l.idx_k_norm_b, s);
                gpu::gemv(l.idx_gate, d.xn, l.pend_g + (t % P) * ID, d.q8, s);
                if ((t + 1) % P == 0)
                    gpu::kpool(l.pend_k, l.pend_g, l.idx_ape, l.pooled + (t / P) * ID, (int) P, (int) ID, s);
                const int64_t nv = (t + 1) / P, n_top = c.indexer_top_k / P;
                std::vector<int64_t> pools((size_t) nv);
                std::iota(pools.begin(), pools.end(), 0);
                if (nv > n_top) {
                    gpu::gemv(l.idx_proj, d.xn, d.wts, d.q8, s);
                    gpu::scale(d.wts, (int) IH, 1.0f / std::sqrt((float) (ID * IH)), s);
                    gpu::kpool_scores(d.iq, l.pooled, d.wts, d.pscore, (int) nv, (int) IH, (int) ID, s);
                    ok(cudaMemcpyAsync(I.h_pscore, d.pscore, (size_t) nv * sizeof(float), cudaMemcpyDeviceToHost,
                                       d.stream), "scores");
                    ok(cudaStreamSynchronize(d.stream), "sync");
                    const float* sc = I.h_pscore;
                    std::stable_sort(pools.begin(), pools.end(), [&](int64_t a, int64_t b) { return sc[a] > sc[b]; });
                    pools.resize((size_t) n_top);
                }
                I.sel_host.clear();
                for (int64_t p : pools)
                    for (int64_t mm = 0; mm < P; ++mm) I.sel_host.push_back((int32_t) (p * P + mm));
                if (c.indexer_select_tail)
                    for (int64_t kk = 0; kk < (t + 1) % P; ++kk) I.sel_host.push_back((int32_t) (t - kk));
            }
            const int n_sel = (int) I.sel_host.size();
            if (n_sel == 0 || n_sel > I.max_sel) throw std::runtime_error("indexer selection of an unexpected size");
            ok(cudaMemcpyAsync(d.sel, I.sel_host.data(), (size_t) n_sel * sizeof(int32_t), cudaMemcpyHostToDevice,
                               d.stream), "selection");
            // pageable source: the copy is complete when cudaMemcpyAsync returns, so sel_host may change after
            gpu::mla_attend(d.q_abs, l.cache, d.sel, n_sel, d.scores, d.lat, (int) H, (int) KL,
                            1.0f / std::sqrt((float) HK), s);
            for (int64_t h = 0; h < H; ++h)
                gpu::gemv(l.v_b.rows(h * HV, HV), d.lat + h * KL, d.heads + h * HV, d.q8, s);
            gpu::gemv(l.wo, d.heads, d.y, d.q8, s);
        }
        gpu::hc_post(d.y, d.R, d.pc, d.R2, (int) hc, (int) E, s);
        std::swap(d.R, d.R2);

        // ---- FFN half: hc_pre, ffn_norm, dense FFN or MoE, hc_post
        gpu::rms_norm(d.R, d.flat, (int) (hc * E), c.rms_eps, nullptr, s);
        gpu::gemv(l.hc_ffn_fn, d.flat, d.mixes, d.q8, s);
        gpu::hc_weights(d.mixes, l.hc_ffn_scale, l.hc_ffn_base, d.pc, (int) hc, (int) c.hc_sinkhorn_iters, c.hc_eps,
                        s);
        gpu::hc_mix(d.R, d.pc, d.x, (int) hc, (int) E, s);
        gpu::rms_norm(d.x, d.xn, (int) E, c.rms_eps, l.ffn_norm, s);

        if (c.is_dense(il)) {
            const int64_t F = c.n_ff;
            gpu::gemv2(l.ffn_gate, l.ffn_up, d.xn, d.fg, d.fu, d.q8, s);
            gpu::swiglu(d.fg, d.fu, d.fact, (int) F, c.clamp_shexp[(size_t) il], s);
            gpu::gemv(l.ffn_down, d.fact, d.y, d.q8, s);
        } else {
            const int KU = (int) c.n_expert_used;
            gpu::gemv(l.gate_inp, d.xn, d.logits_e, d.q8, s);
            gpu::route(d.logits_e, l.exp_probs_b, (int) c.n_expert, KU, c.expert_gating_func, c.expert_weights_norm,
                       c.expert_weights_scale, d.ids, d.wexp, s);
            ok(cudaMemcpyAsync(I.h_x, d.xn, (size_t) E * sizeof(float), cudaMemcpyDeviceToHost, d.stream), "x");
            ok(cudaMemcpyAsync(I.h_ids, d.ids, (size_t) KU * sizeof(int32_t), cudaMemcpyDeviceToHost, d.stream), "ids");
            ok(cudaMemcpyAsync(I.h_w, d.wexp, (size_t) KU * sizeof(float), cudaMemcpyDeviceToHost, d.stream), "w");
            ok(cudaStreamSynchronize(d.stream), "sync");
            I.cache->record(il, I.h_ids, KU);
            // the cached experts on the GPU, in routing order; the rest go to the CPU
            const int64_t F = c.n_ff_exp[(size_t) il];
            ok(cudaMemsetAsync(d.ygpu, 0, (size_t) E * sizeof(float), d.stream), "memset");
            I.miss_ids.clear();
            I.miss_w.clear();
            for (int j = 0; j < KU; ++j) {
                const int slot = I.cache->slot(il, I.h_ids[j]);
                if (slot < 0) {
                    I.miss_ids.push_back(I.h_ids[j]);
                    I.miss_w.push_back(I.h_w[j]);
                    continue;
                }
                DevMatrix G, U, Dn;
                I.cache->matrices(il, slot, G, U, Dn);
                gpu::gemv2(G, U, d.xn, d.fg, d.fu, d.q8, s);
                gpu::swiglu(d.fg, d.fu, d.fact, (int) F, c.clamp_exp[(size_t) il], s);
                gpu::gemv(Dn, d.fact, d.yexp, d.q8, s);
                gpu::axpy(d.ygpu, d.yexp, I.h_w[j], (int) E, s);
            }
            // the shared expert on the GPU while the CPU computes the routed ones
            if (c.n_expert_shared > 0) {
                const int64_t FS = c.n_ff_exp[(size_t) il] * c.n_expert_shared;
                gpu::gemv2(l.gate_shexp, l.up_shexp, d.xn, d.fg, d.fu, d.q8, s);
                gpu::swiglu(d.fg, d.fu, d.fact, (int) FS, c.clamp_shexp[(size_t) il], s);
                gpu::gemv(l.down_shexp, d.fact, d.ysh, d.q8, s);
            }
            if (!I.miss_ids.empty()) {
                I.experts->run(il, I.h_x, I.miss_ids.data(), I.miss_w.data(), (int) I.miss_ids.size(), I.h_yr);
                ok(cudaMemcpyAsync(d.y, I.h_yr, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, d.stream),
                   "experts");
            } else {
                ok(cudaMemsetAsync(d.y, 0, (size_t) E * sizeof(float), d.stream), "memset");
            }
            gpu::add(d.y, d.ygpu, (int) E, s);
            if (c.n_expert_shared > 0) gpu::add(d.y, d.ysh, (int) E, s);
        }
        gpu::hc_post(d.y, d.R, d.pc, d.R2, (int) hc, (int) E, s);
        std::swap(d.R, d.R2);

        if (trace) {
            std::vector<float> r((size_t) (hc * E));
            ok(cudaMemcpyAsync(r.data(), d.R, r.size() * sizeof(float), cudaMemcpyDeviceToHost, d.stream), "trace");
            ok(cudaStreamSynchronize(d.stream), "sync");
            trace->layer_out.push_back(std::move(r));
        }
    }

    // ---- hc_head, output norm, output head
    DevCtx& d = I.ctx[(size_t) I.out_ctx];
    if (cur != I.out_ctx) throw std::runtime_error("the output head is not on the last layer's device");
    ok(cudaSetDevice(d.id), "cudaSetDevice");
    gpu::hc_mean(d.R, d.x, (int) hc, (int) E, d.stream);
    gpu::rms_norm(d.x, d.xn, (int) E, c.rms_eps, I.output_norm, d.stream);
    gpu::gemv(I.output, d.xn, d.out_logits, d.q8, d.stream);
    ok(cudaMemcpyAsync(I.logits.data(), d.out_logits, (size_t) c.n_vocab * sizeof(float), cudaMemcpyDeviceToHost,
                       d.stream), "logits");
    ok(cudaStreamSynchronize(d.stream), "sync");
    I.cache->maintain();
    ++pos_;
    return I.logits;
}

}  // namespace strata::glm
