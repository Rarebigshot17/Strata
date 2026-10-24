// src/glm/engine.cu - GLM-5.3-Flash on the GPUs, one token at a time (engine.hpp).
//
// Each block below names the cpu_ref.cpp function it reproduces. The routed experts run on the CPU (CpuExperts)
// while the GPU computes the shared expert; the indexer's top pools are chosen on the host from the GPU's
// scores, which is also where cpu_ref chooses them.
#include "strata/glm/engine.hpp"

#include "engine_impl.hpp"
#include "strata/glm/gpu_kernels.hpp"
#include "strata/glm/gpu_prefill.hpp"

#include "ggml.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>

namespace strata::glm {

using namespace detail;

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
        // what grows with the context: the f16 latent cache and, where the layer runs its own indexer, the fp32
        // pooled keys (l.pooled). The other context-sized buffers (pscore, the prompt arena's pool scores) are on
        // every device, whichever layers it holds.
        if (!c.is_kda(il)) {
            // with --kv-in-ram only the dense range's rows (max_sel) are in VRAM
            const int64_t rows = opt.kv_in_ram ? std::min<int64_t>(opt.max_ctx, c.indexer_top_k + c.indexer_kpool - 1)
                                               : opt.max_ctx;
            b += (double) rows * c.kv_lora_rank * 2.0;
            if (c.indexer_full[(size_t) il])
                b += (double) (opt.max_ctx / c.indexer_kpool + 1) * c.indexer_head_size * 4.0;
        }
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
                                     c.kda_head_dim, maxH * c.head_k_mla, maxH * c.kv_lora_rank});

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
        d.n_sel1 = dalloc<int32_t>(1, b);
        if (opt.kv_in_ram) {
            d.kvsel = dalloc<uint16_t>((size_t) (I.max_sel * c.kv_lora_rank), b);
            d.iota = dalloc<int32_t>((size_t) I.max_sel, b);
            std::vector<int32_t> io((size_t) I.max_sel);
            std::iota(io.begin(), io.end(), 0);
            ok(cudaMemcpy(d.iota, io.data(), io.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "iota");
        }
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
        d.q8x = dalloc<uint8_t>(gpu::q8_scratch_bytes(E), b);
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
            if (opt.kv_in_ram) {
                // page-locked and mapped: the kernels write rows into it and read the selected rows over PCIe
                const size_t n = (size_t) (opt.max_ctx * c.kv_lora_rank);
                void* p = nullptr;
                ok(cudaHostAlloc(&p, n * sizeof(uint16_t), cudaHostAllocMapped | cudaHostAllocPortable),
                   "cudaHostAlloc for the latent cache (--kv-in-ram)");
                std::memset(p, 0, n * sizeof(uint16_t));
                I.kv_host.push_back(p);
                l.cache = static_cast<uint16_t*>(p);
                l.head = dalloc<uint16_t>((size_t) (I.max_sel * c.kv_lora_rank), b);
            } else {
                l.cache = dalloc<uint16_t>((size_t) (opt.max_ctx * c.kv_lora_rank), b);
            }
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

    // ---- the prompt path's buffers, then the VRAM expert cache: what is left on each device, minus the reserve
    setup_prefill();
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
    const int threads = opt.cpu_threads > 0 ? opt.cpu_threads : ThreadPool::physical_cores();
    I.pool = std::make_unique<ThreadPool>(threads);
    I.timing = std::getenv("STRATA_GLM_TIMING") && std::atoi(std::getenv("STRATA_GLM_TIMING")) > 0;
    if (I.timing) I.prev_ids.assign((size_t) (c.n_layer * c.n_expert_used), -1);
    I.experts = std::make_unique<CpuExperts>(model, *I.pool);
    I.h_x = halloc<float>((size_t) E);
    I.h_yr = halloc<float>((size_t) E);
    I.h_w = halloc<float>((size_t) c.n_expert_used);
    I.h_ids = halloc<int32_t>((size_t) c.n_expert_used);
    if (opt.decode_help > 0) {
        if (!I.blob_max || std::find(I.experts_pinned.begin(), I.experts_pinned.end(), 1) == I.experts_pinned.end()) {
            std::fprintf(stderr, "--decode-help needs the experts page-locked (--pin-experts) and the prompt path's "
                                 "buffers (--prefill-chunk 2 or more); off\n");
            I.opt.decode_help = 0;
        } else {
            I.h_yhelp.assign(I.ctx.size(), nullptr);
            for (size_t di = 0; di < I.ctx.size(); ++di) {
                ok(cudaSetDevice(I.ctx[di].id), "cudaSetDevice");
                ok(cudaEventCreateWithFlags(&I.ctx[di].ev_help, cudaEventDisableTiming), "cudaEventCreate");
                I.h_yhelp[di] = halloc<float>((size_t) E);
            }
        }
    }
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
    if (I.opt.kv_in_ram) {
        char line[160];
        std::snprintf(line, sizeof line, "latent cache in page-locked RAM: %.2f GB (%zu MLA layers)",
                      (double) I.kv_host.size() * (double) I.opt.max_ctx * (double) c.kv_lora_rank * 2.0 / 1e9,
                      I.kv_host.size());
        placement_.push_back(line);
    }
    if (I.opt.prefill_chunk > 1) {
        char line[160];
        std::snprintf(line, sizeof line, "prompt chunks: %d tokens, %d at a time outside the routed experts",
                      I.opt.prefill_chunk, I.prefill_rows);
        placement_.push_back(line);
    }
    reset();
}

double GpuEngine::cache_hit_rate() const {
    const auto& c = *impl_->cache;
    return c.lookups() ? (double) c.hits() / (double) c.lookups() : 0.0;
}

std::string GpuEngine::cache_summary() const { return impl_->cache->summary(); }

int64_t GpuEngine::cache_hits() const { return impl_->cache->hits(); }

int64_t GpuEngine::cache_lookups() const { return impl_->cache->lookups(); }

int64_t GpuEngine::cache_slots() const { return impl_->cache->total_slots(); }

size_t GpuEngine::cache_bytes() const { return impl_->cache->total_bytes(); }

void GpuEngine::save_profile() const { impl_->cache->save_profile(); }

GpuEngine::~GpuEngine() {
    // device memory is released with the process; streams and pinned buffers here
    if (!impl_) return;
    if (impl_->timing && impl_->t_tokens > 0) {
        const Impl& I = *impl_;
        const double n = (double) I.t_tokens, ms = 1e3 / n;
        std::fprintf(stderr,
                     "timing over %lld tokens, ms per token: eval %.2f = waiting for the GPU at routing %.2f + CPU "
                     "experts %.2f + waiting for the GPUs' share of them %.2f + waiting for the logits %.2f + cache "
                     "upkeep %.2f + the rest (launches, host) %.2f\n",
                     (long long) I.t_tokens, I.t_eval * ms, I.t_route_wait * ms, I.t_cpu * ms, I.t_help_wait * ms,
                     I.t_logits_wait * ms, I.t_maintain * ms,
                     (I.t_eval - I.t_route_wait - I.t_cpu - I.t_help_wait - I.t_logits_wait - I.t_maintain) * ms);
        if (I.r_layers > 0) {
            const double nl = (double) I.r_layers;
            std::fprintf(stderr,
                         "routing over %lld MoE layers of tokens that follow a token: experts per layer %.2f for the "
                         "token and %.2f for it and the one before together, of them on the CPU %.2f and %.2f\n",
                         (long long) I.r_layers, (double) I.c.n_expert_used, I.r_union / nl, I.r_miss / nl,
                         I.r_union_miss / nl);
        }
        std::fprintf(stderr, "expert cache over the run: %lld swaps and %lld fills queued, %.1f%% of lookups hit\n",
                     (long long) I.cache->swaps_queued(), (long long) I.cache->fills_queued(),
                     I.cache->lookups() ? 100.0 * (double) I.cache->hits() / (double) I.cache->lookups() : 0.0);
    }
    for (auto& d : impl_->ctx)
        if (d.stream) cudaStreamDestroy(d.stream);
    for (void* p : {(void*) impl_->h_x, (void*) impl_->h_yr, (void*) impl_->h_w, (void*) impl_->h_ids})
        if (p) cudaFreeHost(p);
    for (void* p : impl_->kv_host) cudaFreeHost(p);
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
    // the clears ran on the legacy stream, which the engine's non-blocking streams do not wait for
    for (auto& d : I.ctx) {
        ok(cudaSetDevice(d.id), "cudaSetDevice");
        ok(cudaDeviceSynchronize(), "sync");
    }
}

namespace {

/// Per layer (empty where the layer has none). conv holds the three windows one after another.
struct GpuSnapshot : Snapshot {
    std::vector<std::vector<float>> S, conv, pend_k, pend_g, pooled;
    std::vector<std::vector<uint16_t>> cache;
};

}  // namespace

std::shared_ptr<Snapshot> GpuEngine::snapshot(bool full) {
    Impl& I = *impl_;
    const Config& c = I.c;
    for (auto& d : I.ctx) {
        ok(cudaSetDevice(d.id), "cudaSetDevice");
        ok(cudaDeviceSynchronize(), "sync");
    }
    auto sn = std::make_shared<GpuSnapshot>();
    sn->pos = pos_;
    sn->full = full;
    const size_t NL = (size_t) c.n_layer;
    for (auto* v : {&sn->S, &sn->conv, &sn->pend_k, &sn->pend_g, &sn->pooled}) v->resize(NL);
    sn->cache.resize(NL);
    size_t bytes = 0;
    auto down = [&](auto& dst, const void* src, size_t n, size_t at = 0) {
        if (dst.size() < at + n) dst.resize(at + n);
        ok(cudaMemcpy(dst.data() + at, src, n * sizeof(dst[0]), cudaMemcpyDeviceToHost), "cudaMemcpy");
        bytes += n * sizeof(dst[0]);
    };
    const int64_t ID = c.indexer_head_size, P = c.indexer_kpool;
    for (int64_t il = 0; il < c.n_layer; ++il) {
        const DevLayer& l = I.L[(size_t) il];
        const size_t i = (size_t) il;
        ok(cudaSetDevice(l.dev), "cudaSetDevice");
        if (c.is_kda(il)) {
            const int64_t H = c.n_head[i], D = c.kda_head_dim, n_conv = (c.ssm_d_conv - 1) * H * D;
            down(sn->S[i], l.S, (size_t) (H * D * D));
            for (int j = 0; j < 3; ++j) down(sn->conv[i], l.conv_st[j], (size_t) n_conv, (size_t) (j * n_conv));
            continue;
        }
        if (c.indexer_full[i]) {                       // the pool being filled: rows t % P, overwritten later
            down(sn->pend_k[i], l.pend_k, (size_t) (P * ID));
            down(sn->pend_g[i], l.pend_g, (size_t) (P * ID));
        }
        if (full) {
            if (l.head) {                              // --kv-in-ram: the cache is host memory already
                const size_t n = (size_t) (pos_ * c.kv_lora_rank);
                sn->cache[i].assign(l.cache, l.cache + n);
                bytes += n * sizeof(uint16_t);
            } else {
                down(sn->cache[i], l.cache, (size_t) (pos_ * c.kv_lora_rank));
            }
            if (c.indexer_full[i]) down(sn->pooled[i], l.pooled, (size_t) (pos_ / P * ID));
        }
    }
    sn->bytes = bytes;
    return sn;
}

size_t GpuEngine::snapshot_bytes(bool full) const {
    const Config& c = impl_->c;
    const int64_t ID = c.indexer_head_size, P = c.indexer_kpool;
    size_t b = 0;
    for (int64_t il = 0; il < c.n_layer; ++il) {
        const size_t i = (size_t) il;
        if (c.is_kda(il)) {
            const int64_t H = c.n_head[i], D = c.kda_head_dim;
            b += (size_t) (H * D * D + 3 * (c.ssm_d_conv - 1) * H * D) * sizeof(float);
            continue;
        }
        if (c.indexer_full[i]) b += (size_t) (2 * P * ID) * sizeof(float);
        if (full) {
            b += (size_t) (pos_ * c.kv_lora_rank) * sizeof(uint16_t);
            if (c.indexer_full[i]) b += (size_t) (pos_ / P * ID) * sizeof(float);
        }
    }
    return b;
}

void GpuEngine::restore(const Snapshot& snap) {
    Impl& I = *impl_;
    const Config& c = I.c;
    const auto* sn = dynamic_cast<const GpuSnapshot*>(&snap);
    if (!sn || sn->S.size() != (size_t) c.n_layer) throw std::runtime_error("GpuEngine::restore: not this engine's snapshot");
    if (sn->pos > I.opt.max_ctx) throw std::runtime_error("GpuEngine::restore: the snapshot is longer than the context");
    for (auto& d : I.ctx) {                            // nothing may still be reading the state
        ok(cudaSetDevice(d.id), "cudaSetDevice");
        ok(cudaDeviceSynchronize(), "sync");
    }
    auto up = [&](void* dst, const auto& src, size_t n, size_t at = 0) {
        if (n == 0) return;
        if (src.size() < at + n) throw std::runtime_error("GpuEngine::restore: a truncated snapshot");
        ok(cudaMemcpy(dst, src.data() + at, n * sizeof(src[0]), cudaMemcpyHostToDevice), "cudaMemcpy");
    };
    const int64_t ID = c.indexer_head_size, P = c.indexer_kpool;
    for (int64_t il = 0; il < c.n_layer; ++il) {
        const DevLayer& l = I.L[(size_t) il];
        const size_t i = (size_t) il;
        ok(cudaSetDevice(l.dev), "cudaSetDevice");
        if (c.is_kda(il)) {
            const int64_t H = c.n_head[i], D = c.kda_head_dim, n_conv = (c.ssm_d_conv - 1) * H * D;
            up(l.S, sn->S[i], (size_t) (H * D * D));
            for (int j = 0; j < 3; ++j) up(l.conv_st[j], sn->conv[i], (size_t) n_conv, (size_t) (j * n_conv));
            continue;
        }
        if (c.indexer_full[i]) {
            up(l.pend_k, sn->pend_k[i], (size_t) (P * ID));
            up(l.pend_g, sn->pend_g[i], (size_t) (P * ID));
        }
        if (sn->full) {
            if (l.head) {
                // --kv-in-ram: the host cache, then the dense range's rows into the VRAM head
                const size_t n = (size_t) (sn->pos * c.kv_lora_rank);
                if (sn->cache[i].size() < n) throw std::runtime_error("GpuEngine::restore: a truncated snapshot");
                std::memcpy(l.cache, sn->cache[i].data(), n * sizeof(uint16_t));
                const size_t nh = (size_t) (std::min<int64_t>(sn->pos, I.max_sel) * c.kv_lora_rank);
                if (nh) ok(cudaMemcpy(l.head, l.cache, nh * sizeof(uint16_t), cudaMemcpyHostToDevice), "cudaMemcpy");
            } else {
                up(l.cache, sn->cache[i], (size_t) (sn->pos * c.kv_lora_rank));
            }
            if (c.indexer_full[i]) up(l.pooled, sn->pooled[i], (size_t) (sn->pos / P * ID));
        }
    }
    // pageable host-to-device copies can return before the DMA ends, and the engine's streams do not wait for them
    for (auto& d : I.ctx) {
        ok(cudaSetDevice(d.id), "cudaSetDevice");
        ok(cudaDeviceSynchronize(), "sync");
    }
    pos_ = sn->pos;
}

const std::vector<float>& GpuEngine::eval(int32_t token, Trace* trace) {
    Impl& I = *impl_;
    const Config& c = I.c;
    const Model& m = I.m;
    const int64_t E = c.n_embd, hc = c.hc, t = pos_;
    if (t >= I.opt.max_ctx) throw std::runtime_error("the sequence is longer than the context (--ctx)");
    if (token < 0 || token >= c.n_vocab) throw std::runtime_error("token id outside the vocabulary");
    using tclock = std::chrono::steady_clock;
    const auto sec = [](tclock::time_point a) { return std::chrono::duration<double>(tclock::now() - a).count(); };
    const auto t_eval0 = tclock::now();

    // ---- embedding (hc_init), dequantized on the host
    {
        const Tensor& te = m.tok_embd();
        if (te.type == GGML_TYPE_F32) std::memcpy(I.emb.data(), te.row(token), (size_t) E * sizeof(float));
        else ggml_get_type_traits((ggml_type) te.type)->to_float(te.row(token), I.emb.data(), E);
        for (int64_t h = 1; h < hc; ++h) std::memcpy(I.emb.data() + h * E, I.emb.data(), (size_t) E * sizeof(float));
        DevCtx& d0 = I.ctx[(size_t) I.ctx_of_layer[0]];
        ok(cudaSetDevice(d0.id), "cudaSetDevice");
        // on the first layer's stream: the engine's streams are non-blocking, so their kernels do not wait for a copy
        // on the legacy stream, and a cudaMemcpy from pageable memory can return before its DMA has landed (with an
        // expert swap's copy holding the link, the first layer then read the last token's streams)
        ok(cudaMemcpyAsync(d0.R, I.emb.data(), (size_t) (hc * E) * sizeof(float), cudaMemcpyHostToDevice, d0.stream),
           "embedding");
    }
    if (trace) trace->layer_out.clear();

    int cur = I.ctx_of_layer[0];
    for (int64_t il = 0; il < c.n_layer; ++il) {
        ticks_.fetch_add(1, std::memory_order_relaxed);
        const int ci = I.ctx_of_layer[(size_t) il];
        if (ci != cur) {                                  // the residual streams move to the next device
            DevCtx& from = I.ctx[(size_t) cur];
            DevCtx& to = I.ctx[(size_t) ci];
            ok(cudaSetDevice(from.id), "cudaSetDevice");
            ok(cudaStreamSynchronize(from.stream), "sync");
            // on the next device's stream (cudaMemcpyPeer returns before the copy is done, on the legacy stream)
            ok(cudaMemcpyPeerAsync(to.R, to.id, from.R, from.id, (size_t) (hc * E) * sizeof(float), to.stream),
               "cudaMemcpyPeerAsync");
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
            if (l.head && t < I.max_sel) gpu::store_f16_row(d.kv, l.head, t, (int) KL, s);
            if (l.k_b.n_in == HK && l.k_b.n_out == H * KL) {
                gpu::gemv_heads(l.k_b, (int) H, d.qm, d.q_abs, d.q8, s);
            } else {
                for (int64_t h = 0; h < H; ++h)
                    gpu::gemv(l.k_b.rows(h * KL, KL), d.qm + h * HK, d.q_abs + h * KL, d.q8, s);
            }
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
                // the n_top best of the nv complete pools on the GPU (gpu_prefill.hpp's topk_pools_rows, as the prompt
                // path selects: ties to the lower pool, cells in pool order, then the tail), so the host neither
                // waits for the scores nor sorts them: one sort of (t+1)/4 pools for each of the 11 layers with an
                // indexer of their own took most of a token's time at 458K tokens of context (napster-ai)
                const int64_t nv = (t + 1) / P, n_top = c.indexer_top_k / P;
                if (nv > n_top) {
                    gpu::gemv(l.idx_proj, d.xn, d.wts, d.q8, s);
                    gpu::scale(d.wts, (int) IH, 1.0f / std::sqrt((float) (ID * IH)), s);
                    gpu::kpool_scores(d.iq, l.pooled, d.wts, d.pscore, (int) nv, (int) IH, (int) ID, s);
                }
                gpu::topk_pools_rows(d.pscore, std::max<int64_t>(nv, 1), 1, t, (int) P, (int) n_top,
                                     c.indexer_select_tail, (int) I.max_sel, d.sel, d.n_sel1, s);
                d.dec_sel_have = ++I.dec_sel_version;
                I.dec_sel_ctx = ci;
            } else if (d.dec_sel_have != I.dec_sel_version) {
                // a shared indexer layer on another device than the last full one: its selection from there (that
                // device's stream finished when the layers moved on from it)
                const DevCtx& from = I.ctx[(size_t) I.dec_sel_ctx];
                ok(cudaMemcpyPeerAsync(d.sel, d.id, from.sel, from.id, (size_t) I.max_sel * sizeof(int32_t), d.stream),
                   "selection");
                d.dec_sel_have = I.dec_sel_version;
            }
            // the cells selected: every one up to t while the pools fit in n_top, else n_top pools, and the tail
            const int64_t P = c.indexer_kpool, nv = (t + 1) / P, n_top = std::max<int64_t>(c.indexer_top_k / P, 0);
            const int n_sel = (int) (std::min(nv, n_top) * P + (c.indexer_select_tail ? (t + 1) % P : 0));
            if (n_sel == 0 || n_sel > I.max_sel) throw std::runtime_error("indexer selection of an unexpected size");

            // --kv-in-ram: the selected rows read from host memory into kvsel, then attended in order; in the
            // dense range every selected cell is in the VRAM head
            const uint16_t* kc = l.cache;
            const int32_t* ks = d.sel;
            if (l.head && t < I.max_sel) {
                kc = l.head;
            } else if (l.head) {
                gpu::gather_host_rows(l.cache, d.sel, nullptr, n_sel, (int) KL, d.kvsel, s);
                kc = d.kvsel;
                ks = d.iota;
            }
            gpu::mla_attend(d.q_abs, kc, ks, n_sel, d.scores, d.lat, (int) H, (int) KL,
                            1.0f / std::sqrt((float) HK), s);
            if (l.v_b.n_in == KL && l.v_b.n_out == H * HV) {
                gpu::gemv_heads(l.v_b, (int) H, d.lat, d.heads, d.q8, s);
            } else {
                for (int64_t h = 0; h < H; ++h)
                    gpu::gemv(l.v_b.rows(h * HV, HV), d.lat + h * KL, d.heads + h * HV, d.q8, s);
            }
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
            const auto t_rw = tclock::now();
            ok(cudaStreamSynchronize(d.stream), "sync");
            if (I.timing) I.t_route_wait += sec(t_rw);
            I.cache->record(il, I.h_ids, KU);
            // the cached experts on the GPU, in routing order; the rest go to the CPU
            const int64_t F = c.n_ff_exp[(size_t) il];
            I.miss_ids.clear();
            I.miss_w.clear();
            for (int j = 0; j < KU; ++j) {
                if (I.cache->slot(il, I.h_ids[j]) >= 0) continue;
                I.miss_ids.push_back(I.h_ids[j]);
                I.miss_w.push_back(I.h_w[j]);
            }
            if (I.timing) {
                // the last token's experts that this one does not use, and those of them not cached now
                int32_t* prev = I.prev_ids.data() + (size_t) il * (size_t) KU;
                if (prev[0] >= 0) {
                    int64_t u = KU, um = (int64_t) I.miss_ids.size();
                    for (int j = 0; j < KU; ++j) {
                        if (std::find(I.h_ids, I.h_ids + KU, prev[j]) != I.h_ids + KU) continue;
                        ++u;
                        if (I.cache->slot(il, prev[j]) < 0) ++um;
                    }
                    ++I.r_layers;
                    I.r_union += u;
                    I.r_miss += (int64_t) I.miss_ids.size();
                    I.r_union_miss += um;
                }
                std::copy(I.h_ids, I.h_ids + KU, prev);
            }
            // the GPU's share: launched by this thread while the other CPU threads start on the missed experts
            const auto launch_gpu = [&] {
                ok(cudaMemsetAsync(d.ygpu, 0, (size_t) E * sizeof(float), d.stream), "memset");
                bool xq = false;   // xn quantized into q8x, once for the layer's cached experts
                for (int j = 0; j < KU; ++j) {
                    const int slot = I.cache->slot(il, I.h_ids[j]);
                    if (slot < 0) continue;
                    DevMatrix G, U, Dn;
                    I.cache->matrices(il, slot, G, U, Dn);
                    if (gpu::quantizes_alike(G, U)) {
                        if (!xq) gpu::quantize_for(G, d.xn, d.q8x, s);
                        xq = true;
                        gpu::gemv_pre(G, d.xn, d.q8x, d.fg, s);
                        gpu::gemv_pre(U, d.xn, d.q8x, d.fu, s);
                    } else {
                        gpu::gemv2(G, U, d.xn, d.fg, d.fu, d.q8, s);
                    }
                    gpu::swiglu(d.fg, d.fu, d.fact, (int) F, c.clamp_exp[(size_t) il], s);
                    gpu::gemv(Dn, d.fact, d.yexp, d.q8, s);
                    gpu::axpy(d.ygpu, d.yexp, I.h_w[j], (int) E, s);
                }
                // the shared expert
                if (c.n_expert_shared > 0) {
                    const int64_t FS = c.n_ff_exp[(size_t) il] * c.n_expert_shared;
                    gpu::gemv2(l.gate_shexp, l.up_shexp, d.xn, d.fg, d.fu, d.q8, s);
                    gpu::swiglu(d.fg, d.fu, d.fact, (int) FS, c.clamp_shexp[(size_t) il], s);
                    gpu::gemv(l.down_shexp, d.fact, d.ysh, d.q8, s);
                }
            };
            // --decode-help (page-locked experts only): the last N misses go to the GPUs instead, the first to this
            // card and the next ones to the others (other links to RAM first); each is copied in from RAM by DMA into
            // a staging buffer and computed there while the CPU computes the rest. This card adds its share into
            // ygpu; another card sends the sum of its share back through RAM, added to the CPU's.
            I.help_ids.clear();
            I.help_w.clear();
            if (I.opt.decode_help > 0 && (size_t) il < I.experts_pinned.size() && I.experts_pinned[(size_t) il] &&
                I.cache->layer(il).cacheable)
                while ((int) I.help_ids.size() < I.opt.decode_help && !I.miss_ids.empty()) {
                    I.help_ids.push_back(I.miss_ids.back());
                    I.help_w.push_back(I.miss_w.back());
                    I.miss_ids.pop_back();
                    I.miss_w.pop_back();
                }
            const size_t nd = I.ctx.size();
            std::vector<char> helped(nd, 0);
            bool other_cards = false;
            if (!I.help_ids.empty()) {
                launch_gpu();                                   // ygpu starts with the cached experts
                const ExpertSlots& es = I.cache->layer(il);
                const LayerWeights& lw = I.m.layer(il);
                std::vector<int> hb(nd, 0);
                for (size_t q = 0; q < I.help_ids.size(); ++q) {
                    // this card, then the cards on other links to RAM, then the rest (measure_links)
                    const size_t k = I.help_order.empty() ? ((size_t) ci + q) % nd
                                                          : (size_t) I.help_order[(size_t) ci][q % nd];
                    DevCtx& h = I.ctx[k];
                    ok(cudaSetDevice(h.id), "cudaSetDevice");
                    void* hs = h.stream;
                    if (!helped[k] && k != (size_t) ci) {      // this token's xn to the card, its sum from zero
                        ok(cudaMemcpyAsync(h.xn, I.h_x, (size_t) E * sizeof(float), cudaMemcpyHostToDevice, h.stream),
                           "x to a helper");
                        ok(cudaMemsetAsync(h.ygpu, 0, (size_t) E * sizeof(float), h.stream), "memset");
                        other_cards = true;
                    }
                    helped[k] = 1;
                    const int e = I.help_ids[q];
                    const int bb = hb[k];
                    uint8_t* base = h.dstage[bb];
                    const Tensor g = lw.gate_exps.slice2(e), u = lw.up_exps.slice2(e), dn = lw.down_exps.slice2(e);
                    const size_t db = es.down.row_bytes * (size_t) es.down.n_out;
                    ok(cudaStreamWaitEvent(h.copy_stream, h.ev_free[bb], 0), "cudaStreamWaitEvent");
                    ok(cudaMemcpyAsync(base, g.data, es.up_off, cudaMemcpyHostToDevice, h.copy_stream), "expert");
                    ok(cudaMemcpyAsync(base + es.up_off, u.data, es.down_off - es.up_off, cudaMemcpyHostToDevice,
                                       h.copy_stream), "expert");
                    ok(cudaMemcpyAsync(base + es.down_off, dn.data, db, cudaMemcpyHostToDevice, h.copy_stream),
                       "expert");
                    ok(cudaEventRecord(h.ev_copied[bb], h.copy_stream), "cudaEventRecord");
                    ok(cudaStreamWaitEvent(h.stream, h.ev_copied[bb], 0), "cudaStreamWaitEvent");
                    DevMatrix G = es.gate, U = es.up, Dn = es.down;
                    G.w = base;
                    U.w = base + es.up_off;
                    Dn.w = base + es.down_off;
                    gpu::gemv2(G, U, h.xn, h.fg, h.fu, h.q8, hs);
                    gpu::swiglu(h.fg, h.fu, h.fact, (int) F, c.clamp_exp[(size_t) il], hs);
                    gpu::gemv(Dn, h.fact, h.yexp, h.q8, hs);
                    gpu::axpy(h.ygpu, h.yexp, I.help_w[q], (int) E, hs);
                    ok(cudaEventRecord(h.ev_free[bb], h.stream), "cudaEventRecord");
                    hb[k] ^= 1;
                }
                for (size_t k = 0; k < nd; ++k) {
                    if (!helped[k] || k == (size_t) ci) continue;
                    DevCtx& h = I.ctx[k];
                    ok(cudaSetDevice(h.id), "cudaSetDevice");
                    ok(cudaMemcpyAsync(I.h_yhelp[k], h.ygpu, (size_t) E * sizeof(float), cudaMemcpyDeviceToHost,
                                       h.stream), "a helper's share back");
                    ok(cudaEventRecord(h.ev_help, h.stream), "cudaEventRecord");
                }
                ok(cudaSetDevice(d.id), "cudaSetDevice");
            }
            if (I.miss_ids.empty() && I.help_ids.empty()) launch_gpu();
            if (!I.miss_ids.empty() || other_cards) {
                if (!I.miss_ids.empty()) {
                    const auto t_c = tclock::now();
                    I.experts->run(il, I.h_x, I.miss_ids.data(), I.miss_w.data(), (int) I.miss_ids.size(), I.h_yr,
                                   I.help_ids.empty() ? std::function<void()>(launch_gpu) : std::function<void()>());
                    if (I.timing) I.t_cpu += sec(t_c);
                } else {
                    std::fill(I.h_yr, I.h_yr + E, 0.0f);
                }
                if (other_cards) {
                    const auto t_h = tclock::now();
                    for (size_t k = 0; k < nd; ++k) {
                        if (!helped[k] || k == (size_t) ci) continue;
                        ok(cudaEventSynchronize(I.ctx[k].ev_help), "a helper's share");
                        for (int64_t i = 0; i < E; ++i) I.h_yr[i] += I.h_yhelp[k][i];
                    }
                    if (I.timing) I.t_help_wait += sec(t_h);
                }
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
    const auto t_lw = tclock::now();
    ok(cudaStreamSynchronize(d.stream), "sync");
    const auto t_m = tclock::now();
    I.cache->maintain();
    if (I.timing) {
        I.t_logits_wait += std::chrono::duration<double>(t_m - t_lw).count();
        I.t_maintain += sec(t_m);
        I.t_eval += sec(t_eval0);
        ++I.t_tokens;
    }
    ++pos_;
    return I.logits;
}

}  // namespace strata::glm
