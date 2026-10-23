// src/glm/glm_main.cpp - strata-glm: GLM-5.3-Flash on the GPUs.
//
//   strata-glm -m MODEL.gguf --tokens-file IDS.txt [--out LOGITS.bin] [--ctx N] [--devices 0,1,2]
//              [--split 24,12,10] [--threads N] [--check-cpu] [--max-tokens N]
//              [--vram-reserve GB] [--expert-profile PATH] [--max-swaps N] [--no-expert-cache]
//              [--pin-experts [--pin-keep-free GB]]
//
// --out writes the --dump-logits layout (tools/glm53_compare.py); with --prefill it holds one row, the logits after
// the prompt. --check-cpu also runs the CPU reference on the same tokens and compares, per position, the residual
// streams after every layer and the logits: the first layer where they part is where a kernel is wrong. Whether
// the logits pass is check.hpp's rule, with the limits for the model's weights (quantized or float).
// --pin-experts copies the routed experts into page-locked RAM before the engine starts (pinned_experts.hpp), as
// long as MemAvailable stays above --pin-keep-free GB (16); the VRAM expert cache is filled from there.
// --prefill-rows N: of each --prefill-chunk, the tokens run at a time outside the routed experts (default: as many
// as fit 512 MiB of attention buffers, at most 512); a test setting, to make the chunks split.
// --kv-in-ram keeps the MLA layers' f16 latent cache (11 KB a token for the 3-bit file) in page-locked host RAM,
// with only the indexer's dense range (its first max_sel rows) in VRAM too; the VRAM it frees goes to the expert
// cache. Each token's selected rows, and a prompt batch's, are read over PCIe.
// --prefill-spread: with the experts page-locked (--pin-experts) and more than one card, a prompt chunk's streamed
// experts of each layer are shared out over every card, not only the one that holds the layer; each card copies its
// share in from RAM and sends back the weighted sum of its experts' outputs.
#include "strata/glm/check.hpp"
#include "strata/glm/cpu_ref.hpp"
#include "strata/glm/engine.hpp"
#include "strata/glm/pinned_experts.hpp"
#include "strata/glm/serve_loop.hpp"
#include "strata/glm/serve_signals.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

bool read_ids(const std::string& path, std::vector<int32_t>& out) {
    std::ifstream in(path);
    if (!in) return false;
    std::stringstream ss;
    ss << in.rdbuf();
    std::string text = ss.str();
    for (char& ch : text)
        if (ch == ',') ch = ' ';
    std::istringstream is(text);
    long long v;
    while (is >> v) out.push_back((int32_t) v);
    return is.eof() && !out.empty();
}

template <class T>
std::vector<T> parse_list(const std::string& s) {
    std::vector<T> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back((T) std::atof(item.c_str()));
    return out;
}

/// Whether some weight matrix of the model is of a type other than F32 and F16: check.hpp's limits differ.
bool quantized_weights(const strata::glm::Model& m) {
    const auto& g = m.gguf();
    for (size_t s = 0; s < g.size(); ++s)
        for (const auto& t : g.shard(s).tensors())
            if (t.shape.size() >= 2 && t.shape[1] > 1 && !strata::glm::float_weight_type(t.type)) return true;
    return false;
}

/// --pin-experts: the experts into page-locked RAM, with a line on the progress and one on the result.
std::unique_ptr<strata::glm::PinnedExperts> pin_experts(strata::glm::Model& model, int threads, double keep_gb) {
    const auto t0 = std::chrono::steady_clock::now();
    auto p = std::make_unique<strata::glm::PinnedExperts>(
        model, threads, (size_t) (std::max(0.0, keep_gb) * 1073741824.0), [&](size_t done, size_t total) {
            const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::fprintf(stderr, "pinning the experts: %.1f of %.1f GB (%.0f s, %.2f GB/s)\n", done / 1e9, total / 1e9,
                         s, s > 0 ? done / 1e9 / s : 0.0);
        });
    std::fprintf(stderr, "experts in page-locked RAM: %d of %d MoE layers, %.1f GB%s%s\n", p->layers(),
                 p->layers() + p->left(), p->bytes() / 1e9, p->left() ? "; the rest stay mapped: " : "",
                 p->why().c_str());
    return p;
}

struct GpuEvaluator : strata::glm::Evaluator {
    strata::glm::GpuEngine& e;
    explicit GpuEvaluator(strata::glm::GpuEngine& eng) : e(eng) {}
    void reset() override { e.reset(); }
    const std::vector<float>& eval(int32_t t) override { return e.eval(t); }
    int64_t hits() const override { return e.cache_hits(); }
    int64_t lookups() const override { return e.cache_lookups(); }
    uint64_t ticks() const override { return e.ticks(); }
    int64_t pos() const override { return e.pos(); }
    const std::vector<float>& prefill(const int32_t* ids, int64_t n,
                                      const std::function<bool(int64_t)>& progress) override {
        return e.prefill(ids, n, progress);
    }
    bool can_snapshot() const override { return true; }
    std::shared_ptr<strata::glm::Snapshot> snapshot(bool full) override { return e.snapshot(full); }
    size_t snapshot_bytes(bool full) const override { return e.snapshot_bytes(full); }
    void restore(const strata::glm::Snapshot& s) override { e.restore(s); }
};

int usage() {
    std::fprintf(stderr,
                 "usage: strata-glm -m MODEL.gguf --tokens-file IDS.txt [--out LOGITS.bin] [--ctx N]\n"
                 "                  [--devices 0,1,2] [--split 24,12,10] [--threads N] [--check-cpu]\n"
                 "                  [--max-tokens N] [--vram-reserve GB] [--expert-profile PATH]\n"
                 "                  [--max-swaps N] [--cache-decay D] [--no-expert-cache] [--prefill] [--prefill-chunk N]\n"
                 "                  [--pin-experts [--pin-keep-free GB]] [--prefill-rows N] [--kv-in-ram]\n"
                 "                  [--prefill-spread]\n"
                 "       strata-glm -m MODEL.gguf --serve [--ctx N] [--eos-ids A,B] [--park-mib N] [--no-warm]\n"
                 "                  [--checkpoint-min N] [same GPU and cache options]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, ids_path, out_path;
    strata::glm::EngineOptions opt;
    bool check_cpu = false, serve = false, use_prefill = false, warm = true, pin = false;
    double pin_keep_gb = 16.0;
    long long max_tokens = 0;
    std::vector<int32_t> eos;
    strata::glm::ServeOptions so;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (a == "-m" && (v = next())) model_path = v;
        else if (a == "--tokens-file" && (v = next())) ids_path = v;
        else if (a == "--out" && (v = next())) out_path = v;
        else if (a == "--ctx" && (v = next())) opt.max_ctx = std::atoll(v);
        else if (a == "--devices" && (v = next())) opt.devices = parse_list<int>(v);
        else if (a == "--split" && (v = next())) opt.split = parse_list<double>(v);
        else if ((a == "--threads" || a == "-t") && (v = next())) opt.cpu_threads = std::atoi(v);
        else if (a == "--max-tokens" && (v = next())) max_tokens = std::atoll(v);
        else if (a == "--check-cpu") check_cpu = true;
        else if (a == "--vram-reserve" && (v = next())) opt.cache.vram_reserve_gb = std::atof(v);
        else if (a == "--expert-profile" && (v = next())) opt.cache.profile = v;
        else if (a == "--max-swaps" && (v = next())) opt.cache.max_swaps_per_token = std::atoi(v);
        else if (a == "--cache-decay" && (v = next())) opt.cache.decay = std::atof(v);
        else if (a == "--no-expert-cache") opt.cache.disabled = true;
        else if (a == "--serve") serve = true;
        else if (a == "--no-warm") warm = false;
        else if (a == "--pin-experts") pin = true;
        else if (a == "--pin-keep-free" && (v = next())) pin_keep_gb = std::atof(v);
        else if (a == "--prefill") use_prefill = true;
        else if (a == "--prefill-chunk" && (v = next())) opt.prefill_chunk = std::atoi(v);
        else if (a == "--prefill-rows" && (v = next())) opt.prefill_rows = std::atoi(v);
        else if (a == "--kv-in-ram") opt.kv_in_ram = true;
        else if (a == "--prefill-spread") opt.prefill_spread = true;
        else if (a == "--eos-ids" && (v = next())) eos = parse_list<int32_t>(v);
        else if (a == "--park-mib" && (v = next())) so.park_bytes = (size_t) (std::max(0.0, std::atof(v)) * 1048576.0);
        else if (a == "--checkpoint-min" && (v = next())) so.checkpoint_min = std::atoll(v);
        else return usage();
    }
    if (serve && !model_path.empty()) {
        strata::glm::serve_signals_starting();   // ends with its server while it starts (serve_signals.hpp)
        try {
            strata::glm::Model model(model_path);
            const int copy_threads = opt.cpu_threads > 0 ? std::min(opt.cpu_threads, 16) : 8;
            std::unique_ptr<strata::glm::PinnedExperts> pinned;
            if (pin) pinned = pin_experts(model, copy_threads, pin_keep_gb);   // before the cache fills from them
            strata::glm::GpuEngine eng(model, opt);
            for (const auto& line : eng.placement()) std::fprintf(stderr, "%s\n", line.c_str());
            if (warm && (!pinned || pinned->left() > 0)) {
                // the experts into the page cache before READY, so the first prompt does not read them from disk
                const auto w0 = std::chrono::steady_clock::now();
                model.warm_experts(copy_threads, [&](size_t done, size_t total) {
                    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
                    std::fprintf(stderr, "warming the experts: %.1f of %.1f GB (%.0f s, %.2f GB/s)\n", done / 1e9,
                                 total / 1e9, s, s > 0 ? done / 1e9 / s : 0.0);
                });
            }
            GpuEvaluator ev(eng);
            so.max_ctx = opt.max_ctx;
            so.n_vocab = model.config().n_vocab;
            so.eos_ids = eos;
            so.info.push_back("backend=cuda");
            so.info.push_back("expert_slots=" + std::to_string(eng.cache_slots()));
            so.info.push_back("expert_cache_mib=" + std::to_string(eng.cache_bytes() >> 20));
            strata::glm::serve_signals_serving();   // the server ends it with QUIT, which saves the profile below
            const int rc = strata::glm::serve_loop(ev, so);
            eng.save_profile();
            return rc;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata-glm: %s\n", e.what());
            return 1;
        }
    }
    if (model_path.empty() || ids_path.empty()) return usage();

    try {
        std::vector<int32_t> ids;
        if (!read_ids(ids_path, ids)) {
            std::fprintf(stderr, "cannot read token ids from %s\n", ids_path.c_str());
            return 1;
        }
        if (max_tokens > 0 && (long long) ids.size() > max_tokens) ids.resize((size_t) max_tokens);
        // the prompt check decodes a few tokens past the prompt on both engines: room for them too
        const int check_decode = use_prefill && check_cpu ? 4 : 0;
        const int64_t n_ctx = (int64_t) ids.size() + check_decode;
        if (n_ctx > opt.max_ctx) opt.max_ctx = n_ctx;

        const auto t0 = std::chrono::steady_clock::now();
        strata::glm::Model model(model_path);
        const auto& c = model.config();
        std::printf("model: %lld layers, n_embd %lld, vocab %lld, %lld experts top-%lld\n", (long long) c.n_layer,
                    (long long) c.n_embd, (long long) c.n_vocab, (long long) c.n_expert, (long long) c.n_expert_used);
        std::unique_ptr<strata::glm::PinnedExperts> pinned;
        if (pin) pinned = pin_experts(model, opt.cpu_threads > 0 ? std::min(opt.cpu_threads, 16) : 8, pin_keep_gb);
        strata::glm::GpuEngine eng(model, opt);
        for (const auto& line : eng.placement()) std::printf("%s\n", line.c_str());
        std::unique_ptr<strata::glm::CpuRef> ref;
        if (check_cpu) ref = std::make_unique<strata::glm::CpuRef>(model, n_ctx, opt.cpu_threads > 0 ? opt.cpu_threads : strata::glm::ThreadPool::physical_cores());
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("loaded in %.1f s\n", std::chrono::duration<double>(t1 - t0).count());

        FILE* out = nullptr;
        if (!out_path.empty()) {
            out = std::fopen(out_path.c_str(), "wb");
            if (!out) {
                std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
                return 1;
            }
            // the prompt path gives the logits of its last token only: one row
            const int32_t hdr[2] = {(int32_t) c.n_vocab, use_prefill ? 1 : (int32_t) ids.size()};
            std::fwrite(hdr, sizeof hdr, 1, out);
        }

        if (use_prefill) {
            // the prompt in chunks on the GPU, against the CPU reference token by token; then a few decoded tokens
            // on both, which checks the state the chunks left (KDA, the latent cache, the pooled keys)
            const auto a = std::chrono::steady_clock::now();
            std::vector<float> lg = eng.prefill(ids.data(), (int64_t) ids.size());
            const double s_pp = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
            std::printf("prefill: %zu tokens in %.2f s (%.1f tokens/s)\n", ids.size(), s_pp, (double) ids.size() / s_pp);
            if (out) {
                std::fwrite(lg.data(), sizeof(float), lg.size(), out);
                std::fclose(out);
            }
            if (!check_cpu) {
                eng.save_profile();
                return 0;
            }
            const auto am = [](const std::vector<float>& v) { return (int32_t) (std::max_element(v.begin(), v.end()) - v.begin()); };
            const strata::glm::CheckLimits lim = strata::glm::check_limits(quantized_weights(model));
            strata::glm::LogitCheck chk;
            std::vector<float> lc;
            for (int32_t id : ids) lc = ref->eval(id);
            double dl = chk.add(lc, lg);
            std::printf("after the prompt: logits relative diff %.3g, KL %.3g, top-1 %s\n", dl,
                        strata::glm::kl_div(lc, lg), am(lc) == am(lg) ? "same" : "DIFFERENT");
            for (int step = 0; step < check_decode; ++step) {
                const int32_t next = am(lc);
                lc = ref->eval(next);
                lg = eng.eval(next);
                dl = chk.add(lc, lg);
                std::printf("decode %d: logits relative diff %.3g, KL %.3g, top-1 %s\n", step, dl,
                            strata::glm::kl_div(lc, lg), am(lc) == am(lg) ? "same" : "DIFFERENT");
            }
            const bool pass = chk.pass(lim);
            std::printf("check-prefill: %s: %s\n", chk.summary(lim).c_str(), pass ? "PASS" : "FAIL");
            return pass ? 0 : 1;
        }

        strata::glm::Trace tg, tc;
        const strata::glm::CheckLimits lim = strata::glm::check_limits(quantized_weights(model));
        strata::glm::LogitCheck chk;
        strata::glm::LayerCheck layers = strata::glm::layer_check(lim);
        double gpu_s = 0;
        for (size_t i = 0; i < ids.size(); ++i) {
            const auto a = std::chrono::steady_clock::now();
            const std::vector<float>& lg = eng.eval(ids[i], check_cpu ? &tg : nullptr);
            gpu_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
            if (out) std::fwrite(lg.data(), sizeof(float), lg.size(), out);
            if (check_cpu) {
                const std::vector<float>& lc = ref->eval(ids[i], &tc);
                const double dl = chk.add(lc, lg);
                const auto am = [](const std::vector<float>& v) { return std::max_element(v.begin(), v.end()) - v.begin(); };
                for (size_t l = 0; l < tc.layer_out.size(); ++l) {
                    const double d = layers.add(tc.layer_out[l], tg.layer_out[l]);
                    if (layers.parted(d)) {
                        std::printf("position %zu: layer %zu residual differs by %.3g (relative)%s\n", i, l, d,
                                    model.config().is_kda((int64_t) l) ? " [KDA]" : " [MLA]");
                        break;
                    }
                }
                if (i < 4 || (i + 1) % 16 == 0)
                    std::printf("position %zu: logits relative diff %.3g, KL %.3g, top-1 %s\n", i, dl,
                                strata::glm::kl_div(lc, lg), am(lc) == am(lg) ? "same" : "DIFFERENT");
            } else if ((i + 1) % 16 == 0 || i + 1 == ids.size()) {
                std::printf("position %zu/%zu\n", i + 1, ids.size());
            }
            std::fflush(stdout);
        }
        if (out) std::fclose(out);
        std::printf("%zu tokens, %.2f tokens/s (one at a time), expert cache hit rate %.1f%%\n", ids.size(),
                    (double) ids.size() / gpu_s, 100.0 * eng.cache_hit_rate());
        eng.save_profile();
        if (check_cpu) {
            const bool pass = chk.pass(lim);
            std::printf("check-cpu: %s; worst layer diff %.3g: %s\n", chk.summary(lim).c_str(), layers.worst,
                        pass ? "PASS" : "FAIL");
            return pass ? 0 : 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-glm: %s\n", e.what());
        return 1;
    }
    return 0;
}
