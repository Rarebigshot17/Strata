// src/glm/glm_main.cpp - strata-glm: GLM-5.3-Flash on the GPUs.
//
//   strata-glm -m MODEL.gguf --tokens-file IDS.txt [--out LOGITS.bin] [--ctx N] [--devices 0,1,2]
//              [--split 24,12,10] [--threads N] [--check-cpu] [--max-tokens N]
//              [--vram-reserve GB] [--expert-profile PATH] [--max-swaps N] [--no-expert-cache]
//
// --out writes the --dump-logits layout (tools/glm53_compare.py). --check-cpu also runs the CPU reference on the
// same tokens and compares, per position, the residual streams after every layer and the logits: the first layer
// where they part is where a kernel is wrong.
#include "strata/glm/cpu_ref.hpp"
#include "strata/glm/engine.hpp"
#include "strata/glm/serve_loop.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
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

/// max |a - b| relative to max |a|
double rel_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0, den = 1e-30;
    for (size_t i = 0; i < a.size(); ++i) {
        num = std::max(num, (double) std::fabs(a[i] - b[i]));
        den = std::max(den, (double) std::fabs(a[i]));
    }
    return num / den;
}

struct GpuEvaluator : strata::glm::Evaluator {
    strata::glm::GpuEngine& e;
    explicit GpuEvaluator(strata::glm::GpuEngine& eng) : e(eng) {}
    void reset() override { e.reset(); }
    const std::vector<float>& eval(int32_t t) override { return e.eval(t); }
    int64_t hits() const override { return e.cache_hits(); }
    int64_t lookups() const override { return e.cache_lookups(); }
    int64_t pos() const override { return e.pos(); }
    const std::vector<float>& prefill(const int32_t* ids, int64_t n,
                                      const std::function<bool(int64_t)>& progress) override {
        return e.prefill(ids, n, progress);
    }
};

int usage() {
    std::fprintf(stderr,
                 "usage: strata-glm -m MODEL.gguf --tokens-file IDS.txt [--out LOGITS.bin] [--ctx N]\n"
                 "                  [--devices 0,1,2] [--split 24,12,10] [--threads N] [--check-cpu]\n"
                 "                  [--max-tokens N] [--vram-reserve GB] [--expert-profile PATH]\n"
                 "                  [--max-swaps N] [--no-expert-cache] [--prefill] [--prefill-chunk N]\n"
                 "       strata-glm -m MODEL.gguf --serve [--ctx N] [--eos-ids A,B] [same GPU and cache options]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, ids_path, out_path;
    strata::glm::EngineOptions opt;
    bool check_cpu = false, serve = false, use_prefill = false;
    long long max_tokens = 0;
    std::vector<int32_t> eos;
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
        else if (a == "--no-expert-cache") opt.cache.disabled = true;
        else if (a == "--serve") serve = true;
        else if (a == "--prefill") use_prefill = true;
        else if (a == "--prefill-chunk" && (v = next())) opt.prefill_chunk = std::atoi(v);
        else if (a == "--eos-ids" && (v = next())) eos = parse_list<int32_t>(v);
        else return usage();
    }
    if (serve && !model_path.empty()) {
        try {
            strata::glm::Model model(model_path);
            strata::glm::GpuEngine eng(model, opt);
            for (const auto& line : eng.placement()) std::fprintf(stderr, "%s\n", line.c_str());
            GpuEvaluator ev(eng);
            strata::glm::ServeOptions so;
            so.max_ctx = opt.max_ctx;
            so.eos_ids = eos;
            so.info.push_back("backend=cuda");
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
        if ((int64_t) ids.size() > opt.max_ctx) opt.max_ctx = (int64_t) ids.size();

        const auto t0 = std::chrono::steady_clock::now();
        strata::glm::Model model(model_path);
        const auto& c = model.config();
        std::printf("model: %lld layers, n_embd %lld, vocab %lld, %lld experts top-%lld\n", (long long) c.n_layer,
                    (long long) c.n_embd, (long long) c.n_vocab, (long long) c.n_expert, (long long) c.n_expert_used);
        strata::glm::GpuEngine eng(model, opt);
        for (const auto& line : eng.placement()) std::printf("%s\n", line.c_str());
        std::unique_ptr<strata::glm::CpuRef> ref;
        if (check_cpu) ref = std::make_unique<strata::glm::CpuRef>(model, (int64_t) ids.size(), opt.cpu_threads > 0 ? opt.cpu_threads : (int) std::thread::hardware_concurrency());
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("loaded in %.1f s\n", std::chrono::duration<double>(t1 - t0).count());

        FILE* out = nullptr;
        if (!out_path.empty()) {
            out = std::fopen(out_path.c_str(), "wb");
            if (!out) {
                std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
                return 1;
            }
            const int32_t hdr[2] = {(int32_t) c.n_vocab, (int32_t) ids.size()};
            std::fwrite(hdr, sizeof hdr, 1, out);
        }

        if (use_prefill) {
            // the prompt in chunks on the GPU, against the CPU reference token by token; then a few decoded tokens
            // on both, which checks the state the chunks left (KDA, the latent cache, the pooled keys)
            const auto a = std::chrono::steady_clock::now();
            std::vector<float> lg = eng.prefill(ids.data(), (int64_t) ids.size());
            const double s_pp = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
            std::printf("prefill: %zu tokens in %.2f s (%.1f tokens/s)\n", ids.size(), s_pp, (double) ids.size() / s_pp);
            if (!check_cpu) {
                if (out_path.size()) {
                    FILE* f = std::fopen(out_path.c_str(), "wb");
                    const int32_t hdr[2] = {(int32_t) c.n_vocab, 1};
                    std::fwrite(hdr, sizeof hdr, 1, f);
                    std::fwrite(lg.data(), sizeof(float), lg.size(), f);
                    std::fclose(f);
                }
                eng.save_profile();
                return 0;
            }
            const auto am = [](const std::vector<float>& v) { return (int32_t) (std::max_element(v.begin(), v.end()) - v.begin()); };
            std::vector<float> lc;
            for (int32_t id : ids) lc = ref->eval(id);
            double worst = rel_diff(lc, lg);
            int agree = am(lc) == am(lg);
            std::printf("after the prompt: logits relative diff %.3g, top-1 %s\n", worst, agree ? "same" : "DIFFERENT");
            for (int step = 0; step < 4; ++step) {
                const int32_t next = am(lc);
                lc = ref->eval(next);
                lg = eng.eval(next);
                const double dl = rel_diff(lc, lg);
                worst = std::max(worst, dl);
                agree += am(lc) == am(lg);
                std::printf("decode %d: logits relative diff %.3g, top-1 %s\n", step, dl,
                            am(lc) == am(lg) ? "same" : "DIFFERENT");
            }
            const bool pass = worst < 2e-2 && agree >= 4;
            std::printf("check-prefill: worst logits relative diff %.3g, top-1 agreement %d/5: %s\n", worst, agree,
                        pass ? "PASS" : "FAIL");
            return pass ? 0 : 1;
        }

        strata::glm::Trace tg, tc;
        double worst_logit = 0, worst_layer = 0;
        int top1_agree = 0;
        double gpu_s = 0;
        for (size_t i = 0; i < ids.size(); ++i) {
            const auto a = std::chrono::steady_clock::now();
            const std::vector<float>& lg = eng.eval(ids[i], check_cpu ? &tg : nullptr);
            gpu_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
            if (out) std::fwrite(lg.data(), sizeof(float), lg.size(), out);
            if (check_cpu) {
                const std::vector<float>& lc = ref->eval(ids[i], &tc);
                const double dl = rel_diff(lc, lg);
                worst_logit = std::max(worst_logit, dl);
                const auto am = [](const std::vector<float>& v) { return std::max_element(v.begin(), v.end()) - v.begin(); };
                top1_agree += am(lc) == am(lg);
                for (size_t l = 0; l < tc.layer_out.size(); ++l) {
                    const double d = rel_diff(tc.layer_out[l], tg.layer_out[l]);
                    worst_layer = std::max(worst_layer, d);
                    if (d > 1e-2) {
                        std::printf("position %zu: layer %zu residual differs by %.3g (relative)%s\n", i, l, d,
                                    model.config().is_kda((int64_t) l) ? " [KDA]" : " [MLA]");
                        break;
                    }
                }
                if (i < 4 || (i + 1) % 16 == 0)
                    std::printf("position %zu: logits relative diff %.3g, top-1 %s\n", i, dl,
                                am(lc) == am(lg) ? "same" : "DIFFERENT");
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
            const bool pass = worst_logit < 2e-2 && top1_agree >= (int) (0.98 * (double) ids.size());
            std::printf("check-cpu: worst logits relative diff %.3g, worst layer diff %.3g, top-1 agreement %d/%zu: %s\n",
                        worst_logit, worst_layer, top1_agree, ids.size(), pass ? "PASS" : "FAIL");
            return pass ? 0 : 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-glm: %s\n", e.what());
        return 1;
    }
    return 0;
}
