// src/glm/glm_main.cpp - strata-glm: GLM-5.3-Flash on the GPUs.
//
//   strata-glm -m MODEL.gguf --tokens-file IDS.txt [--out LOGITS.bin] [--ctx N] [--devices 0,1,2]
//              [--split 24,12,10] [--threads N] [--check-cpu] [--max-tokens N]
//
// --out writes the --dump-logits layout (tools/glm53_compare.py). --check-cpu also runs the CPU reference on the
// same tokens and compares, per position, the residual streams after every layer and the logits: the first layer
// where they part is where a kernel is wrong.
#include "strata/glm/cpu_ref.hpp"
#include "strata/glm/engine.hpp"

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

int usage() {
    std::fprintf(stderr,
                 "usage: strata-glm -m MODEL.gguf --tokens-file IDS.txt [--out LOGITS.bin] [--ctx N]\n"
                 "                  [--devices 0,1,2] [--split 24,12,10] [--threads N] [--check-cpu]\n"
                 "                  [--max-tokens N]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, ids_path, out_path;
    strata::glm::EngineOptions opt;
    bool check_cpu = false;
    long long max_tokens = 0;
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
        else return usage();
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
        std::printf("%zu tokens, %.2f tokens/s (one at a time)\n", ids.size(), (double) ids.size() / gpu_s);
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
