// src/glm/ref_main.cpp - strata-glm-ref: GLM-5.3-Flash logits from the CPU reference (cpu_ref.hpp).
//
//   strata-glm-ref -m MODEL.gguf --tokens-file IDS.txt --out LOGITS.bin [--threads N] [--dump-layers PATH]
//
// LOGITS.bin uses the layout of `strata generate --dump-logits` and `llama-strata-ref --out`, one row per
// position, so tools/glm53_compare.py compares it with llama.cpp's directly. --dump-layers writes, per position and
// layer, the residual streams after the layer's attention and after its FFN: n_layer x 2 x hc x n_embd float32, the
// layout `llama-strata-ref --dump-layers` writes (tools/glm53_layers_compare.py compares the two).
#include "strata/glm/cpu_ref.hpp"
#include "strata/glm/serve_loop.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    for (char& c : text)
        if (c == ',') c = ' ';
    std::istringstream is(text);
    long long v;
    while (is >> v) out.push_back((int32_t) v);
    return is.eof() && !out.empty();
}

struct RefEvaluator : strata::glm::Evaluator {
    strata::glm::CpuRef& r;
    explicit RefEvaluator(strata::glm::CpuRef& ref) : r(ref) {}
    void reset() override { r.reset(); }
    const std::vector<float>& eval(int32_t t) override { return r.eval(t); }
    int64_t pos() const override { return r.pos(); }
    uint64_t ticks() const override { return r.ticks(); }
    bool can_snapshot() const override { return true; }
    std::shared_ptr<strata::glm::Snapshot> snapshot(bool full) override { return r.snapshot(full); }
    size_t snapshot_bytes(bool full) const override { return r.snapshot_bytes(full); }
    void restore(const strata::glm::Snapshot& s) override { r.restore(s); }
};

std::vector<int32_t> parse_ids_list(const std::string& s) {
    std::vector<int32_t> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back((int32_t) std::atoi(item.c_str()));
    return out;
}

int usage() {
    std::fprintf(stderr,
                 "usage: strata-glm-ref -m MODEL.gguf --tokens-file IDS.txt --out LOGITS.bin\n"
                 "                      [--threads N] [--dump-layers PATH] [--max-tokens N]\n"
                 "       strata-glm-ref -m MODEL.gguf --serve [--ctx N] [--eos-ids A,B] [--threads N]\n"
                 "                      [--park-mib N] [--checkpoint-min N] [--no-warm]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, ids_path, out_path, layers_path;
    int threads = strata::glm::ThreadPool::physical_cores();
    long long max_tokens = 0, ctx = 4096;
    bool serve = false;
    std::vector<int32_t> eos;
    strata::glm::ServeOptions so;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
        const char* v = nullptr;
        if (a == "-m" && (v = next())) model_path = v;
        else if (a == "--tokens-file" && (v = next())) ids_path = v;
        else if (a == "--out" && (v = next())) out_path = v;
        else if (a == "--dump-layers" && (v = next())) layers_path = v;
        else if ((a == "--threads" || a == "-t") && (v = next())) threads = std::atoi(v);
        else if (a == "--max-tokens" && (v = next())) max_tokens = std::atoll(v);
        else if (a == "--serve") serve = true;
        else if (a == "--no-warm") {}                 // strata-glm's option; nothing to warm here
        else if (a == "--ctx" && (v = next())) ctx = std::atoll(v);
        else if (a == "--eos-ids" && (v = next())) eos = parse_ids_list(v);
        else if (a == "--park-mib" && (v = next())) so.park_bytes = (size_t) (std::max(0.0, std::atof(v)) * 1048576.0);
        else if (a == "--checkpoint-min" && (v = next())) so.checkpoint_min = std::atoll(v);
        else return usage();
    }
    if (serve && !model_path.empty()) {
        try {
            strata::glm::Model model(model_path);
            strata::glm::CpuRef ref(model, ctx, threads);
            RefEvaluator ev(ref);
            so.max_ctx = ctx;
            so.n_vocab = model.config().n_vocab;
            so.eos_ids = eos;
            so.info.push_back("backend=cpu-reference");
            return strata::glm::serve_loop(ev, so);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "strata-glm-ref: %s\n", e.what());
            return 1;
        }
    }
    if (model_path.empty() || ids_path.empty() || out_path.empty()) return usage();

    try {
        std::vector<int32_t> ids;
        if (!read_ids(ids_path, ids)) {
            std::fprintf(stderr, "cannot read token ids from %s\n", ids_path.c_str());
            return 1;
        }
        if (max_tokens > 0 && (long long) ids.size() > max_tokens) ids.resize((size_t) max_tokens);

        const auto t0 = std::chrono::steady_clock::now();
        strata::glm::Model model(model_path);
        const strata::glm::Config& c = model.config();
        std::printf("model: %s, %lld layers (%lld KDA), n_embd %lld, vocab %lld, %lld experts top-%lld\n",
                    c.arch.c_str(), (long long) c.n_layer,
                    (long long) [&] { long long n = 0; for (int64_t l = 0; l < c.n_layer; ++l) n += c.is_kda(l); return n; }(),
                    (long long) c.n_embd, (long long) c.n_vocab, (long long) c.n_expert, (long long) c.n_expert_used);
        strata::glm::CpuRef ref(model, (int64_t) ids.size(), threads);
        const auto t1 = std::chrono::steady_clock::now();

        FILE* out = std::fopen(out_path.c_str(), "wb");
        FILE* lay = layers_path.empty() ? nullptr : std::fopen(layers_path.c_str(), "wb");
        if (!out || (!layers_path.empty() && !lay)) {
            std::fprintf(stderr, "cannot write the output files\n");
            return 1;
        }
        const int32_t hdr[2] = {(int32_t) c.n_vocab, (int32_t) ids.size()};
        std::fwrite(hdr, sizeof hdr, 1, out);
        strata::glm::Trace trace;
        for (size_t i = 0; i < ids.size(); ++i) {
            const std::vector<float>& logits = ref.eval(ids[i], lay ? &trace : nullptr);
            std::fwrite(logits.data(), sizeof(float), logits.size(), out);
            if (lay)
                for (size_t l = 0; l < trace.layer_out.size(); ++l) {
                    std::fwrite(trace.attn_out[l].data(), sizeof(float), trace.attn_out[l].size(), lay);
                    std::fwrite(trace.layer_out[l].data(), sizeof(float), trace.layer_out[l].size(), lay);
                }
            if ((i + 1) % 16 == 0 || i + 1 == ids.size()) {
                std::printf("position %zu/%zu\n", i + 1, ids.size());
                std::fflush(stdout);
            }
        }
        std::fclose(out);
        if (lay) std::fclose(lay);
        const auto t2 = std::chrono::steady_clock::now();
        const double load_s = std::chrono::duration<double>(t1 - t0).count();
        const double run_s = std::chrono::duration<double>(t2 - t1).count();
        std::printf("wrote %zu rows of %lld logits to %s (load %.1f s, %.2f tokens/s on %d threads)\n", ids.size(),
                    (long long) c.n_vocab, out_path.c_str(), load_s, (double) ids.size() / run_s, threads);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-glm-ref: %s\n", e.what());
        return 1;
    }
    return 0;
}
