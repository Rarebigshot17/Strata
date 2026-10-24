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
// --decode-help N: with the experts page-locked, of each decoded token's experts a layer's cache misses, up to N are
// computed on the GPUs instead of the CPU (the first on the layer's card, the next ones on the others), each copied
// in from RAM while the CPU computes the rest.
// Each layer's ops up to the routing that do not depend on the position (a KDA layer's attention half and the
// router, an MLA layer's from its FFN half on; a dense layer's all) are captured as a CUDA graph at the first decoded
// token and launched as one from then on, instead of one call into the driver per op; --no-decode-graphs launches
// them op by op (the same values: on napster-ai, 2026-10-24, 300 tokens' logits bitwise the same, and 1.6% faster
// with the graphs). --decode-graphs is the default, and is still taken.
// Between prompts the prompt path's large buffers (the chunk's streams, the arena, the copy-ahead pools) hold more
// slots of the VRAM expert cache. A prompt of --lend-min-prompt tokens or more (64) takes them back, and their experts
// leave the cache; the next decoded token gives them back empty, and the misses fill them again from RAM. A shorter
// prompt runs token by token and leaves the cache as it is. On napster-ai (2026-10-24, chunk 4096) the cache grew
// from 23.2 to 31.2 GB and decoding went from 15.1 / 15.2 / 14.4 to 17.7 / 17.6 / 16.0 tokens/s on three prompts,
// the 1,971-token one taking the buffers back. --no-lend-prompt-buffers keeps them for the prompt path alone;
// --lend-prompt-buffers is the default, and is still taken.
// --cache-max-slots N: at most N VRAM cache slots of its own per layer (a test setting: the GPU check's small models
// would otherwise fit every expert in them, leaving nothing to lend).
// --mtp: decode passes of a token and its draft (GpuEngine::eval_pair, drop_last), and the model's MTP block when it
// has one. --check-pairs (with --tokens-file) runs the tokens in pairs, some taken back, against the same engine one
// token at a time, and passes when every row of logits is bitwise the same. With an MTP block, --check-cpu also
// compares the block's logits at every position (its draft from the token there and the hidden state before) with
// the CPU reference's (check-mtp; after the prompt in chunks with --prefill, at the decoded tokens), and --mtp-out
// PATH writes them, the layout of --out and `llama-strata-ref --mtp-out`. --check-generate N (with --mtp and an MTP
// block) reads each line of the tokens file as a prompt, then decodes N tokens greedily one at a time and again with
// the block's drafts (mtp_draft, eval_pair, drop_last when the draft is not the next token), the expert cache warmed
// by a first run and then kept as it is for both; it passes when both give the same tokens, and reports the drafts
// taken and the time. --out then writes the replies' token ids, a line each. --bench-generate N times the same with
// the expert cache live, as a server runs it (from --expert-profile), every prompt both ways each of --bench-rounds
// rounds (2), the order alternating by round.
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
#include <cstring>
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
    bool can_draft() const override { return e.mtp_ready(); }
    int32_t draft(int32_t t) override { return e.mtp_draft(t); }
    double draft_prob() const override { return e.mtp_draft_prob(); }
    const float* eval_pair(int32_t a, int32_t b) override { return e.eval_pair(a, b); }
    void drop_last() override { e.drop_last(); }
};

/// --check-cpu with the MTP block: its logits at each position against the CPU reference's, by check.hpp's rule. A
/// position where the GPU's indexer took other cells than the CPU's, on pool scores the CPU had within kSelTie of each
/// other, is a tie: the GPU's 8-bit activations round those scores apart (napster-ai, the Q4_0 test model,
/// 2026-10-24: up to 4.0% at the 4 positions printed, and a 0.9% gap that came out the other way), and attending
/// other cells moves the logits a lot. Ties are counted and left out of the rule; more than a quarter of the rows
/// being ties fails the check.
struct MtpCheck {
    static constexpr double kSelTie = 0.10;
    strata::glm::LogitCheck chk;
    int rows = 0, ties = 0;
    double add(const std::vector<float>& cpu, const std::vector<float>& gpu, std::vector<int32_t> cpu_sel,
               double cpu_margin, std::vector<int32_t> gpu_sel) {
        ++rows;
        std::sort(cpu_sel.begin(), cpu_sel.end());
        std::sort(gpu_sel.begin(), gpu_sel.end());
        if (cpu_sel != gpu_sel && cpu_margin < kSelTie) {
            ++ties;
            return -1;
        }
        return chk.add(cpu, gpu);
    }
    bool pass(const strata::glm::CheckLimits& l) const { return chk.pass(l) && 4 * ties <= rows; }
    std::string summary(const strata::glm::CheckLimits& l) const {
        return chk.summary(l) + "; " + std::to_string(ties) + " of " + std::to_string(rows) +
               " rows a selection tie (other cells than the CPU's, on pool scores within 10%), not counted";
    }
};

int usage() {
    std::fprintf(stderr,
                 "usage: strata-glm -m MODEL.gguf --tokens-file IDS.txt [--out LOGITS.bin] [--ctx N]\n"
                 "                  [--devices 0,1,2] [--split 24,12,10] [--threads N] [--check-cpu]\n"
                 "                  [--max-tokens N] [--vram-reserve GB] [--expert-profile PATH]\n"
                 "                  [--max-swaps N] [--cache-decay D] [--no-expert-cache] [--prefill] [--prefill-chunk N]\n"
                 "                  [--pin-experts [--pin-keep-free GB]] [--prefill-rows N] [--kv-in-ram]\n"
                 "                  [--prefill-spread] [--decode-help N] [--no-decode-graphs]\n"
                 "                  [--no-lend-prompt-buffers] [--lend-min-prompt N] [--cache-max-slots N]\n"
                 "                  [--mtp [--mtp-min-prob P] [--check-pairs] [--mtp-out MTP.bin]\n"
                 "                  [--check-generate N] [--bench-generate N [--bench-rounds R]]]\n"
                 "       strata-glm -m MODEL.gguf --serve [--ctx N] [--eos-ids A,B] [--park-mib N] [--no-warm]\n"
                 "                  [--checkpoint-min N] [same GPU and cache options]\n");
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_path, ids_path, out_path, mtp_path;
    strata::glm::EngineOptions opt;
    bool check_cpu = false, serve = false, use_prefill = false, warm = true, pin = false, check_pairs = false;
    int check_generate = 0, bench_generate = 0, bench_rounds = 2;
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
        else if (a == "--cache-max-slots" && (v = next())) opt.cache.max_slots = std::max(0, std::atoi(v));
        else if (a == "--serve") serve = true;
        else if (a == "--no-warm") warm = false;
        else if (a == "--pin-experts") pin = true;
        else if (a == "--pin-keep-free" && (v = next())) pin_keep_gb = std::atof(v);
        else if (a == "--prefill") use_prefill = true;
        else if (a == "--prefill-chunk" && (v = next())) opt.prefill_chunk = std::atoi(v);
        else if (a == "--prefill-rows" && (v = next())) opt.prefill_rows = std::atoi(v);
        else if (a == "--kv-in-ram") opt.kv_in_ram = true;
        else if (a == "--prefill-spread") opt.prefill_spread = true;
        else if (a == "--decode-help" && (v = next())) opt.decode_help = std::max(0, std::atoi(v));
        else if (a == "--decode-graphs") opt.decode_graphs = true;
        else if (a == "--mtp") opt.mtp = true;
        else if (a == "--check-pairs") check_pairs = true;
        else if (a == "--mtp-out" && (v = next())) mtp_path = v;
        else if (a == "--check-generate" && (v = next())) check_generate = std::max(1, std::atoi(v));
        else if (a == "--bench-generate" && (v = next())) bench_generate = std::max(1, std::atoi(v));
        else if (a == "--bench-rounds" && (v = next())) bench_rounds = std::max(1, std::atoi(v));
        else if (a == "--mtp-min-prob" && (v = next())) so.draft_min_prob = std::atof(v);
        else if (a == "--no-decode-graphs") opt.decode_graphs = false;
        else if (a == "--lend-prompt-buffers") opt.lend_prompt_buffers = true;
        else if (a == "--no-lend-prompt-buffers") opt.lend_prompt_buffers = false;
        else if (a == "--lend-min-prompt" && (v = next())) opt.lend_min_prompt = std::max(0, std::atoi(v));
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
            if (warm && eng.mtp_ready()) {
                // the MTP block's experts stay in the mapped file (--pin-experts locks the trunk's), and every draft
                // reads 8 of them: into the page cache too
                const auto w0 = std::chrono::steady_clock::now();
                model.warm_experts(copy_threads, [&](size_t done, size_t total) {
                    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
                    std::fprintf(stderr, "warming the MTP block's experts: %.1f of %.1f GB (%.0f s)\n", done / 1e9,
                                 total / 1e9, s);
                }, true);
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
        const int64_t n_gen = std::max(check_generate, bench_generate);
        const int64_t n_ctx = (int64_t) ids.size() + check_decode + (n_gen > 0 ? n_gen + 2 : 0);
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

        if (check_pairs) {
            // pairs (eval_pair: a token and the next in one pass; every third one's second token a wrong draft, taken
            // back with drop_last) against the same engine one token at a time, bitwise. A pass over the tokens first
            // warms the expert cache, which then keeps what it holds, so both runs place every expert alike.
            if (!opt.mtp) {
                std::fprintf(stderr, "--check-pairs needs --mtp\n");
                return 2;
            }
            for (int32_t id : ids) eng.eval(id);
            eng.freeze_cache();
            eng.reset();
            std::vector<std::vector<float>> want;
            const auto w0 = std::chrono::steady_clock::now();
            for (int32_t id : ids) want.push_back(eng.eval(id));
            const double s_one = std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
            eng.reset();
            size_t passes = 0;
            const auto p0 = std::chrono::steady_clock::now();
            const int64_t V = c.n_vocab;
            size_t rows = 0, same = 0, taken_back = 0;
            double worst = 0;
            const auto cmp = [&](const float* got, const std::vector<float>& w) {
                ++rows;
                if (std::memcmp(got, w.data(), w.size() * sizeof(float)) == 0) {
                    ++same;
                    return;
                }
                for (size_t v = 0; v < w.size(); ++v) worst = std::max(worst, (double) std::fabs(got[v] - w[v]));
            };
            for (size_t i = 0, k = 0; i < ids.size();) {
                if (i + 1 == ids.size()) {
                    cmp(eng.eval(ids[i]).data(), want[i]);
                    ++i;
                } else if (k++ % 3 == 2) {
                    cmp(eng.eval_pair(ids[i], (int32_t) ((ids[i + 1] + 1) % V)), want[i]);
                    eng.drop_last();
                    ++taken_back;
                    ++passes;
                    ++i;
                } else {
                    const float* lg = eng.eval_pair(ids[i], ids[i + 1]);
                    cmp(lg, want[i]);
                    cmp(lg + V, want[i + 1]);
                    ++passes;
                    i += 2;
                }
            }
            const double s_pairs = std::chrono::duration<double>(std::chrono::steady_clock::now() - p0).count();
            std::printf("one token at a time: %.2f ms a token; pairs: %.2f ms a pass (%zu passes, the cache frozen in "
                        "both)\n", 1e3 * s_one / (double) ids.size(),
                        1e3 * s_pairs / (double) std::max<size_t>(passes, 1), passes);
            const bool pass = same == rows;
            std::printf("check-pairs: %zu rows (%zu pairs taken back), %zu bitwise the same as one token at a time, "
                        "worst |diff| %.3g: %s\n", rows, taken_back, same, worst, pass ? "PASS" : "FAIL");
            return pass ? 0 : 1;
        }

        if (check_generate > 0 || bench_generate > 0) {
            // each prompt (in chunks), then N tokens chosen greedily: one at a time, and with the MTP block's drafts,
            // each checked by a pass over the token and its draft (the draft is taken when it is the token the first
            // row of logits chooses, else drop_last). Bitwise the same logits, so the same tokens: a first run warms
            // the expert cache, which then keeps what it holds, so both place every expert alike.
            if (!eng.mtp_ready()) {
                std::fprintf(stderr, "--check-generate needs --mtp and a model with an MTP block\n");
                return 2;
            }
            std::vector<std::vector<int32_t>> prompts;
            {
                std::ifstream in(ids_path);
                for (std::string line; std::getline(in, line);) {
                    for (char& ch : line)
                        if (ch == ',') ch = ' ';
                    std::istringstream is(line);
                    std::vector<int32_t> p;
                    for (long long v; is >> v;) p.push_back((int32_t) v);
                    if (!p.empty()) prompts.push_back(std::move(p));
                }
            }
            using clk = std::chrono::steady_clock;
            const auto since = [](clk::time_point a) { return std::chrono::duration<double>(clk::now() - a).count(); };
            const int N = check_generate > 0 ? check_generate : bench_generate;
            const bool live = check_generate == 0;   // --bench-generate: the expert cache as a server runs it
            const int64_t V = c.n_vocab;
            const auto am = [&](const float* v) { return (int32_t) (std::max_element(v, v + V) - v); };
            const auto plain = [&](const std::vector<int32_t>& pr, std::vector<int32_t>& toks) {
                eng.reset();
                int32_t x = am(eng.prefill(pr.data(), (int64_t) pr.size()).data());
                toks.assign(1, x);
                const auto a = clk::now();
                while ((int) toks.size() < N) toks.push_back(x = am(eng.eval(x).data()));
                return since(a);
            };
            int64_t drafts = 0, taken = 0, skipped = 0;
            double s_draft = 0, s_pass = 0, s_one = 0, s_two = 0;
            // the drafts by the MTP block's probability of its own draft (softmax of its logits): how many, how many
            // taken, in bands below 0.5, 0.8, 0.95 and above
            const double band_top[4] = {0.5, 0.8, 0.95, 1.01};
            int64_t band_n[4] = {0, 0, 0, 0}, band_taken[4] = {0, 0, 0, 0};
            const auto draft_prob = [&] {
                const std::vector<float>& ml = eng.mtp_logits();
                const float mx = *std::max_element(ml.begin(), ml.end());
                double sum = 0;
                for (float v : ml) sum += std::exp((double) v - mx);
                return 1.0 / sum;
            };
            const auto drafted = [&](const std::vector<int32_t>& pr, std::vector<int32_t>& toks) {
                eng.reset();
                int32_t x = am(eng.prefill(pr.data(), (int64_t) pr.size()).data());
                toks.assign(1, x);
                const auto a = clk::now();
                while ((int) toks.size() < N) {
                    const auto b = clk::now();
                    const int32_t d = eng.mtp_draft(x);
                    s_draft += since(b);
                    const double pd = draft_prob();
                    if (pd < so.draft_min_prob) {    // --mtp-min-prob: an unlikely draft, x alone
                        ++skipped;
                        toks.push_back(x = am(eng.eval(x).data()));
                        continue;
                    }
                    const int band = pd < band_top[0] ? 0 : pd < band_top[1] ? 1 : pd < band_top[2] ? 2 : 3;
                    ++band_n[band];
                    const auto e = clk::now();
                    const float* lg = eng.eval_pair(x, d);
                    s_pass += since(e);
                    ++drafts;
                    const int32_t y = am(lg);
                    if (y == d) {                    // the draft is the next token: the second row gives the one after
                        ++taken;
                        ++band_taken[band];
                        toks.push_back(d);
                        x = am(lg + V);
                        if ((int) toks.size() < N) toks.push_back(x);
                    } else {
                        eng.drop_last();
                        toks.push_back(x = y);
                    }
                }
                return since(a);
            };
            // the MTP block's experts into the page cache, as --serve does before READY: the drafts read them from the
            // mapped file, and the first ones would otherwise wait for the disk inside the timing
            model.warm_experts(opt.cpu_threads > 0 ? std::min(opt.cpu_threads, 16) : 8, nullptr, true);
            bool same = true;
            std::vector<std::vector<int32_t>> replies;
            size_t prompt_tokens = 0;
            if (live) {
                // --bench-generate: every prompt one token at a time and with drafts, each round, with the expert cache
                // learning and swapping as in a server (it starts from --expert-profile); the order alternates by round
                // so neither mode always meets the cache the other left. The replies can part where the cache's
                // placement flips a near tie, so the time is per token of each reply.
                int same_n = 0, runs = 0;
                for (int r = 0; r < bench_rounds; ++r)
                    for (size_t k = 0; k < prompts.size(); ++k) {
                        const std::vector<int32_t>& pr = prompts[k];
                        std::vector<int32_t> one, two;
                        const int64_t d0 = drafts, k0 = taken;
                        double o = 0, w = 0;
                        if (r % 2 == 0) {
                            o = plain(pr, one);
                            w = drafted(pr, two);
                        } else {
                            w = drafted(pr, two);
                            o = plain(pr, one);
                        }
                        s_one += o;
                        s_two += w;
                        ++runs;
                        same_n += one == two;
                        const double per = 1e3 / (double) std::max(N - 1, 1);
                        std::printf("round %d, prompt %zu: one token at a time %.2f ms a token, with drafts %.2f ms a "
                                    "token; %lld of %lld drafts taken; %s\n", r + 1, k + 1, o * per, w * per,
                                    (long long) (taken - k0), (long long) (drafts - d0),
                                    one == two ? "the same tokens" : "other tokens");
                    }
                const double nt = (double) runs * std::max(N - 1, 1);
                std::printf("bench-generate: %d rounds of %zu prompts, %d tokens each: one token at a time %.2f ms a "
                            "token, with drafts %.2f ms a token (%.1f%% faster; %lld drafts, %.1f%% taken, a draft "
                            "%.2f ms, a pass %.2f ms); %d of %d replies the same\n", bench_rounds, prompts.size(), N,
                            1e3 * s_one / nt, 1e3 * s_two / nt, 100.0 * (s_one / s_two - 1.0), (long long) drafts,
                            100.0 * (double) taken / (double) std::max<int64_t>(drafts, 1),
                            1e3 * s_draft / (double) std::max<int64_t>(drafts, 1),
                            1e3 * s_pass / (double) std::max<int64_t>(drafts, 1), same_n, runs);
                eng.save_profile();
                return 0;
            }
            for (size_t k = 0; k < prompts.size(); ++k) {
                const std::vector<int32_t>& pr = prompts[k];
                prompt_tokens += pr.size();
                std::vector<int32_t> warm_toks, one, two;
                eng.freeze_cache(false);
                plain(pr, warm_toks);
                eng.freeze_cache();
                const int64_t d0 = drafts, k0 = taken;
                const double o = plain(pr, one), w = drafted(pr, two);
                s_one += o;
                s_two += w;
                size_t first_diff = 0;
                const size_t nb = std::min(one.size(), two.size());
                while (first_diff < nb && one[first_diff] == two[first_diff]) ++first_diff;
                const double per = 1e3 / (double) std::max(N - 1, 1);
                std::printf("prompt %zu (%zu tokens): one token at a time %.2f ms a token, with drafts %.2f ms a "
                            "token; %lld of %lld drafts taken; %s\n", k + 1, pr.size(), o * per, w * per,
                            (long long) (taken - k0), (long long) (drafts - d0),
                            one == two ? "the same tokens" : "NOT the same tokens");
                if (one != two) {
                    same = false;
                    std::printf("token %zu differs: %d one at a time, %d with drafts\n", first_diff,
                                first_diff < one.size() ? one[first_diff] : -1,
                                first_diff < two.size() ? two[first_diff] : -1);
                }
                replies.push_back(std::move(two));
            }
            const double nt = (double) prompts.size() * std::max(N - 1, 1);
            std::printf("one token at a time: %.2f ms a token; with drafts: %.2f ms a token (a draft %.2f ms, a pass "
                        "over a token and its draft %.2f ms; %lld drafts, %lld taken, %.1f%%)\n",
                        1e3 * s_one / nt, 1e3 * s_two / nt, 1e3 * s_draft / (double) std::max<int64_t>(drafts, 1),
                        1e3 * s_pass / (double) std::max<int64_t>(drafts, 1), (long long) drafts, (long long) taken,
                        100.0 * (double) taken / (double) std::max<int64_t>(drafts, 1));
            if (so.draft_min_prob > 0)
                std::printf("--mtp-min-prob %g: %lld drafts below it, their tokens run alone\n", so.draft_min_prob,
                            (long long) skipped);
            std::printf("drafts by the MTP block's probability of its draft: below 0.5 %lld (%lld taken), 0.5-0.8 %lld "
                        "(%lld), 0.8-0.95 %lld (%lld), 0.95 and above %lld (%lld)\n", (long long) band_n[0],
                        (long long) band_taken[0], (long long) band_n[1], (long long) band_taken[1],
                        (long long) band_n[2], (long long) band_taken[2], (long long) band_n[3],
                        (long long) band_taken[3]);
            std::printf("check-generate: %zu prompt%s (%zu tokens), %d tokens each, %s with drafts as one at a time: "
                        "%s\n", prompts.size(), prompts.size() == 1 ? "" : "s", prompt_tokens, N,
                        same ? "the same" : "NOT the same", same ? "PASS" : "FAIL");
            if (out) {
                // the drafted replies' token ids, a line each
                std::fclose(out);
                if (FILE* f = std::fopen(out_path.c_str(), "w")) {
                    for (const auto& r : replies)
                        for (size_t k = 0; k < r.size(); ++k)
                            std::fprintf(f, "%d%s", r[k], k + 1 < r.size() ? "," : "\n");
                    std::fclose(f);
                }
            }
            return same && !prompts.empty() ? 0 : 1;
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
            MtpCheck mchk;
            const bool mtp = eng.mtp_ready();
            eng.mtp_keep_selection(mtp);
            std::vector<float> lc, h_prev((size_t) c.n_embd, 0.0f);
            for (int32_t id : ids) {
                if (mtp) ref->mtp_eval(id, h_prev.data());          // the CPU's MTP rows, the whole block each
                lc = ref->eval(id);
                h_prev = ref->h();
            }
            double dl = chk.add(lc, lg);
            std::printf("after the prompt: logits relative diff %.3g, KL %.3g, top-1 %s\n", dl,
                        strata::glm::kl_div(lc, lg), am(lc) == am(lg) ? "same" : "DIFFERENT");
            for (int step = 0; step < check_decode; ++step) {
                const int32_t next = am(lc);
                if (mtp) {
                    // the MTP block reads the rows the prompt's chunks wrote for it
                    eng.mtp_draft(next);
                    const std::vector<float>& mg = eng.mtp_logits();
                    const std::vector<float>& mc = ref->mtp_eval(next, h_prev.data());
                    const double dm = mchk.add(mc, mg, ref->mtp_selection(), ref->mtp_margin(), eng.mtp_selection());
                    std::printf("decode %d (token %d): MTP logits relative diff %.3g, KL %.3g, top-1 %s%s\n", step,
                                next, dm < 0 ? strata::glm::rel_diff(mc, mg) : dm, strata::glm::kl_div(mc, mg),
                                am(mc) == am(mg) ? "same" : "DIFFERENT", dm < 0 ? " (a selection tie)" : "");
                }
                lc = ref->eval(next);
                h_prev = ref->h();
                lg = eng.eval(next);
                dl = chk.add(lc, lg);
                std::printf("decode %d: logits relative diff %.3g, KL %.3g, top-1 %s\n", step, dl,
                            strata::glm::kl_div(lc, lg), am(lc) == am(lg) ? "same" : "DIFFERENT");
            }
            const bool pass = chk.pass(lim), mpass = !mtp || mchk.pass(lim);
            std::printf("check-prefill: %s: %s\n", chk.summary(lim).c_str(), pass ? "PASS" : "FAIL");
            if (mtp) std::printf("check-mtp: %s: %s\n", mchk.summary(lim).c_str(), mpass ? "PASS" : "FAIL");
            return pass && mpass ? 0 : 1;
        }

        strata::glm::Trace tg, tc;
        const strata::glm::CheckLimits lim = strata::glm::check_limits(quantized_weights(model));
        strata::glm::LogitCheck chk;
        MtpCheck mchk;
        strata::glm::LayerCheck layers = strata::glm::layer_check(lim);
        double gpu_s = 0;
        // with an MTP block: its logits at every position (the draft from the token there and the hidden state the
        // trunk left before it), against the CPU reference's and into --mtp-out
        const bool mtp = eng.mtp_ready();
        if (!mtp_path.empty() && !mtp) {
            std::fprintf(stderr, "--mtp-out needs --mtp and a model with an MTP block\n");
            return 2;
        }
        FILE* mout = mtp_path.empty() ? nullptr : std::fopen(mtp_path.c_str(), "wb");
        if (!mtp_path.empty() && !mout) {
            std::fprintf(stderr, "cannot write %s\n", mtp_path.c_str());
            return 1;
        }
        if (mout) {
            const int32_t hdr[2] = {(int32_t) c.n_vocab, (int32_t) ids.size()};
            std::fwrite(hdr, sizeof hdr, 1, mout);
        }
        std::vector<float> h_prev((size_t) c.n_embd, 0.0f);
        eng.mtp_keep_selection(mtp && check_cpu);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (mtp) {
                eng.mtp_draft(ids[i]);
                const std::vector<float>& mg = eng.mtp_logits();
                if (mout) std::fwrite(mg.data(), sizeof(float), mg.size(), mout);
                if (check_cpu) {
                    const std::vector<float>& mc = ref->mtp_eval(ids[i], h_prev.data());
                    if (mchk.add(mc, mg, ref->mtp_selection(), ref->mtp_margin(), eng.mtp_selection()) < 0)
                        std::printf("position %zu: the MTP block's selection is a tie (logits relative diff %.3g)\n",
                                    i, strata::glm::rel_diff(mc, mg));
                }
            }
            const auto a = std::chrono::steady_clock::now();
            const std::vector<float>& lg = eng.eval(ids[i], check_cpu ? &tg : nullptr);
            gpu_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
            if (out) std::fwrite(lg.data(), sizeof(float), lg.size(), out);
            if (check_cpu) {
                const std::vector<float>& lc = ref->eval(ids[i], &tc);
                h_prev = ref->h();
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
        if (mout) std::fclose(mout);
        std::printf("%zu tokens, %.2f tokens/s (one at a time), expert cache hit rate %.1f%%\n", ids.size(),
                    (double) ids.size() / gpu_s, 100.0 * eng.cache_hit_rate());
        eng.save_profile();
        if (check_cpu) {
            const bool pass = chk.pass(lim), mpass = !mtp || mchk.pass(lim);
            std::printf("check-cpu: %s; worst layer diff %.3g: %s\n", chk.summary(lim).c_str(), layers.worst,
                        pass ? "PASS" : "FAIL");
            if (mtp) std::printf("check-mtp: %s: %s\n", mchk.summary(lim).c_str(), mpass ? "PASS" : "FAIL");
            return pass && mpass ? 0 : 1;
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-glm: %s\n", e.what());
        return 1;
    }
    return 0;
}
