// tests/glm/serve_loop_test.cpp - the serve loop's line protocol (serve_loop.hpp) against scripted engines.
//
// Each case runs serve_loop in a forked child whose stdin and stdout are pipes, writes the request lines, closes
// stdin and reads everything the child printed. The fake engine predicts token (last + 1) % 32 and can be slow
// (with or without progress) or fail, so the cases cover INFO/READY, generation, end of turn, logit_bias,
// refusals, ERR internal, HB heartbeats (only while the engine progresses) and STOP of a queued request. A second
// fake keeps snapshots (Conv) for the conversations: parking, checkpoints and --park-mib.
#include "strata/glm/serve_loop.hpp"

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace strata::glm;

namespace {

constexpr int V = 32;

struct Fake : Evaluator {
    std::string mode;                 // plain | slow-progress | slow-hung | throw
    std::vector<float> logits = std::vector<float>(V, 0.0f);
    int64_t pos_ = 0;
    std::atomic<uint64_t> t{0};

    void reset() override { pos_ = 0; }
    const std::vector<float>& eval(int32_t tok) override {
        if (mode == "throw") throw std::runtime_error("boom");
        ++pos_;
        ++t;
        logits.assign(V, 0.0f);
        logits[(size_t) ((tok + 1) % V)] = 5.0f;
        return logits;
    }
    const std::vector<float>& prefill(const int32_t* ids, int64_t n,
                                      const std::function<bool(int64_t)>& progress) override {
        if (mode == "slow-progress" || mode == "slow-hung")
            for (int i = 0; i < 45; ++i) {               // 4.5 s, longer than the 3 s of quiet a heartbeat needs
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                if (mode == "slow-progress") ++t;
            }
        const std::vector<float>* last = nullptr;
        for (int64_t i = 0; i < n; ++i) last = &eval(ids[i]);
        if (progress) progress(n);
        return *last;
    }
    int64_t pos() const override { return pos_; }
    uint64_t ticks() const override { return t.load(); }
};

/// An engine whose state is its token history: the next token is a hash of all of it, so a state resumed wrongly
/// gives other tokens than a fresh one. A full snapshot holds the history (1000 bytes + 100 per token), a light one
/// 1000 bytes; restoring a light one fails unless the history below it is still its conversation's, as the real
/// engines need (snapshot.hpp).
struct Conv : Evaluator {
    struct Snap : Snapshot {
        std::vector<int32_t> hist;   // in a light one only to check the restore
        uint64_t h = 0;
    };
    std::vector<int32_t> hist;
    uint64_t h = 0;
    std::vector<float> logits = std::vector<float>(V, 0.0f);

    static uint64_t step(uint64_t h, int32_t tok) { return (h ^ (uint64_t) (tok + 1)) * 0x100000001b3ull + 7; }
    static int32_t next(uint64_t h) { return (int32_t) ((h >> 29) % (V - 1)); }   // never 31, the end of turn
    void reset() override {
        hist.clear();
        h = 0;
    }
    const std::vector<float>& eval(int32_t tok) override {
        hist.push_back(tok);
        h = step(h, tok);
        logits.assign(V, 0.0f);
        logits[(size_t) next(h)] = 5.0f;
        return logits;
    }
    int64_t pos() const override { return (int64_t) hist.size(); }
    bool can_snapshot() const override { return true; }
    size_t snapshot_bytes(bool full) const override { return 1000 + (full ? 100 * hist.size() : 0); }
    std::shared_ptr<Snapshot> snapshot(bool full) override {
        auto s = std::make_shared<Snap>();
        s->pos = pos();
        s->full = full;
        s->bytes = snapshot_bytes(full);
        s->hist = hist;
        s->h = h;
        return s;
    }
    void restore(const Snapshot& snap) override {
        const auto& s = dynamic_cast<const Snap&>(snap);
        if (!s.full && (hist.size() < s.hist.size() || !std::equal(s.hist.begin(), s.hist.end(), hist.begin())))
            throw std::runtime_error("a checkpoint restored over another conversation's state");
        hist = s.hist;
        h = s.h;
    }

    /// What a fresh engine answers to this prompt (greedy, max_new tokens, no end of turn).
    static std::vector<int32_t> answer(const std::vector<int32_t>& prompt, int max_new) {
        uint64_t x = 0;
        for (int32_t t : prompt) x = step(x, t);
        std::vector<int32_t> out;
        for (int i = 0; i < max_new; ++i) {
            out.push_back(next(x));
            x = step(x, out.back());
        }
        return out;
    }
};

/// The child's stdout lines for this input, with `body` running the serve loop (stderr to /dev/null if `quiet`).
std::vector<std::string> run_child(const std::string& input, const std::function<int()>& body, bool quiet = false) {
    int in[2], out[2];
    if (pipe(in) || pipe(out)) throw std::runtime_error("pipe");
    const pid_t pid = fork();
    if (pid == 0) {
        dup2(in[0], 0);
        dup2(out[1], 1);
        if (quiet) dup2(open("/dev/null", O_WRONLY), 2);
        close(in[0]); close(in[1]); close(out[0]); close(out[1]);
        _exit(body());
    }
    close(in[0]);
    close(out[1]);
    if (write(in[1], input.data(), input.size()) != (ssize_t) input.size()) throw std::runtime_error("write");
    close(in[1]);
    std::string text;
    char buf[4096];
    for (ssize_t n; (n = read(out[0], buf, sizeof buf)) > 0;) text.append(buf, (size_t) n);
    close(out[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    std::vector<std::string> lines;
    std::istringstream is(text);
    for (std::string l; std::getline(is, l);) lines.push_back(l);
    return lines;
}

/// The child's stdout lines for this mode of Fake and input.
std::vector<std::string> run(const std::string& mode, const std::string& input) {
    return run_child(input, [&] {
        Fake ev;
        ev.mode = mode;
        ServeOptions opt;
        opt.max_ctx = 64;
        opt.n_vocab = V;
        opt.eos_ids = {31};
        return serve_loop(ev, opt);
    });
}

/// The same with Conv, --park-mib in bytes and --checkpoint-min.
std::vector<std::string> run_conv(const std::string& input, size_t park_bytes, int64_t checkpoint_min) {
    return run_child(
        input,
        [&] {
            Conv ev;
            ServeOptions opt;
            opt.max_ctx = 4096;
            opt.n_vocab = V;
            opt.eos_ids = {31};
            opt.park_bytes = park_bytes;
            opt.checkpoint_min = checkpoint_min;
            return serve_loop(ev, opt);
        },
        true);
}

/// One request's answer: its RESUME, T and DONE or ERR lines.
struct Reply {
    long resume = 0, reused = -1;
    std::vector<int32_t> toks;
    std::string err;
};

std::vector<Reply> replies(const std::vector<std::string>& lines) {
    std::vector<Reply> out;
    Reply r;
    for (const auto& l : lines) {
        std::istringstream is(l);
        std::string w;
        is >> w;
        if (w == "RESUME") is >> r.resume;
        else if (w == "T") {
            int32_t t = 0;
            is >> t;
            r.toks.push_back(t);
        } else if (w == "DONE" || w == "ERR") {
            if (w == "ERR") r.err = l;
            else
                for (int i = 1; i <= 8; ++i)
                    if (is >> w && i == 8) r.reused = std::stol(w);   // DONE's <reused>
            out.push_back(r);
            r = Reply{};
        }
    }
    return out;
}

std::string gen_line(int max_new, const std::vector<int32_t>& ids, long cp = -1) {
    std::string s = "GEN " + std::to_string(max_new) + (cp >= 0 ? " cp=" + std::to_string(cp) : "") + " ";
    for (size_t i = 0; i < ids.size(); ++i) s += (i ? "," : "") + std::to_string(ids[i]);
    return s + "\n";
}

int failures = 0;

void expect(bool ok, const std::string& what, const std::vector<std::string>& lines) {
    std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) {
        ++failures;
        for (const auto& l : lines) std::printf("    | %s\n", l.c_str());
    }
}

bool has(const std::vector<std::string>& lines, const std::string& s) {
    for (const auto& l : lines)
        if (l == s) return true;
    return false;
}

bool starts(const std::vector<std::string>& lines, const std::string& s) {
    for (const auto& l : lines)
        if (l.rfind(s, 0) == 0) return true;
    return false;
}

int count(const std::vector<std::string>& lines, const std::string& s) {
    int n = 0;
    for (const auto& l : lines) n += l == s;
    return n;
}

}  // namespace

int main() {
    {
        const auto l = run("plain", "GEN 3 1,2\nQUIT\n");
        expect(l.size() >= 2 && l[0].rfind("INFO engine=", 0) == 0 && l[0].find(" logit_bias=1") != std::string::npos &&
                   l[0].find(" park_mib=0") != std::string::npos && l[1] == "READY 64 stop",
               "INFO (logit_bias=1, an integer park_mib) then READY", l);
        expect(has(l, "T 3") && has(l, "T 4") && has(l, "T 5") && starts(l, "DONE 3 2 ") &&
                   l.back().find(" length ") != std::string::npos,
               "greedy tokens, then DONE length", l);
    }
    {
        const auto l = run("plain", "GEN 10 29,30\n");
        expect(has(l, "T 31") && starts(l, "DONE 1 2 ") && l.back().find(" stop ") != std::string::npos,
               "the end-of-turn id is printed, then DONE stop", l);
    }
    {
        const auto l = run("plain", "GEN 1 logit_bias=3:-100,7:100 1,2\nGEN 1 logit_bias=3:500 1,2\nGEN 1 1,99\n"
                                    "GEN 1 logit_bias=40:1 1,2\nPING\n");
        expect(has(l, "T 7"), "logit_bias: the ban and the boost", l);
        expect(has(l, "ERR invalid logit_bias value (expected -100..100)") &&
                   has(l, "ERR a token id outside the vocabulary") &&
                   has(l, "ERR logit_bias ID outside the vocabulary") && has(l, "ERR unsupported command: PING"),
               "refusals: a bias out of range, a token and a bias id outside the vocabulary, an unknown command", l);
    }
    {
        const auto l = run("throw", "GEN 2 1,2\nGEN 2 3\n");
        expect(count(l, "ERR internal: boom") == 2 && !starts(l, "DONE"),
               "an engine failure is ERR internal, with no DONE, and the next request is still served", l);
    }
    {
        const auto l = run("slow-progress", "GEN 1 1,2\n");
        expect(has(l, "HB") && starts(l, "DONE 1 2 "), "HB while a slow prompt progresses", l);
    }
    {
        const auto l = run("slow-hung", "GEN 1 1,2\n");
        expect(!has(l, "HB") && starts(l, "DONE 1 2 "), "no HB while the engine makes no progress", l);
    }
    {
        // the second request is still queued behind the slow first one when the STOP comes
        const auto l = run("slow-progress", "GEN 2 1,2\nGEN 2 3,4\nSTOP\n");
        int dones = 0;
        bool first_ok = false, second_cancel = false;
        for (const auto& s : l)
            if (s.rfind("DONE", 0) == 0) {
                ++dones;
                if (dones == 1) first_ok = s.find(" length ") != std::string::npos;
                if (dones == 2) second_cancel = s == "DONE 0 2 0.0 0.0 cancel 0 0 0 0 0 0 0 0.0 0 0";
            }
        expect(dones == 2 && first_ok && second_cancel, "STOP ends the queued request it follows, not the running one",
               l);
    }
    {
        // 4294967297 and -4294967295 are 1 in 32 bits
        const auto l = run("plain", "GEN 1 4294967297\nGEN 1 2,-4294967295\n");
        expect(count(l, "ERR malformed GEN line") == 2 && !starts(l, "T "),
               "a token id beyond 32 bits is refused, not read as another id", l);
    }
    std::mt19937 rng(1);
    auto random_ids = [&](size_t n) {
        std::vector<int32_t> v(n);
        for (auto& t : v) t = (int32_t) (rng() % 31);
        return v;
    };
    auto cat = [](std::vector<int32_t> a, const std::vector<int32_t>& b) {
        a.insert(a.end(), b.begin(), b.end());
        return a;
    };
    {
        // A conversation is parked with its checkpoints and dropped with them. --park-mib here holds B parked (3128
        // bytes and a 1128-byte checkpoint) and A's park alone (5128) but not with its checkpoint (6256): parking A
        // when C comes would drop B first (the least recently used), then A itself. A is not kept; B stays.
        std::vector<int32_t> B = random_ids(20), A = random_ids(40), C = random_ids(10);
        B[0] = 1, A[0] = 2, C[0] = 3;
        const std::vector<int32_t> B2 = cat(cat(B, Conv::answer(B, 1)), {4, 5, 6});
        const auto l = run_conv(gen_line(1, B) + gen_line(1, A) + gen_line(1, C) + gen_line(1, B2), 6000, 1);
        const auto r = replies(l);
        expect(r.size() == 4 && r[3].err.empty() && r[3].resume == 20 && r[3].toks == Conv::answer(B2, 1),
               "a conversation that cannot stay parked with its checkpoints is not parked, and drops no other", l);
    }
    {
        // conversations taking turns at random (next turns, regenerated and edited turns, cp=, a shared system
        // prompt) under budgets from none to plenty: every answer is a fresh engine's, RESUME and DONE agree, and
        // no checkpoint is restored over another conversation's state
        int bad = 0, resumed = 0;
        std::vector<std::string> shown;
        for (int round = 0; round < 40; ++round) {
            const size_t budgets[] = {0, 2500, 6000, 12000, 30000, (size_t) 1 << 30};
            const size_t budget = budgets[rng() % 6];
            const int64_t cmin = rng() % 3 == 0 ? 1 : (int64_t) (rng() % 40);
            const std::vector<int32_t> system = random_ids(12);
            std::vector<std::vector<int32_t>> last(5), last_out(5), prompts;
            std::vector<int> max_news;
            std::string input;
            for (int k = 0; k < 30; ++k) {
                const size_t c = rng() % 5;
                const int act = last[c].empty() ? 0 : (int) (rng() % 6);
                std::vector<int32_t> p;
                if (act == 0) p = cat(rng() % 2 ? system : std::vector<int32_t>{}, random_ids(5 + rng() % 16));
                else if (act <= 3) p = cat(cat(last[c], last_out[c]), random_ids(1 + rng() % 8));   // the next turn
                else if (act == 4) p = last[c];                                                     // regenerated
                else {                                                                              // edited
                    p = last[c];
                    for (size_t i = p.size() - 1 - rng() % std::min<size_t>(5, p.size()); i < p.size(); ++i)
                        p[i] = (p[i] + 1) % 31;
                }
                const int max_new = 1 + (int) (rng() % 6);
                input += gen_line(max_new, p, rng() % 2 ? (long) p.size() - 2 : -1);
                prompts.push_back(p);
                max_news.push_back(max_new);
                last[c] = p;
                last_out[c] = Conv::answer(p, max_new);
            }
            const auto l = run_conv(input, budget, cmin);
            const auto r = replies(l);
            bool ok = r.size() == prompts.size();
            for (size_t i = 0; ok && i < r.size(); ++i) {
                ok = r[i].err.empty() && r[i].toks == Conv::answer(prompts[i], max_news[i]) &&
                     r[i].reused == r[i].resume && r[i].resume <= (long) prompts[i].size();
                resumed += r[i].resume > 0;
            }
            if (!ok && !bad++) shown = l;
        }
        expect(bad == 0 && resumed > 0, "conversations taking turns: every answer is a fresh engine's (" +
                                            std::to_string(resumed) + " resumed)", shown);
    }
    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures != 0;
}
