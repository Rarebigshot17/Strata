// tests/glm/serve_loop_test.cpp - the serve loop's line protocol (serve_loop.hpp) against scripted engines.
//
// Each case runs serve_loop in a forked child whose stdin and stdout are pipes, writes the request lines, closes
// stdin and reads everything the child printed. The fake engine predicts token (last + 1) % 32 and can be slow
// (with or without progress) or fail, so the cases cover INFO/READY, generation, end of turn, end=, logit_bias,
// refusals, ERR internal, HB heartbeats (only while the engine progresses) and STOP of a queued request. A second
// fake keeps snapshots (Conv) for the conversations: parking, checkpoints, --park-mib and the state after end=.
// The last cases are strata-glm's signals in --serve mode (serve_signals.hpp) and how an engine ends when its server
// is gone: idle, during a request, and while it starts.
#include "strata/glm/serve_loop.hpp"
#include "strata/glm/serve_signals.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace strata::glm;

namespace {

constexpr int V = 32;

struct Fake : Evaluator {
    std::string mode;                 // plain | slow-progress | slow-hung | slow-decode | throw
    std::vector<float> logits = std::vector<float>(V, 0.0f);
    int64_t pos_ = 0;
    std::atomic<uint64_t> t{0};

    void reset() override { pos_ = 0; }
    const std::vector<float>& eval(int32_t tok) override {
        if (mode == "throw") throw std::runtime_error("boom");
        if (mode == "slow-decode") std::this_thread::sleep_for(std::chrono::milliseconds(5));
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
    std::fflush(stdout);                               // else the child prints this process's buffered lines again
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

std::string gen_line(int max_new, const std::vector<int32_t>& ids, long cp = -1, const std::string& keys = "") {
    std::string s = "GEN " + std::to_string(max_new) + (cp >= 0 ? " cp=" + std::to_string(cp) : "") + keys + " ";
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

using Clock = std::chrono::steady_clock;

/// An engine in a forked child for the signal cases: this process holds the other ends of its stdin and stdout.
struct Proc {
    pid_t pid = -1;
    int in = -1, out = -1;
    std::string got;          ///< what it printed so far
};

Proc spawn(const std::function<int()>& body) {
    int in[2], out[2];
    if (pipe(in) || pipe(out)) throw std::runtime_error("pipe");
    std::fflush(stdout);
    Proc p;
    p.pid = fork();
    if (p.pid == 0) {
        dup2(in[0], 0);
        dup2(out[1], 1);
        dup2(open("/dev/null", O_WRONLY), 2);          // the notes of the ignored signals
        close(in[0]); close(in[1]); close(out[0]); close(out[1]);
        _exit(body());
    }
    close(in[0]);
    close(out[1]);
    p.in = in[1];
    p.out = out[0];
    return p;
}

/// Reads the child's stdout until a line starts with `prefix` (true), its output ends or `secs` pass (false).
bool read_until(Proc& p, const std::string& prefix, double secs) {
    const auto end = Clock::now() + std::chrono::duration<double>(secs);
    for (;;) {
        for (size_t at = 0; at < p.got.size();) {
            const size_t nl = p.got.find('\n', at);
            if (nl == std::string::npos) break;
            if (p.got.compare(at, prefix.size(), prefix) == 0) return true;
            at = nl + 1;
        }
        const int ms = (int) std::chrono::duration_cast<std::chrono::milliseconds>(end - Clock::now()).count();
        if (ms <= 0) return false;
        pollfd fd{p.out, POLLIN, 0};
        if (poll(&fd, 1, ms) <= 0) continue;
        char buf[4096];
        const ssize_t n = read(p.out, buf, sizeof buf);
        if (n <= 0) return false;
        p.got.append(buf, (size_t) n);
    }
}

/// waitpid within `secs`: the status, or -1 while it still runs.
int wait_for(pid_t pid, double secs) {
    const auto end = Clock::now() + std::chrono::duration<double>(secs);
    for (;;) {
        int st = 0;
        if (waitpid(pid, &st, WNOHANG) == pid) return st;
        if (Clock::now() >= end) return -1;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

/// Ends a child a case left running.
void reap(pid_t pid) {
    kill(pid, SIGKILL);
    waitpid(pid, nullptr, 0);
}

/// Whether a process that is not this one's child (yet) still runs: a zombie has ended.
bool running(pid_t pid) {
    char path[64];
    std::snprintf(path, sizeof path, "/proc/%d/stat", (int) pid);
    FILE* f = std::fopen(path, "r");
    if (!f) return false;
    char buf[512] = {0};
    const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
    std::fclose(f);
    const char* close_paren = std::strrchr(buf, ')');
    return n > 0 && close_paren && close_paren[1] == ' ' && close_paren[2] != 'Z';
}

/// A "server" child that starts an engine child running `engine` once it has called serve_signals_starting()
/// (and, with `serving`, serve_signals_serving()), then waits to be killed. Returns {server, engine}.
std::pair<pid_t, pid_t> server_with_engine(bool serving) {
    int e2s[2], s2t[2];
    if (pipe(e2s) || pipe(s2t)) throw std::runtime_error("pipe");
    std::fflush(stdout);
    const pid_t server = fork();
    if (server == 0) {
        const pid_t engine = fork();
        if (engine == 0) {
            serve_signals_starting();
            if (serving) serve_signals_serving();
            dup2(open("/dev/null", O_WRONLY), 2);
            const char c = 's';
            if (write(e2s[1], &c, 1) != 1) _exit(3);
            for (;;) pause();                          // starting (the warm-up), or serving with nothing to do
        }
        char c = 0;
        if (read(e2s[0], &c, 1) != 1 || write(s2t[1], &engine, sizeof engine) != (ssize_t) sizeof engine) _exit(3);
        for (;;) pause();
    }
    close(e2s[0]); close(e2s[1]); close(s2t[1]);
    pid_t engine = -1;
    if (read(s2t[0], &engine, sizeof engine) != (ssize_t) sizeof engine) engine = -1;
    close(s2t[0]);
    return {server, engine};
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
        const auto l = run("plain", "GEN 1 penalty_repeat=0 1,2\nGEN 1 temperature=inf 1,2\n");
        expect(has(l, "ERR penalty_repeat must be above 0") && has(l, "ERR temperature must be a finite number") &&
                   !starts(l, "T "),
               "sampling values that leave nothing to draw from are refused", l);
    }
    {
        // 4294967297 and -4294967295 are 1 in 32 bits
        const auto l = run("plain", "GEN 1 4294967297\nGEN 1 2,-4294967295\n");
        expect(count(l, "ERR malformed GEN line") == 2 && !starts(l, "T "),
               "a token id beyond 32 bits is refused, not read as another id", l);
    }
    {
        const auto l = run("plain", "GEN 10 end=9,5 1,2\nGEN 3 end= 1,2\nGEN 3 end=4x 1,2\n");
        const auto r = replies(l);
        expect(r.size() == 3 && r[0].toks == std::vector<int32_t>{3, 4, 5} && starts(l, "DONE 3 2 ") &&
                   l[l.size() - 3].find(" stop ") != std::string::npos && r[1].err == "ERR malformed GEN line" &&
                   r[2].err == "ERR malformed GEN line",
               "end=: the request ends after one of its ids, as at an end of turn; a malformed list is refused", l);
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
        // The same when switching back to a parked conversation, whose checkpoint becomes the live one's: A (70
        // tokens: 8128 bytes parked, 9256 with its checkpoint) fits 10000 bytes, but not next to B's checkpoint
        // (1128). Parking A would drop D (the least recently used), then A. A is not kept; D stays.
        std::vector<int32_t> D = random_ids(5), B = random_ids(20), A = random_ids(70);
        D[0] = 1, B[0] = 2, A[0] = 3;
        const std::vector<int32_t> B2 = cat(cat(B, Conv::answer(B, 1)), {4, 5, 6});
        const std::vector<int32_t> D2 = cat(cat(D, Conv::answer(D, 1)), {7, 8});
        const auto l =
            run_conv(gen_line(1, D) + gen_line(1, B) + gen_line(1, A) + gen_line(1, B2) + gen_line(1, D2), 10000, 1);
        const auto r = replies(l);
        expect(r.size() == 5 && r[3].resume == 20 && r[4].err.empty() && r[4].resume == 5 &&
                   r[4].toks == Conv::answer(D2, 1),
               "... nor when the conversation it gives way to keeps its checkpoints", l);
    }
    {
        // end=: the id it ended after is not in the state, so the prompt and every token it printed, and more,
        // continue the state (the server's next pass after a forced call's </think>)
        std::vector<int32_t> P = random_ids(30);
        P[0] = 5;
        const auto ans = Conv::answer(P, 8);
        size_t k = 2;                                  // from the third token on, the first not printed before it
        while (k < ans.size() && std::find(ans.begin(), ans.begin() + (long) k, ans[k]) != ans.begin() + (long) k) ++k;
        const std::vector<int32_t> toks(ans.begin(), ans.begin() + (long) std::min(k + 1, ans.size()));
        const std::vector<int32_t> P2 = cat(cat(P, toks), {7, 8});
        const auto l = run_conv(gen_line(8, P, -1, " end=" + std::to_string(toks.back())) + gen_line(2, P2),
                                (size_t) 1 << 30, 1000);
        const auto r = replies(l);
        expect(k < ans.size() && r.size() == 2 && r[0].toks == toks && r[1].err.empty() &&
                   r[1].resume == (long) (P.size() + toks.size() - 1) && r[1].toks == Conv::answer(P2, 2),
               "end=: the id it ended after is not taken in, and the next prompt goes on from the state", l);
    }
    {
        // cp=0: no checkpoint (the server's later passes of a reply, whose prompts end inside it); without cp= one is
        // kept at the prompt's end, and a prompt that leaves the state after it resumes there
        std::vector<int32_t> P = random_ids(20);
        P[0] = 6;
        const auto ans = Conv::answer(P, 3);
        const std::vector<int32_t> P2 = cat(P, {(ans[0] + 1) % 31});     // leaves the state right after P
        const auto with = replies(run_conv(gen_line(3, P) + gen_line(1, P2), (size_t) 1 << 30, 1));
        const auto l = run_conv(gen_line(3, P, 0) + gen_line(1, P2), (size_t) 1 << 30, 1);
        const auto r = replies(l);
        expect(with.size() == 2 && with[1].resume == (long) P.size() && r.size() == 2 && r[1].err.empty() &&
                   r[1].resume == 0 && r[1].toks == Conv::answer(P2, 1),
               "cp=0: no checkpoint, where without cp= one is kept at the prompt's end", l);
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
    // strata-glm in --serve mode: serve_signals_starting() first, serve_signals_serving() before the loop. This
    // process writes to engines that may have ended (EPIPE, not the end of the test); its children inherit SIGPIPE
    // ignored, as from a launcher that ignores it, and serve_signals_starting() must restore it.
    std::signal(SIGPIPE, SIG_IGN);
    // (exit code 40: the loop returned, where strata-glm saves its expert profile; nothing is printed after it)
    auto engine = [](const std::string& mode, int64_t max_ctx, std::vector<int32_t> eos) {
        return [=] {
            serve_signals_starting();
            Fake ev;
            ev.mode = mode;
            ServeOptions opt;
            opt.max_ctx = max_ctx;
            opt.n_vocab = V;
            opt.eos_ids = eos;
            serve_signals_serving();
            return 40 + serve_loop(ev, opt);
        };
    };
    {
        Proc p = spawn(engine("plain", 64, {31}));
        const bool ready = read_until(p, "READY", 10);
        for (int sig : {SIGINT, SIGHUP, SIGTERM}) kill(p.pid, sig);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const std::string in = "GEN 2 1,2\nQUIT\n";
        const bool sent = write(p.in, in.data(), in.size()) == (ssize_t) in.size();
        const bool done = read_until(p, "DONE 2 2 ", 10);
        const int st = wait_for(p.pid, 10);
        close(p.in);
        close(p.out);
        if (st == -1) reap(p.pid);
        expect(ready && sent && done && st != -1 && WIFEXITED(st) && WEXITSTATUS(st) == 40,
               "serving: SIGINT, SIGHUP and SIGTERM are ignored while the server runs; QUIT ends the engine, which "
               "saves its profile", {p.got});
    }
    {
        Proc p = spawn(engine("plain", 64, {31}));
        const bool ready = read_until(p, "READY", 10);
        close(p.in);                                     // the server is gone: both its ends close
        close(p.out);
        const int st = wait_for(p.pid, 10);
        if (st == -1) reap(p.pid);
        expect(ready && st != -1 && WIFEXITED(st) && WEXITSTATUS(st) == 40,
               "idle when its server goes: the engine leaves its loop at the end of its input (the profile is saved)",
               {p.got});
    }
    {
        // no end of turn and a context of a million: without the broken pipe it would run for hours
        Proc p = spawn(engine("slow-decode", 1 << 20, {}));
        const std::string in = "GEN 1000000 1,2\n";
        const bool sent = write(p.in, in.data(), in.size()) == (ssize_t) in.size();
        const bool decoding = read_until(p, "T ", 10);
        close(p.in);
        close(p.out);
        const int st = wait_for(p.pid, 5);
        if (st == -1) reap(p.pid);
        expect(sent && decoding && st != -1 && WIFSIGNALED(st) && WTERMSIG(st) == SIGPIPE,
               "a request running when its server goes: the engine ends at its next output line (SIGPIPE)", {});
    }
    {
        // a prompt chunk of 4.5 s: the next HB or PP line meets the broken pipe
        Proc p = spawn(engine("slow-progress", 1 << 20, {}));
        const std::string in = "GEN 1000000 1,2\n";
        const bool sent = write(p.in, in.data(), in.size()) == (ssize_t) in.size();
        const bool ready = read_until(p, "READY", 10);
        close(p.in);
        close(p.out);
        const int st = wait_for(p.pid, 10);
        if (st == -1) reap(p.pid);
        expect(sent && ready && st != -1 && WIFSIGNALED(st) && WTERMSIG(st) == SIGPIPE,
               "a prompt being read when its server goes: the engine ends at its next heartbeat (SIGPIPE)", {});
    }
    // the orphaned engines below become this process's children, so their end can be waited for
    const bool reaper = prctl(PR_SET_CHILD_SUBREAPER, 1) == 0;
    {
        const auto [server, eng] = server_with_engine(false);
        kill(server, SIGKILL);
        waitpid(server, nullptr, 0);
        const int st = eng > 0 ? wait_for(eng, 5) : -1;
        if (st == -1 && eng > 0) reap(eng);
        expect(reaper && st != -1 && WIFSIGNALED(st) && WTERMSIG(st) == SIGKILL,
               "while it starts (the warm-up), the engine ends at once when its server is gone", {});
    }
    {
        const auto [server, eng] = server_with_engine(true);
        kill(eng, SIGTERM);                              // a systemd stop signals every process of the service
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        const bool ignored = eng > 0 && running(eng);
        kill(server, SIGKILL);
        waitpid(server, nullptr, 0);
        kill(eng, SIGTERM);
        const int st = eng > 0 ? wait_for(eng, 5) : -1;
        if (st == -1 && eng > 0) reap(eng);
        expect(reaper && ignored && st != -1 && WIFSIGNALED(st) && WTERMSIG(st) == SIGTERM,
               "serving: SIGTERM is ignored while the server runs, and ends the engine once the server is gone", {});
    }
    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures != 0;
}
