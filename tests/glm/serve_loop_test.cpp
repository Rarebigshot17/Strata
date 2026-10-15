// tests/glm/serve_loop_test.cpp - the serve loop's line protocol (serve_loop.hpp) against scripted engines.
//
// Each case runs serve_loop in a forked child whose stdin and stdout are pipes, writes the request lines, closes
// stdin and reads everything the child printed. The fake engine predicts token (last + 1) % 32 and can be slow
// (with or without progress) or fail, so the cases cover INFO/READY, generation, end of turn, logit_bias,
// refusals, ERR internal, HB heartbeats (only while the engine progresses) and STOP of a queued request.
#include "strata/glm/serve_loop.hpp"

#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
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

/// The child's stdout lines for this mode and input.
std::vector<std::string> run(const std::string& mode, const std::string& input) {
    int in[2], out[2];
    if (pipe(in) || pipe(out)) throw std::runtime_error("pipe");
    const pid_t pid = fork();
    if (pid == 0) {
        dup2(in[0], 0);
        dup2(out[1], 1);
        close(in[0]); close(in[1]); close(out[0]); close(out[1]);
        Fake ev;
        ev.mode = mode;
        ServeOptions opt;
        opt.max_ctx = 64;
        opt.n_vocab = V;
        opt.eos_ids = {31};
        _exit(serve_loop(ev, opt));
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
    std::printf(failures ? "%d FAILED\n" : "all passed\n", failures);
    return failures != 0;
}
