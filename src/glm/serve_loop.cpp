// src/glm/serve_loop.cpp - the engine side of the line protocol (serve_loop.hpp).
#include "strata/glm/serve_loop.hpp"

#include "strata/glm/sampler.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

namespace strata::glm {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

/// stdin on a thread: STOP sets the flag at once (it arrives while a request runs); every other line is queued.
struct Input {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    std::atomic<bool> stop{false};

    void run() {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line == "STOP") {
                stop = true;
                continue;
            }
            std::lock_guard<std::mutex> lk(mu);
            lines.push_back(line);
            cv.notify_one();
        }
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    }

    bool next(std::string& out) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return eof || !lines.empty(); });
        if (lines.empty()) return false;
        out = std::move(lines.front());
        lines.pop_front();
        return true;
    }
};

bool parse_ids(const std::string& s, std::vector<int32_t>& out) {
    out.clear();
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) return false;
        char* end = nullptr;
        const long v = std::strtol(item.c_str(), &end, 10);
        if (*end != '\0') return false;
        out.push_back((int32_t) v);
    }
    return !out.empty();
}

}  // namespace

int serve_loop(Evaluator& ev, const ServeOptions& opt) {
    Input in;
    std::thread reader([&] { in.run(); });
    reader.detach();

    std::printf("INFO engine=glm-0.1 eos=%zu", opt.eos_ids.size());
    for (const auto& kv : opt.info) std::printf(" %s", kv.c_str());
    std::printf("\nREADY %lld stop\n", (long long) opt.max_ctx);
    std::fflush(stdout);

    std::vector<int32_t> seq;          // the tokens whose effect the state holds
    std::vector<float> last_logits;    // the logits after seq's last token
    std::string line;
    while (in.next(line)) {
        if (line == "QUIT") break;
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR unsupported command: %s\n", line.substr(0, line.find(' ')).c_str());
            std::fflush(stdout);
            continue;
        }
        // GEN <max_new> [key=value ...] <ids>
        std::vector<std::string> f;
        {
            std::stringstream ss(line);
            std::string w;
            while (ss >> w) f.push_back(w);
        }
        std::vector<int32_t> ids;
        SamplingParams sp;
        bool bad = f.size() < 3 || !parse_ids(f.back(), ids);
        const long long max_new = bad ? 0 : std::atoll(f[1].c_str());
        for (size_t i = 2; !bad && i + 1 < f.size(); ++i) {
            const size_t eq = f[i].find('=');
            if (eq != std::string::npos) sp.set(f[i].substr(0, eq), f[i].substr(eq + 1));   // unknown keys: ignored
        }
        if (bad || max_new < 0) {
            std::printf("ERR malformed GEN line\n");
            std::fflush(stdout);
            continue;
        }
        if ((int64_t) ids.size() + 1 > opt.max_ctx) {
            std::printf("ERR the prompt (%zu tokens) does not fit the context (%lld)\n", ids.size(),
                        (long long) opt.max_ctx);
            std::fflush(stdout);
            continue;
        }
        in.stop = false;
        const auto t0 = Clock::now();
        const int64_t hits0 = ev.hits(), look0 = ev.lookups();

        // reuse the state when this prompt continues it
        size_t reuse = 0;
        if (!seq.empty() && ids.size() >= seq.size() && std::equal(seq.begin(), seq.end(), ids.begin()) &&
            !last_logits.empty())
            reuse = seq.size();
        if (reuse == 0) {
            ev.reset();
            seq.clear();
            last_logits.clear();
        }
        if (reuse > 0) std::printf("RESUME %zu\n", reuse);

        std::string finish;
        try {
            const size_t total = ids.size();
            for (size_t i = reuse; i < total; ++i) {
                last_logits = ev.eval(ids[i]);
                seq.push_back(ids[i]);
                const size_t done = i + 1;
                if ((done - reuse) % 64 == 0 || done == total) {
                    const double ms = ms_since(t0);
                    std::printf("PP %zu %zu %.1f %.1f\n", done, total, ms,
                                ms > 0 ? (double) (done - reuse) * 1000.0 / ms : 0.0);
                    std::fflush(stdout);
                }
                if (in.stop) {
                    finish = "cancel";
                    break;
                }
            }
        } catch (const std::exception& e) {
            std::printf("ERR %s\n", e.what());
            std::fflush(stdout);
            ev.reset();
            seq.clear();
            last_logits.clear();
            continue;
        }
        const double prompt_ms = ms_since(t0);
        const auto t1 = Clock::now();
        long long produced = 0;
        if (finish.empty() && max_new == 0) finish = "length";
        if (finish.empty()) {
            Sampler sampler(sp);
            std::vector<int32_t> hist = seq;
            try {
                for (;;) {
                    bool nonfinite = false;
                    for (float v : last_logits)
                        if (!std::isfinite(v)) { nonfinite = true; break; }
                    if (nonfinite) { finish = "nonfinite"; break; }
                    const int32_t y = sampler.sample(last_logits, hist);
                    std::printf("T %d\n", y);
                    std::fflush(stdout);
                    ++produced;
                    hist.push_back(y);
                    if (std::find(opt.eos_ids.begin(), opt.eos_ids.end(), y) != opt.eos_ids.end()) { finish = "stop"; break; }
                    if (in.stop) { finish = "cancel"; break; }
                    if (produced >= max_new || (int64_t) seq.size() + 1 >= opt.max_ctx) { finish = "length"; break; }
                    last_logits = ev.eval(y);
                    seq.push_back(y);
                }
            } catch (const std::exception& e) {
                std::printf("ERR %s\n", e.what());
                std::fflush(stdout);
                ev.reset();
                seq.clear();
                last_logits.clear();
                continue;
            }
        }
        if (finish == "nonfinite") {               // a state that produced NaN is not reused
            ev.reset();
            seq.clear();
            last_logits.clear();
        }
        const double decode_ms = ms_since(t1);
        const size_t read = seq.size() >= reuse ? std::min(seq.size(), ids.size()) - reuse : 0;
        std::printf("DONE %lld %zu %.1f %.1f %s 0 0 %zu %lld %lld 0 0 0.0 %zu 0\n", produced, ids.size(), prompt_ms,
                    decode_ms, finish.c_str(), reuse, (long long) (ev.hits() - hits0),
                    (long long) (ev.lookups() - look0), read);
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace strata::glm
