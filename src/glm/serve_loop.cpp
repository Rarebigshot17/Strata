// src/glm/serve_loop.cpp - the engine side of the line protocol, and the conversations it keeps (serve_loop.hpp).
#include "strata/glm/serve_loop.hpp"

#include "strata/glm/sampler.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <cstdarg>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>

#include <unistd.h>

namespace strata::glm {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

/// stdin on a thread: STOP takes effect at once (it arrives while a request runs); every other line is queued.
/// A STOP ends the request of the last GEN line before it, even one still queued, and nothing later.
/// The thread is still blocked reading when QUIT returns, so it reads the descriptor itself: blocked in
/// std::getline it would hold stdin's stdio lock, which exit() waits for, and the process would never end.
struct Input {
    std::mutex mu;
    std::condition_variable cv;
    std::deque<std::string> lines;
    bool eof = false;
    uint64_t gens = 0;                    ///< GEN lines read (this thread only)
    std::atomic<uint64_t> stop_for{0};    ///< the number of the GEN line the last STOP ended

    static bool is_gen(const std::string& line) { return line.rfind("GEN ", 0) == 0; }
    /// Whether the n-th GEN line was stopped.
    bool stopped(uint64_t n) const { return stop_for.load() == n; }

    void run() {
        std::string line;
        char buf[1 << 16];
        for (;;) {
            const ssize_t n = ::read(STDIN_FILENO, buf, sizeof buf);
            if (n < 0 && errno == EINTR) continue;
            if (n <= 0) break;
            for (ssize_t i = 0; i < n; ++i) {
                if (buf[i] != '\n') {
                    line.push_back(buf[i]);
                    continue;
                }
                take(line);
                line.clear();
            }
        }
        if (!line.empty()) take(line);
        std::lock_guard<std::mutex> lk(mu);
        eof = true;
        cv.notify_one();
    }

    void take(std::string line) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "STOP") {
            stop_for = gens;
            return;
        }
        if (is_gen(line)) ++gens;
        std::lock_guard<std::mutex> lk(mu);
        lines.push_back(std::move(line));
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
        if (*end != '\0' || v != (int32_t) v) return false;   // beyond 32 bits it would wrap to another id
        out.push_back((int32_t) v);
    }
    return !out.empty();
}

size_t common_prefix(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    const size_t n = std::min(a.size(), b.size());
    size_t i = 0;
    while (i < n && a[i] == b[i]) ++i;
    return i;
}

double mib(size_t b) { return (double) b / (double) (1ull << 20); }

/// The conversations: the live one, whose state is in the engine, and parked ones, whose state is a full snapshot
/// in host memory. Each may have checkpoints (light snapshots) to restart from inside it.
class Conversations {
public:
    Conversations(Evaluator& ev, const ServeOptions& opt)
        : ev_(ev), opt_(opt), enabled_(opt.park_bytes > 0 && ev.can_snapshot()) {
        sessions_.emplace_back();
    }

    bool enabled() const { return enabled_; }
    /// The tokens the engine's state holds.
    const std::vector<int32_t>& tokens() const { return sessions_[cur_].tokens; }
    void append(const int32_t* ids, int64_t n) {
        auto& t = sessions_[cur_].tokens;
        t.insert(t.end(), ids, ids + n);
    }
    std::vector<float> logits;    ///< after the last of tokens()

    /// Puts the engine in the saved state that covers the longest prefix of `ids`, and returns that length
    /// (0: the engine was reset). The live conversation is then the one it came from, cut to that prefix.
    size_t prepare(const std::vector<int32_t>& ids) {
        const uint64_t now = ++clock_;
        struct Pick {
            size_t sess = 0, pos = 0;
            int cost = 0;
            const Snapshot* cp = nullptr;
        } best;
        // the most tokens; on a tie, the cheapest: the live state, a checkpoint of it, a parked conversation, a
        // checkpoint of one
        auto consider = [&](size_t s, size_t pos, int cost, const Snapshot* cp) {
            if (pos > 0 && (pos > best.pos || (pos == best.pos && cost < best.cost))) best = {s, pos, cost, cp};
        };
        for (size_t s = 0; s < sessions_.size(); ++s) {
            const Session& se = sessions_[s];
            const bool live = s == cur_;
            if (!live && !se.park) continue;
            const size_t common = common_prefix(se.tokens, ids);
            if (common == se.tokens.size() && (!live || !logits.empty())) consider(s, common, live ? 0 : 2, nullptr);
            for (const auto& cp : se.cps)
                if ((size_t) cp->pos <= common) consider(s, (size_t) cp->pos, live ? 1 : 3, cp.get());
        }

        if (best.pos == 0) {                          // nothing to continue: a new conversation
            if (!sessions_[cur_].tokens.empty()) {
                park_live("a new conversation", 0);
                sessions_.emplace_back();
                cur_ = sessions_.size() - 1;
            }
            ev_.reset();
            logits.clear();
            sessions_[cur_].used = now;
            evict();
            return 0;
        }
        const auto t0 = Clock::now();
        if (best.sess != cur_) {                      // another conversation: park this one, bring that one back
            size_t keep = 0;                          // the checkpoints that one keeps
            for (const auto& cp : sessions_[best.sess].cps)
                if ((size_t) cp->pos <= best.pos) keep += cp->bytes;
            park_live("another conversation", keep);
            cur_ = best.sess;
            const std::shared_ptr<Snapshot> p = std::move(sessions_[cur_].park);   // the engine holds it now
            ev_.restore(*p);
            logits = p->logits;
        }
        Session& lv = sessions_[cur_];
        if (best.cp) {
            ev_.restore(*best.cp);
            logits = best.cp->logits;
        }
        if (best.cost > 0)
            std::fprintf(stderr, "conversations: resumed %s at %zu of %zu prompt tokens (%.0f ms)\n",
                         best.cost == 2 ? "a parked conversation" : best.cost == 1 ? "a checkpoint"
                                                                                  : "a parked conversation's checkpoint",
                         best.pos, ids.size(), ms_since(t0));
        lv.tokens.resize(best.pos);
        lv.cps.erase(std::remove_if(lv.cps.begin(), lv.cps.end(),
                                    [&](const std::shared_ptr<Snapshot>& c) { return (size_t) c->pos > best.pos; }),
                     lv.cps.end());
        lv.used = now;
        evict();
        return best.pos;
    }

    /// Saves a checkpoint of the live state.
    void checkpoint() {
        if (!enabled_ || opt_.max_checkpoints <= 0) return;
        Session& lv = sessions_[cur_];
        if (!lv.cps.empty() && (size_t) lv.cps.back()->pos >= lv.tokens.size()) return;
        auto cp = save(false, 0);                      // the newest checkpoint: evict() drops everything else first
        if (!cp) return;
        lv.cps.push_back(std::move(cp));
        while ((int) lv.cps.size() > opt_.max_checkpoints) lv.cps.erase(lv.cps.begin());
        evict();
    }

    /// The live state is unknown (an error) or unusable (NaN): start it over.
    void discard() {
        sessions_[cur_] = Session{};
        logits.clear();
        ev_.reset();
    }

    std::string summary() const {
        size_t parked = 0, bytes = 0, cps = 0;
        for (size_t s = 0; s < sessions_.size(); ++s) {
            parked += s != cur_;
            bytes += sessions_[s].bytes();
            cps += sessions_[s].cps.size();
        }
        char buf[160];
        std::snprintf(buf, sizeof buf, "%zu parked, %zu checkpoints, %.3f of %.1f MiB", parked, cps, mib(bytes),
                      mib(opt_.park_bytes));
        return buf;
    }

private:
    struct Session {
        std::vector<int32_t> tokens;                  ///< what the state holds (parked: what `park` holds)
        std::shared_ptr<Snapshot> park;               ///< the full state, while not live
        std::vector<std::shared_ptr<Snapshot>> cps;   ///< by position, all within `tokens`
        uint64_t used = 0;
        size_t bytes() const {
            size_t b = park ? park->bytes : 0;
            for (const auto& c : cps) b += c->bytes;
            return b;
        }
    };

    /// A snapshot of the live state with its logits; nullptr when it would not fit park_bytes next to the `with`
    /// bytes that evict() keeps as long as it keeps the snapshot (it would be dropped at once, after a copy that
    /// takes seconds for a long conversation).
    std::shared_ptr<Snapshot> save(bool full, size_t with) {
        const size_t est = ev_.snapshot_bytes(full);
        const size_t need = est + logits.size() * sizeof(float);
        if (est > 0 && need + with > opt_.park_bytes) {
            std::fprintf(stderr, "conversations: %s of %zu tokens would take %.1f MiB (%.1f MiB with the checkpoints "
                         "kept along), more than --park-mib; not kept\n", full ? "a conversation" : "a checkpoint",
                         sessions_[cur_].tokens.size(), mib(need), mib(need + with));
            return nullptr;
        }
        auto s = ev_.snapshot(full);
        if (!s) return nullptr;
        if (est > 0 && s->bytes != est)
            std::fprintf(stderr, "conversations: snapshot size estimated %zu bytes, actual %zu\n", est, s->bytes);
        s->logits = logits;
        s->bytes += logits.size() * sizeof(float);
        return s;
    }

    /// The live conversation becomes a parked one (or is dropped by evict() when it cannot be). It is the most
    /// recently used parked one, so evict() keeps it as long as it fits with its own checkpoints and the `keep`
    /// bytes of checkpoints the next live conversation holds.
    void park_live(const char* why, size_t keep) {
        Session& lv = sessions_[cur_];
        if (!enabled_ || lv.tokens.empty() || logits.empty()) return;
        const auto t0 = Clock::now();
        lv.park = save(true, lv.bytes() + keep);
        if (lv.park)
            std::fprintf(stderr, "conversations: parked %zu tokens (%.3f MiB, %.0f ms) for %s\n", lv.tokens.size(),
                         mib(lv.park->bytes), ms_since(t0), why);
    }

    void erase(size_t i) {
        sessions_.erase(sessions_.begin() + (long) i);
        if (cur_ > i) --cur_;
    }

    /// Drops what cannot be resumed, then the least recently used conversations, then the live one's oldest
    /// checkpoints, until the rest fits park_bytes.
    void evict() {
        for (size_t i = 0; i < sessions_.size();) {
            if (i != cur_ && !sessions_[i].park) erase(i);
            else ++i;
        }
        size_t used = 0;
        for (const auto& s : sessions_) used += s.bytes();
        while (used > opt_.park_bytes) {
            size_t victim = sessions_.size();
            for (size_t i = 0; i < sessions_.size(); ++i)
                if (i != cur_ && (victim == sessions_.size() || sessions_[i].used < sessions_[victim].used)) victim = i;
            if (victim < sessions_.size()) {
                std::fprintf(stderr, "conversations: dropped a parked conversation of %zu tokens (memory)\n",
                             sessions_[victim].tokens.size());
                used -= sessions_[victim].bytes();
                erase(victim);
                continue;
            }
            auto& cps = sessions_[cur_].cps;
            if (cps.empty()) break;
            used -= cps.front()->bytes;
            cps.erase(cps.begin());
        }
    }

    Evaluator& ev_;
    const ServeOptions& opt_;
    const bool enabled_;
    std::vector<Session> sessions_;   ///< sessions_[cur_] is live
    size_t cur_ = 0;
    uint64_t clock_ = 0;
};

/// stdout: the protocol's lines, from the request loop and the heartbeat thread, one line at a time.
class Out {
public:
    void line(const char* fmt, ...) __attribute__((format(printf, 2, 3))) {
        std::lock_guard<std::mutex> lk(mu_);
        va_list ap;
        va_start(ap, fmt);
        std::vfprintf(stdout, fmt, ap);
        va_end(ap);
        std::fputc('\n', stdout);
        std::fflush(stdout);
        last_ = Clock::now();
    }
    /// HB when nothing was written for `quiet` seconds.
    void beat(double quiet) {
        std::lock_guard<std::mutex> lk(mu_);
        if (std::chrono::duration<double>(Clock::now() - last_).count() < quiet) return;
        std::fputs("HB\n", stdout);
        std::fflush(stdout);
        last_ = Clock::now();
    }

private:
    std::mutex mu_;
    Clock::time_point last_ = Clock::now();
};

/// While a request runs, says HB every few quiet seconds in which the engine made progress (its ticks moved), so
/// the server can tell a long prompt chunk (or a cold page cache) from a hung engine.
class Heartbeat {
public:
    Heartbeat(Evaluator& ev, Out& out) : ev_(ev), out_(out), th_([this] { run(); }) {}
    ~Heartbeat() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            quit_ = true;
        }
        cv_.notify_one();
        th_.join();
    }
    void active(bool on) { active_ = on; }

private:
    void run() {
        uint64_t seen = ev_.ticks();
        std::unique_lock<std::mutex> lk(mu_);
        while (!cv_.wait_for(lk, std::chrono::seconds(1), [&] { return quit_; })) {
            const uint64_t now = ev_.ticks();
            if (active_ && now != seen) out_.beat(3.0);
            seen = now;
        }
    }
    Evaluator& ev_;
    Out& out_;
    std::mutex mu_;
    std::condition_variable cv_;
    bool quit_ = false;
    std::atomic<bool> active_{false};
    std::thread th_;
};

}  // namespace

int serve_loop(Evaluator& ev, const ServeOptions& opt) {
    auto input = std::make_shared<Input>();          // the reader thread outlives this function
    Input& in = *input;
    std::thread reader([input] { input->run(); });
    reader.detach();
    Conversations conv(ev, opt);
    Out out;
    Heartbeat heartbeat(ev, out);

    {
        std::string info = "INFO engine=" + opt.version + " eos=" + std::to_string(opt.eos_ids.size()) +
                           " park_mib=" + std::to_string(conv.enabled() ? opt.park_bytes >> 20 : (size_t) 0) +
                           " logit_bias=1";
        for (const auto& kv : opt.info) info += " " + kv;
        out.line("%s", info.c_str());
        out.line("READY %lld stop", (long long) opt.max_ctx);
    }

    std::string line;
    uint64_t gen_no = 0;                              // this request's GEN line, counted as Input counts them
    while (in.next(line)) {
        if (line == "QUIT") break;
        if (!Input::is_gen(line)) {
            out.line("ERR unsupported command: %s", line.substr(0, line.find(' ')).c_str());
            continue;
        }
        // GEN <max_new> [key=value ...] <ids>
        ++gen_no;
        std::vector<std::string> f;
        {
            std::stringstream ss(line);
            std::string w;
            while (ss >> w) f.push_back(w);
        }
        std::vector<int32_t> ids, ends;               // ends: end=, ids that end this request as an end of turn
        SamplingParams sp;
        int64_t cp_at = -1;
        bool bad = f.size() < 3 || !parse_ids(f.back(), ids);
        const long long max_new = bad ? 0 : std::atoll(f[1].c_str());
        for (size_t i = 2; !bad && i + 1 < f.size(); ++i) {
            const size_t eq = f[i].find('=');
            if (eq == std::string::npos) continue;
            const std::string key = f[i].substr(0, eq), value = f[i].substr(eq + 1);
            if (key == "cp") cp_at = std::atoll(value.c_str());
            else if (key == "end") bad = !parse_ids(value, ends);
            else sp.set(key, value);                  // unknown keys: ignored
        }
        if (bad || max_new < 0) {
            out.line("ERR malformed GEN line");
            continue;
        }
        if (const std::string why = sp.check(opt.n_vocab); !why.empty()) {
            out.line("ERR %s", why.c_str());
            continue;
        }
        for (int32_t id : ids)
            if (id < 0 || (opt.n_vocab > 0 && id >= opt.n_vocab)) {
                bad = true;
                break;
            }
        if (bad) {
            out.line("ERR a token id outside the vocabulary");
            continue;
        }
        if ((int64_t) ids.size() + 1 > opt.max_ctx) {
            out.line("ERR the prompt (%zu tokens) does not fit the context (%lld)", ids.size(), (long long) opt.max_ctx);
            continue;
        }
        const size_t total = ids.size();
        if (in.stopped(gen_no)) {                     // stopped while queued: the state is not touched
            out.line("DONE 0 %zu 0.0 0.0 cancel 0 0 0 0 0 0 0 0.0 0 0", total);
            continue;
        }
        const auto t0 = Clock::now();
        heartbeat.active(true);

        std::string finish;
        size_t reuse = 0, at = 0;
        try {
            reuse = at = conv.prepare(ids);
            if (reuse > 0) out.line("RESUME %zu", reuse);
            // a checkpoint at cp=<n> (the server's: before the generation prompt), else at the prompt's end
            int64_t ck = cp_at >= 0 ? cp_at : (int64_t) total;
            if (ck <= (int64_t) reuse || ck > (int64_t) total || ck < opt.checkpoint_min) ck = -1;
            auto last_pp = Clock::now();
            for (const int64_t to : {ck, (int64_t) total}) {
                if (to <= (int64_t) at) continue;
                const size_t from = at;
                const int64_t before = ev.pos();
                conv.logits = ev.prefill(ids.data() + from, to - (int64_t) from, [&](int64_t done_new) {
                    const size_t done = from + (size_t) done_new;
                    if (done == total || std::chrono::duration<double>(Clock::now() - last_pp).count() >= 0.25) {
                        last_pp = Clock::now();
                        const double ms = ms_since(t0);
                        out.line("PP %zu %zu %.1f %.1f", done, total, ms,
                                 ms > 0 ? (double) (done - reuse) * 1000.0 / ms : 0.0);
                    }
                    return !in.stopped(gen_no);
                });
                const int64_t got = ev.pos() - before;
                conv.append(ids.data() + from, got);
                at = from + (size_t) got;
                if ((int64_t) at < to) {
                    finish = "cancel";
                    break;
                }
                if (to == ck) conv.checkpoint();
            }
        } catch (const std::exception& e) {
            heartbeat.active(false);
            out.line("ERR internal: %s", e.what());
            conv.discard();
            continue;
        }
        const double prompt_ms = ms_since(t0);
        const auto t1 = Clock::now();
        const int64_t hits0 = ev.hits(), look0 = ev.lookups();   // the expert cache during decode
        long long produced = 0;
        if (finish.empty() && max_new == 0) finish = "length";
        if (finish.empty()) {
            Sampler sampler(sp);
            std::vector<int32_t> hist = conv.tokens();
            try {
                for (;;) {
                    bool nonfinite = false;
                    for (float v : conv.logits)
                        if (!std::isfinite(v)) { nonfinite = true; break; }
                    if (nonfinite) { finish = "nonfinite"; break; }
                    const int32_t y = sampler.sample(conv.logits, hist);
                    out.line("T %d", y);
                    ++produced;
                    hist.push_back(y);
                    if (std::find(opt.eos_ids.begin(), opt.eos_ids.end(), y) != opt.eos_ids.end() ||
                        std::find(ends.begin(), ends.end(), y) != ends.end()) { finish = "stop"; break; }
                    if (in.stopped(gen_no)) { finish = "cancel"; break; }
                    if (produced >= max_new || (int64_t) conv.tokens().size() + 1 >= opt.max_ctx) { finish = "length"; break; }
                    conv.logits = ev.eval(y);
                    conv.append(&y, 1);
                }
            } catch (const std::exception& e) {
                heartbeat.active(false);
                out.line("ERR internal: %s", e.what());
                conv.discard();
                continue;
            }
        }
        heartbeat.active(false);
        if (finish == "nonfinite") conv.discard();   // a state that produced NaN is not reused
        const double decode_ms = ms_since(t1);
        out.line("DONE %lld %zu %.1f %.1f %s 0 0 %zu %lld %lld 0 0 0.0 %zu 0", produced, total, prompt_ms, decode_ms,
                 finish.c_str(), reuse, (long long) (ev.hits() - hits0), (long long) (ev.lookups() - look0),
                 at - reuse);
        if (conv.enabled()) std::fprintf(stderr, "conversations: %s\n", conv.summary().c_str());
    }
    return 0;
}

}  // namespace strata::glm
