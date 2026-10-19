// tests/glm/sampler_test.cpp - the GLM sampler (src/glm/sampler.cpp) against a full-sort implementation of the same
// chain: the draws must follow the reference distribution within sampling noise and never leave its support. Also
// logit_bias (bans, boosts, refusals) and, for information, the time per token on GLM's vocabulary.
#include "strata/glm/sampler.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <numeric>
#include <random>
using namespace strata::glm;

// reference: the old implementation's semantics with a full sort
std::vector<double> reference(std::vector<float> l, const SamplingParams& p) {
    const int n = (int) l.size();
    for (auto [id, v] : p.logit_bias) l[id] = v <= -100 ? -INFINITY : l[id] + v;
    std::vector<int> idx(n); std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return l[a] > l[b]; });
    int keep = (p.top_k > 0 && p.top_k < n) ? p.top_k : n;
    idx.resize(keep);
    auto probs = [&](double t) { std::vector<double> pr(idx.size()); double mx = l[idx[0]], s = 0;
        for (size_t i = 0; i < idx.size(); ++i) s += (pr[i] = std::exp((l[idx[i]] - mx) / t));
        for (auto& v : pr) v /= s; return pr; };
    if (p.top_p < 1) { auto pr = probs(1); double a = 0; size_t cut = idx.size();
        for (size_t i = 0; i < idx.size(); ++i) { a += pr[i]; if (a >= p.top_p) { cut = i + 1; break; } }
        idx.resize(std::max<size_t>(cut, 1)); }
    if (p.min_p > 0) { auto pr = probs(1); size_t cut = 1;
        while (cut < idx.size() && pr[cut] >= p.min_p * pr[0]) ++cut; idx.resize(cut); }
    auto pr = probs(p.temperature);
    std::vector<double> full(n, 0.0);
    for (size_t i = 0; i < idx.size(); ++i) full[idx[i]] = pr[i];
    return full;
}

int main() {
    std::mt19937 g(5);
    int bad = 0;
    struct Case { int n; float temp, top_p; int top_k; float min_p; float sharp; };
    for (Case c : {Case{2000, 1.0f, 1.0f, 0, 0.0f, 3}, Case{2000, 0.7f, 0.9f, 0, 0.0f, 3}, Case{2000, 1.0f, 0.95f, 40, 0.0f, 2},
                   Case{2000, 1.0f, 1.0f, 0, 0.05f, 3}, Case{2000, 0.8f, 0.8f, 0, 0.02f, 1}, Case{3000, 1.2f, 0.99f, 0, 0.0f, 0.5f}}) {
        std::normal_distribution<float> nd(0, c.sharp);
        std::vector<float> l(c.n); for (auto& v : l) v = nd(g);
        SamplingParams p; p.temperature = c.temp; p.top_p = c.top_p; p.top_k = c.top_k; p.min_p = c.min_p;
        p.has_seed = true; p.seed = 11;
        auto ref = reference(l, p);
        Sampler s(p);
        const int N = 60000;
        std::vector<int> cnt(c.n, 0);
        for (int i = 0; i < N; ++i) ++cnt[s.sample(l, {})];
        // total variation distance and support check
        double tv = 0; int outside = 0;
        for (int i = 0; i < c.n; ++i) { tv += std::fabs(cnt[i] / (double) N - ref[i]); if (cnt[i] && ref[i] == 0) ++outside; }
        tv /= 2;
        int support = 0; for (double v : ref) support += v > 0;
        double noise = 0; for (double v : ref) noise += std::sqrt(2 * v * (1 - v) / (3.14159265358979323846 * N));
        noise /= 2;
        const bool ok = tv < 1.3 * noise + 0.003 && outside == 0;
        bad += !ok;
        std::printf("n=%d temp=%.2f top_p=%.2f top_k=%d min_p=%.2f: support %d, TV %.4f (noise %.4f), outside %d %s\n", c.n, c.temp,
                    c.top_p, c.top_k, c.min_p, support, tv, noise, outside, ok ? "ok" : "FAIL");
    }
    // logit_bias: a ban never comes out; +100 always does; greedy honours it
    {
        std::vector<float> l(500, 0.0f); l[7] = 10.0f;
        SamplingParams p; p.temperature = 1.0f; p.set("logit_bias", "7:-100,9:100"); p.has_seed = true; p.seed = 1;
        Sampler s(p); int nine = 0;
        for (int i = 0; i < 1000; ++i) { int t = s.sample(l, {}); if (t == 7) { bad++; break; } nine += t == 9; }
        SamplingParams q; q.set("logit_bias", "7:-100"); Sampler gq(q);
        const int greedy = gq.sample(l, {});
        std::printf("logit_bias: banned never drawn, +100 drawn %d/1000, greedy with the best banned -> %d (not 7) %s\n",
                    nine, greedy, nine == 1000 && greedy != 7 ? "ok" : "FAIL");
        bad += !(nine == 1000 && greedy != 7);
        SamplingParams e;
        for (const char* v : {"7:-101", "7:1,7:2", "x:1", "7", "7:1,"}) { e.set("logit_bias", v);
            std::printf("  refused %-8s -> %s\n", v, e.check(500).c_str()); bad += e.check(500).empty(); }
        e.set("logit_bias", "600:1"); std::printf("  refused 600:1    -> %s\n", e.check(500).c_str()); bad += e.check(500).empty();
        e.set("logit_bias", "1:-100,0:-100"); std::printf("  2-token vocab all banned -> %s\n", e.check(2).c_str()); bad += e.check(2).empty();
    }
    // settings that leave nothing to draw from are refused: a repetition penalty of 0 or below (a positive logit
    // over 0 is infinite; a negative one turned a ban's -inf into +inf, so the banned token came out) and values
    // that are not finite numbers
    {
        std::vector<float> l(10, 0.0f); l[3] = 2.0f;
        SamplingParams q; q.set("logit_bias", "3:-100"); q.set("penalty_repeat", "-1"); Sampler gq(q);
        const int got = gq.sample(l, {3});
        std::printf("penalty_repeat -1 with token 3 banned and in the history: greedy -> %d %s\n", got, got != 3 ? "ok" : "FAIL");
        bad += got == 3;
        for (auto [key, value] : {std::pair{"penalty_repeat", "0"}, {"penalty_repeat", "-1"}, {"penalty_repeat", "nan"},
                                  {"temperature", "inf"}, {"temperature", "1e39"}, {"temperature", "nan"},
                                  {"top_p", "nan"}, {"min_p", "inf"}, {"penalty_freq", "-inf"},
                                  {"penalty_present", "nan"}}) {
            SamplingParams p; p.set("logit_bias", "3:-100"); p.set(key, value);
            const std::string why = p.check(10);
            std::printf("  refused %s=%-5s -> %s\n", key, value, why.empty() ? "(accepted) FAIL" : why.c_str());
            bad += why.empty();
        }
        SamplingParams ok; ok.set("penalty_repeat", "1.1"); ok.set("temperature", "0.7"); ok.set("penalty_freq", "-0.5");
        std::printf("  accepted penalty_repeat=1.1 temperature=0.7 penalty_freq=-0.5 %s\n", ok.check(10).empty() ? "ok" : "FAIL");
        bad += !ok.check(10).empty();
    }
    // finite penalties so large that they overflow: a ban stays one (-inf minus -inf was NaN, which greedy took at
    // id 0 and a draw at the end), and an overflowing boost is the most likely token, not +inf (NaN weights: every
    // draw went to one of the boosted tokens)
    {
        int out = 0;
        for (int ban : {0, 9})
            for (const char* temp : {"0", "1"})
                for (const char* top_p : {"1", "0.9"}) {
                    std::vector<float> l(10, 0.0f); l[5] = 2.0f;
                    SamplingParams p; p.set("logit_bias", (std::to_string(ban) + ":-100").c_str());
                    p.set("penalty_freq", "-2e38"); p.set("temperature", temp); p.set("top_p", top_p); p.set("seed", "1");
                    Sampler s(p);
                    for (int i = 0; i < 100; ++i) out += s.sample(l, {ban, ban, 5}) == ban;
                }
        std::vector<float> l(10, 0.0f); l[5] = 2.0f; l[6] = 1.5f;
        SamplingParams p; p.set("penalty_repeat", "1e-40"); p.set("temperature", "1"); p.set("seed", "1");
        Sampler s(p); int five = 0, six = 0;
        for (int i = 0; i < 400; ++i) { const int t = s.sample(l, {5, 6}); five += t == 5; six += t == 6; }
        std::printf("penalty_freq -2e38 with the banned token in the history: banned drawn %d/800; penalty_repeat 1e-40 "
                    "boosting 5 and 6: %d and %d of 400 %s\n", out, five, six,
                    out == 0 && five > 100 && six > 100 && five + six == 400 ? "ok" : "FAIL");
        bad += !(out == 0 && five > 100 && six > 100 && five + six == 400);
    }
    // integer keys beyond int's range are its nearest end: a penalty window of 2^32 tokens covers the whole history
    // (cast to int it read INT_MIN on x86, a window of nothing), a top_k of 1e20 is off
    {
        std::vector<float> l(10, 0.0f); l[3] = 2.0f;
        SamplingParams p; p.set("penalty_freq", "5"); p.set("penalty_last_n", "4294967296"); Sampler s(p);
        const int got = s.sample(l, {3, 3});
        SamplingParams k; k.set("top_k", "1e20");
        std::printf("penalty_last_n 4294967296: token 3 penalized -> greedy %d (not 3); top_k 1e20 -> %d %s\n", got,
                    k.top_k, got != 3 && k.top_k > 10 ? "ok" : "FAIL");
        bad += !(got != 3 && k.top_k > 10);
    }
    // drawn=: a seeded sampler that skips the draws of a reply's earlier passes draws what one sampler going on would
    // (a draw a token, also where one candidate is left), so a reply the server makes in several passes is the one a
    // single pass gives; without it the second pass drew with the seed's first numbers again
    {
        std::vector<float> l(300); std::normal_distribution<float> nd(0, 1); for (auto& v : l) v = nd(g);
        std::vector<float> one_left = l; one_left[17] = 30.0f;     // min_p 0.05 keeps token 17 alone
        int same = 0;
        for (const char* top_p : {"1", "0.9"}) {
            SamplingParams p; p.set("temperature", "1"); p.set("top_p", top_p); p.set("min_p", "0.05");
            p.set("seed", "3");
            auto logits = [&](int i) -> const std::vector<float>& { return i % 3 ? l : one_left; };
            Sampler one(p); std::vector<int> a, b;
            for (int i = 0; i < 40; ++i) a.push_back(one.sample(logits(i), {}));
            Sampler first(p);
            for (int i = 0; i < 25; ++i) b.push_back(first.sample(logits(i), {}));
            SamplingParams q = p; q.set("drawn", "25"); Sampler rest(q);
            for (int i = 25; i < 40; ++i) b.push_back(rest.sample(logits(i), {}));
            same += a == b;
        }
        std::printf("drawn=25: a pass after 25 tokens draws as one sampler going on: %d of 2 %s\n", same,
                    same == 2 ? "ok" : "FAIL");
        bad += same != 2;
    }
    // timing on GLM's vocabulary
    {
        std::vector<float> l(154880); std::normal_distribution<float> nd(0, 2); for (auto& v : l) v = nd(g);
        for (auto [name, tp, tk] : {std::tuple{"top_p 0.95", 0.95f, 0}, std::tuple{"plain draw", 1.0f, 0}, std::tuple{"top_k 40", 1.0f, 40}}) {
            SamplingParams p; p.temperature = 1.0f; p.top_p = tp; p.top_k = tk; Sampler s(p);
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < 50; ++i) s.sample(l, {});
            std::printf("%-10s %.2f ms per token\n", name, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / 50);
        }
    }
    std::printf(bad ? "FAILED\n" : "all ok\n");
    return bad != 0;
}
