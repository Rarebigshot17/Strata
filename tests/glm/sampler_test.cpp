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
