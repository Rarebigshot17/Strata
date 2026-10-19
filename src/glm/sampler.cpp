// src/glm/sampler.cpp - picking the next token (sampler.hpp).
#include "strata/glm/sampler.hpp"

#include <algorithm>
#include <climits>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <unordered_map>

namespace strata::glm {

namespace {

/// An integer key's value: beyond int's range, its nearest end (a plain cast is undefined there, and on x86 gave
/// INT_MIN: penalty_last_n=4294967296 counted no tokens at all).
int to_int(double v) { return v != v ? 0 : (int) std::clamp(v, (double) INT_MIN, (double) INT_MAX); }

/// `id:value,id:value`, the Qwen engine's rules: values -100..100, -100 a ban, no duplicate ids.
std::string parse_bias(const std::string& text, std::vector<std::pair<int32_t, float>>& out) {
    out.clear();
    if (text.empty()) return "empty logit_bias";
    const char* p = text.c_str();
    for (;;) {
        if (*p < '0' || *p > '9') return "invalid logit_bias token ID";
        char* end = nullptr;
        const unsigned long id = std::strtoul(p, &end, 10);
        if (*end != ':' || id > 0x7fffffffUL) return "logit_bias ID too large or missing ':'";
        p = end + 1;
        const float value = std::strtof(p, &end);
        if (end == p || !std::isfinite(value) || value < -100 || value > 100 || (*end && *end != ','))
            return "invalid logit_bias value (expected -100..100)";
        out.emplace_back((int32_t) id, value);
        if (!*end) break;
        p = end + 1;
    }
    std::vector<int32_t> ids;
    for (const auto& b : out) ids.push_back(b.first);
    std::sort(ids.begin(), ids.end());
    if (std::adjacent_find(ids.begin(), ids.end()) != ids.end()) return "logit_bias ID duplicated";
    return "";
}

}  // namespace

bool SamplingParams::set(const std::string& key, const std::string& value) {
    if (key == "logit_bias") {
        bias_error = parse_bias(value, logit_bias);
        if (!bias_error.empty()) logit_bias.clear();
        return true;
    }
    const double v = std::atof(value.c_str());
    if (key == "temperature") temperature = (float) v;
    else if (key == "top_p") top_p = (float) v;
    else if (key == "top_k") top_k = to_int(v);
    else if (key == "min_p") min_p = (float) v;
    else if (key == "penalty_repeat") penalty_repeat = (float) v;
    else if (key == "penalty_freq") penalty_freq = (float) v;
    else if (key == "penalty_present") penalty_present = (float) v;
    else if (key == "penalty_last_n") penalty_last_n = to_int(v);
    else if (key == "seed") {
        seed = (uint64_t) std::strtoull(value.c_str(), nullptr, 10);
        has_seed = true;
    } else return false;
    return true;
}

std::string SamplingParams::check(int64_t n_vocab) const {
    if (!bias_error.empty()) return bias_error;
    // a draw needs finite weights: an infinite or NaN value leaves none, and a banned token can come out; a
    // repetition penalty of 0 or below is none (a positive logit over 0 is infinite, below 0 the signs flip)
    const std::pair<const char*, float> values[] = {{"temperature", temperature}, {"top_p", top_p}, {"min_p", min_p},
        {"penalty_repeat", penalty_repeat}, {"penalty_freq", penalty_freq}, {"penalty_present", penalty_present}};
    for (const auto& [key, v] : values)
        if (!std::isfinite(v)) return std::string(key) + " must be a finite number";
    if (penalty_repeat <= 0.0f) return "penalty_repeat must be above 0";
    int64_t banned = 0;
    for (const auto& [id, v] : logit_bias) {
        if (id < 0 || id >= n_vocab) return "logit_bias ID outside the vocabulary";
        banned += v <= -100;
    }
    if (n_vocab > 0 && banned >= n_vocab) return "logit_bias bans the entire vocabulary";
    return "";
}

Sampler::Sampler(const SamplingParams& p) : p_(p), rng_(p.has_seed ? p.seed : std::random_device{}()) {}

int32_t Sampler::sample(const std::vector<float>& logits_in, const std::vector<int32_t>& history) {
    std::vector<float> logits = logits_in;
    const int64_t n = (int64_t) logits.size();

    // logit bias, before the penalties (as the Qwen engine and llama.cpp apply it)
    for (const auto& [id, v] : p_.logit_bias)
        if (id >= 0 && id < n) logits[(size_t) id] = v <= -100 ? -INFINITY : logits[(size_t) id] + v;

    // penalties over the last penalty_last_n tokens (llama.cpp's penalties sampler)
    if (p_.penalty_repeat != 1.0f || p_.penalty_freq != 0.0f || p_.penalty_present != 0.0f) {
        std::unordered_map<int32_t, int> count;
        const size_t from = history.size() > (size_t) std::max(p_.penalty_last_n, 0)
                                ? history.size() - (size_t) std::max(p_.penalty_last_n, 0) : 0;
        for (size_t i = from; i < history.size(); ++i) ++count[history[i]];
        for (const auto& [tok, cnt] : count) {
            if (tok < 0 || tok >= n) continue;
            float& l = logits[(size_t) tok];
            if (l == -INFINITY) continue;                // a ban stays one (minus a -inf penalty it was NaN)
            l = l <= 0.0f ? l * p_.penalty_repeat : l / p_.penalty_repeat;
            l -= (float) cnt * p_.penalty_freq + (cnt > 0 ? p_.penalty_present : 0.0f);
            // a huge penalty overflows: the least or the most likely token then, never +-inf or NaN (an infinite
            // best logit gives NaN weights)
            l = std::isnan(l) ? -FLT_MAX : std::clamp(l, -FLT_MAX, FLT_MAX);
        }
    }

    const int32_t best = (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());
    if (p_.temperature <= 0.0f) return best;
    const double mx = logits[(size_t) best];
    auto by_logit = [&](int32_t a, int32_t b) { return logits[(size_t) a] > logits[(size_t) b]; };
    auto weight = [&](int32_t i, double temp) { return std::exp(((double) logits[(size_t) i] - mx) / temp); };

    // candidates, best first. `idx` stays empty while every token is still a candidate, in vocabulary order.
    std::vector<int32_t> idx;
    if (p_.top_k > 0 && p_.top_k < n) {
        idx.resize((size_t) n);
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + p_.top_k, idx.end(), by_logit);
        idx.resize((size_t) p_.top_k);
    }

    // top-p over the candidates' probabilities at temperature 1: the most likely ones, down to and including
    // the one that brings their sum to top_p
    if (p_.top_p < 1.0f) {
        if (!idx.empty()) {                              // top-k sorted them already
            double sum = 0;
            for (int32_t i : idx) sum += weight(i, 1.0);
            double acc = 0;
            size_t cut = idx.size();
            for (size_t i = 0; i < idx.size(); ++i) {
                acc += weight(idx[i], 1.0) / sum;
                if (acc >= p_.top_p) { cut = i + 1; break; }
            }
            idx.resize(std::max<size_t>(cut, 1));
        } else {
            // every token is a candidate. Bucket them by how far below the best logit they are, take whole
            // buckets while their sum stays under top_p, and sort only the bucket where it crosses.
            constexpr int B = 2048;
            constexpr double SPAN = 40.0;                // logits more than 40 below the best: the last bucket
            std::vector<double> w((size_t) n), mass(B, 0.0);
            std::vector<int32_t> bucket((size_t) n);
            double sum = 0;
            for (int64_t i = 0; i < n; ++i) {
                const double d = mx - (double) logits[(size_t) i];
                const int b = std::isfinite(d) ? (int) std::min<double>(B - 1, d * (B / SPAN)) : B - 1;
                sum += (w[(size_t) i] = std::exp(-d));
                mass[(size_t) b] += w[(size_t) i];
                bucket[(size_t) i] = b;
            }
            const double target = p_.top_p * sum;
            double before = 0;
            int cross = B - 1;
            for (int b = 0; b < B; ++b) {
                if (before + mass[(size_t) b] >= target) { cross = b; break; }
                before += mass[(size_t) b];
            }
            std::vector<int32_t> edge;
            for (int64_t i = 0; i < n; ++i) {
                if (bucket[(size_t) i] < cross) idx.push_back((int32_t) i);
                else if (bucket[(size_t) i] == cross) edge.push_back((int32_t) i);
            }
            std::sort(edge.begin(), edge.end(), by_logit);
            for (int32_t i : edge) {
                idx.push_back(i);
                if ((before += w[(size_t) i]) >= target) break;
            }
            if (idx.empty()) idx.push_back(best);
        }
    }

    // min-p: the candidates at least min_p times as likely as the best one
    if (p_.min_p > 0.0f) {
        if (idx.empty()) {
            for (int64_t i = 0; i < n; ++i)
                if (weight((int32_t) i, 1.0) >= p_.min_p) idx.push_back((int32_t) i);
        } else {
            std::vector<int32_t> kept;
            for (int32_t i : idx)
                if (weight(i, 1.0) >= p_.min_p) kept.push_back(i);   // the best token is always a candidate
            idx = std::move(kept);
        }
        if (idx.empty()) idx.push_back(best);
    }

    // the draw at the temperature
    const double temp = p_.temperature;
    if (idx.empty()) {                                   // every token: the distribution as it is
        std::vector<double> w((size_t) n);
        double total = 0;
        for (int64_t i = 0; i < n; ++i) total += (w[(size_t) i] = weight((int32_t) i, temp));
        double r = std::uniform_real_distribution<double>(0.0, total)(rng_);
        int32_t last = best;
        for (int64_t i = 0; i < n; ++i) {
            if (w[(size_t) i] <= 0) continue;
            last = (int32_t) i;
            if ((r -= w[(size_t) i]) < 0) return (int32_t) i;
        }
        return last;
    }
    std::vector<double> pr(idx.size());
    for (size_t i = 0; i < idx.size(); ++i) pr[i] = weight(idx[i], temp);
    std::discrete_distribution<size_t> dist(pr.begin(), pr.end());
    return idx[dist(rng_)];
}

}  // namespace strata::glm
