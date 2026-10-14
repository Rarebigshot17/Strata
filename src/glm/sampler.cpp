// src/glm/sampler.cpp - picking the next token (sampler.hpp).
#include "strata/glm/sampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <unordered_map>

namespace strata::glm {

bool SamplingParams::set(const std::string& key, const std::string& value) {
    const double v = std::atof(value.c_str());
    if (key == "temperature") temperature = (float) v;
    else if (key == "top_p") top_p = (float) v;
    else if (key == "top_k") top_k = (int) v;
    else if (key == "min_p") min_p = (float) v;
    else if (key == "penalty_repeat") penalty_repeat = (float) v;
    else if (key == "penalty_freq") penalty_freq = (float) v;
    else if (key == "penalty_present") penalty_present = (float) v;
    else if (key == "penalty_last_n") penalty_last_n = (int) v;
    else if (key == "seed") {
        seed = (uint64_t) std::strtoull(value.c_str(), nullptr, 10);
        has_seed = true;
    } else return false;
    return true;
}

Sampler::Sampler(const SamplingParams& p) : p_(p), rng_(p.has_seed ? p.seed : std::random_device{}()) {}

int32_t Sampler::sample(const std::vector<float>& logits_in, const std::vector<int32_t>& history) {
    std::vector<float> logits = logits_in;
    const int64_t n = (int64_t) logits.size();

    // penalties over the last penalty_last_n tokens (llama.cpp's penalties sampler)
    if (p_.penalty_repeat != 1.0f || p_.penalty_freq != 0.0f || p_.penalty_present != 0.0f) {
        std::unordered_map<int32_t, int> count;
        const size_t from = history.size() > (size_t) std::max(p_.penalty_last_n, 0)
                                ? history.size() - (size_t) std::max(p_.penalty_last_n, 0) : 0;
        for (size_t i = from; i < history.size(); ++i) ++count[history[i]];
        for (const auto& [tok, cnt] : count) {
            if (tok < 0 || tok >= n) continue;
            float& l = logits[(size_t) tok];
            l = l <= 0.0f ? l * p_.penalty_repeat : l / p_.penalty_repeat;
            l -= (float) cnt * p_.penalty_freq + (cnt > 0 ? p_.penalty_present : 0.0f);
        }
    }

    if (p_.temperature <= 0.0f)
        return (int32_t) (std::max_element(logits.begin(), logits.end()) - logits.begin());

    // candidates, best first
    std::vector<int32_t> idx((size_t) n);
    std::iota(idx.begin(), idx.end(), 0);
    int64_t keep = n;
    if (p_.top_k > 0 && p_.top_k < n) keep = p_.top_k;
    std::partial_sort(idx.begin(), idx.begin() + keep, idx.end(),
                      [&](int32_t a, int32_t b) { return logits[(size_t) a] > logits[(size_t) b]; });
    idx.resize((size_t) keep);

    // probabilities of the kept candidates at temperature 1, for top-p and min-p
    auto probs_of = [&](const std::vector<int32_t>& cand, float temp) {
        std::vector<double> pr(cand.size());
        const double mx = logits[(size_t) cand[0]];
        double sum = 0;
        for (size_t i = 0; i < cand.size(); ++i) sum += (pr[i] = std::exp((logits[(size_t) cand[i]] - mx) / temp));
        for (auto& v : pr) v /= sum;
        return pr;
    };
    if (p_.top_p < 1.0f) {
        const auto pr = probs_of(idx, 1.0f);
        double acc = 0;
        size_t cut = idx.size();
        for (size_t i = 0; i < idx.size(); ++i) {
            acc += pr[i];
            if (acc >= p_.top_p) {
                cut = i + 1;
                break;
            }
        }
        idx.resize(std::max<size_t>(cut, 1));
    }
    if (p_.min_p > 0.0f) {
        const auto pr = probs_of(idx, 1.0f);
        size_t cut = 1;
        while (cut < idx.size() && pr[cut] >= p_.min_p * pr[0]) ++cut;
        idx.resize(cut);
    }
    const auto pr = probs_of(idx, p_.temperature);
    std::discrete_distribution<size_t> dist(pr.begin(), pr.end());
    return idx[dist(rng_)];
}

}  // namespace strata::glm
