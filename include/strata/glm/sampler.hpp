// include/strata/glm/sampler.hpp - picking the next token from the logits.
//
// The order is llama.cpp's sampler chain for these settings: penalties, then top-k, top-p, min-p, temperature,
// and a draw. Temperature 0 (or none given) is greedy.
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace strata::glm {

struct SamplingParams {
    float temperature = 0.0f;       ///< <= 0: greedy
    float top_p = 1.0f;
    int top_k = 0;                  ///< 0: off
    float min_p = 0.0f;
    float penalty_repeat = 1.0f;
    float penalty_freq = 0.0f;
    float penalty_present = 0.0f;
    int penalty_last_n = 64;
    uint64_t seed = 0;              ///< 0: random
    bool has_seed = false;

    /// Reads one `key=value` of the engine protocol; false when the key is not a sampling key.
    bool set(const std::string& key, const std::string& value);
};

class Sampler {
public:
    explicit Sampler(const SamplingParams& p);
    /// The next token. `history` holds the tokens the penalties look at (the prompt and what has been generated).
    int32_t sample(const std::vector<float>& logits, const std::vector<int32_t>& history);

private:
    SamplingParams p_;
    std::mt19937_64 rng_;
};

}  // namespace strata::glm
