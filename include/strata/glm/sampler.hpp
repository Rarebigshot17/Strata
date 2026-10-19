// include/strata/glm/sampler.hpp - picking the next token from the logits.
//
// The order is llama.cpp's sampler chain for these settings: logit bias, penalties, then top-k, top-p, min-p,
// temperature, and a draw. Temperature 0 (or none given) is greedy.
//
// GLM's vocabulary is 154,880 tokens, so nothing here sorts all of it: top-k sorts k, top-p sorts a growing
// prefix until it holds top_p of the probability, min-p is a threshold on the logits, and a plain draw walks the
// distribution once.
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <utility>
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
    uint64_t seed = 0;              ///< used when has_seed (any value, 0 included); otherwise a random seed
    bool has_seed = false;
    /// `drawn=`: the seed's draws a reply's earlier passes made (a draw a sampled token), which this one skips: a
    /// reply the server makes in several passes is then drawn as in one.
    uint64_t drawn = 0;
    /// `logit_bias=id:value,...`: added to the logits before the penalties; -100 bans the token.
    std::vector<std::pair<int32_t, float>> logit_bias;
    std::string bias_error;         ///< why the logit_bias value was refused (empty: fine)

    /// Reads one `key=value` of the engine protocol; false when the key is not a sampling key.
    bool set(const std::string& key, const std::string& value);
    /// Why these settings cannot be used with a vocabulary of `n_vocab` tokens; empty when they can.
    std::string check(int64_t n_vocab) const;
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
