// include/strata/glm/cpu_experts.hpp - the routed experts of one layer on the CPU, for one token.
//
// Strata's split: the experts are the part of the model that does not fit in VRAM, so they stay in RAM (the mapped
// GGUF) and the CPU computes the ones the VRAM cache does not hold. Each product goes through ggml-cpu's own dot
// product with the activation quantized to the weight's vec_dot_type, as in cpu_ref.cpp, so the result is the
// reference's to float rounding. All of a token's experts are spread over the threads together.
#pragma once

#include "strata/glm/cpu_ref.hpp"
#include "strata/glm/model.hpp"

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace strata::glm {

class CpuExperts {
public:
    CpuExperts(const Model& model, ThreadPool& pool);

    /// out (n_embd) = sum over the k experts `ids` of weights[i] * expert(x). x is the FFN-normed activation.
    /// `side` runs on the calling thread while the others start on the experts (ThreadPool::parallel_for).
    void run(int64_t il, const float* x, const int32_t* ids, const float* weights, int k, float* out,
             const std::function<void()>& side = nullptr);
    /// run() for `n` tokens at once: token t's experts ids[t][0..k[t]) with weights w[t], its activation x[t], its sum
    /// into out[t] (zeros when k[t] is 0). An expert two tokens share has its rows read from RAM once for both, and
    /// each token's sum is bitwise run()'s on it alone (the same products, summed in its own routing order).
    void run_multi(int64_t il, int n, const float* const* x, const int32_t* const* ids, const float* const* w,
                   const int* k, float* const* out, const std::function<void()>& side = nullptr);

private:
    const Model& m_;
    ThreadPool& pool_;
    std::vector<uint8_t> xq_g_, xq_u_, hq_;
    std::vector<float> g_, u_;
    std::vector<Tensor> gate_, up_, down_;   ///< the selected experts' matrices
    // run_multi: per token its quantized activations, and the distinct experts with the tokens (and their routing
    // slots) that use each
    std::vector<std::vector<uint8_t>> mxq_g_, mxq_u_;
    std::vector<int32_t> uniq_;
    std::vector<std::vector<std::pair<int, int>>> users_;
};

}  // namespace strata::glm
