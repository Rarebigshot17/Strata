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
#include <vector>

namespace strata::glm {

class CpuExperts {
public:
    CpuExperts(const Model& model, ThreadPool& pool);

    /// out (n_embd) = sum over the k experts `ids` of weights[i] * expert(x). x is the FFN-normed activation.
    void run(int64_t il, const float* x, const int32_t* ids, const float* weights, int k, float* out);

private:
    const Model& m_;
    ThreadPool& pool_;
    std::vector<uint8_t> xq_g_, xq_u_, hq_;
    std::vector<float> g_, u_;
    std::vector<Tensor> gate_, up_, down_;   ///< the selected experts' matrices
};

}  // namespace strata::glm
