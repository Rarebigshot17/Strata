// include/strata/glm/cpu_dot.hpp - CPU dot products faster than ggml-cpu's for some expert types, bitwise equal.
//
// IQ3_S x Q8_K: ggml's AVX2 kernel builds each vector of grid values from eight scalar loads; this one gathers
// them (vpgatherdd). The integer sums land in the same lanes and the float steps are ggml's, so every result is
// bitwise ggml's. On napster-ai's Xeon E5-2699 v4 (Broadwell) it reads 2.2 GB/s of weights on one core against
// ggml's 1.4, and 31.8 GB/s on 22 against 28.7 (measured).
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::glm {

/// ggml-cpu's vec_dot signature (nrc = 1).
using VecDotFn = void (*)(int n, float* s, size_t bs, const void* vx, size_t bx, const void* vy, size_t by, int nrc);

/// A faster dot product for weights of ggml type `type` against their vec_dot_type, or nullptr where there is none
/// (another type, or a CPU without AVX2, FMA and F16C).
VecDotFn fast_vec_dot(int type);

}  // namespace strata::glm
