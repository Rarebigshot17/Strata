// src/kernels/cuda/pdl.cu - the switches of pdl.hpp (programmatic dependent launch).
//
// PDL was for the Qwen engine's captured verify window. The GLM engine records no such window, so pdl_scope()
// stays false and every launch_pdl() in the kernels it uses (native_mmvq.cu) is an ordinary launch.
#include "strata/kernels/pdl.hpp"

namespace strata::kernels {

bool& pdl_scope() {
    static thread_local bool on = false;
    return on;
}

bool pdl_supported() { return false; }

bool pdl_launch_ok(const void*, cudaStream_t) { return false; }

}  // namespace strata::kernels
