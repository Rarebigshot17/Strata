// include/strata/glm/gpu_gemm.hpp - Y = X W^T for a chunk of tokens: the prompt path's products.
//
// Up to 8 tokens use the MMVQ kernels with that many activation columns (the decode path's arithmetic). More
// tokens dequantize the weight to f16 (bf16 kept as bf16) in a scratch buffer and multiply with cuBLAS, f32
// accumulation. X is [T][n_in] and Y is [T][n_out], row-major, f32.
#pragma once

#include "strata/glm/gpu_linear.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::glm::gpu {

/// Per-device state: the cuBLAS handle and the scratch buffers, grown on demand.
class GemmCtx {
public:
    explicit GemmCtx(int device);
    ~GemmCtx();
    GemmCtx(const GemmCtx&) = delete;
    GemmCtx& operator=(const GemmCtx&) = delete;

    /// Y [T][n_out] = X [T][n_in] W^T. `stream` must be the stream the caller uses on this device.
    void gemm(const DevMatrix& w, const float* X, int T, float* Y, void* stream);
    /// The same for `batch` matrices of equal shape and type at a fixed stride (the heads of a 3-D tensor):
    /// Y_b = X_b W_b^T, X_b at X + b * x_stride, Y_b at Y + b * y_stride (in floats). X rows have `ldx` floats.
    void gemm_heads(const DevMatrix& w_all, int batch, const float* X, int ldx, int64_t x_stride, int T, float* Y,
                    int ldy, int64_t y_stride, void* stream);

    /// The cuBLAS handle (cublasHandle_t), for products on other operands (the attention's).
    void* cublas() const { return handle_; }

    /// Reserve scratch for weights of up to `elements` values and activations of up to `act` values.
    void reserve(size_t elements, size_t act);

private:
    void* weight_f16(const DevMatrix& w, void* stream);   ///< w as f16 (or bf16) in scratch
    int device_;
    void* handle_ = nullptr;
    void* wbuf_ = nullptr;
    size_t wcap_ = 0;
    void* xbuf_ = nullptr;
    size_t xcap_ = 0;
    void* q8_ = nullptr;
    size_t q8cap_ = 0;
};

}  // namespace strata::glm::gpu
