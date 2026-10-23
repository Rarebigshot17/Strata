// include/strata/glm/gpu_linear.hpp - y = W x on the GPU for any weight type a GLM-5.3-Flash GGUF can hold.
//
// Quantized weights stay in their GGUF blocks and go through Strata's MMVQ kernels (native_mmvq.hpp, the
// llama.cpp CUDA contract: the activation is quantized to q8_1 first). F32, F16 and BF16 weights use a plain
// kernel. A type with no GPU kernel (Q2_K, IQ1_S, MXFP4) is converted to F16 when it is uploaded.
#pragma once

#include "strata/glm/model.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::glm::gpu {

struct DevMatrix {
    enum Path : int { None = 0, Mmvq, IqMmvq, F32, F16, BF16 };
    void* w = nullptr;          ///< device bytes, rows of `n_in`
    int type = 0;               ///< ggml type as stored on the device
    Path path = None;
    int64_t n_in = 0, n_out = 0;
    size_t row_bytes = 0;
    bool ok() const { return w != nullptr; }
    /// Rows [r0, r0 + n) as a matrix of their own (one head of a [n_in, n, H] tensor: r0 = h * n).
    DevMatrix rows(int64_t r0, int64_t n) const;
};

/// The kernel path for weights of `type` kept as they are (None when the type has no GPU kernel).
DevMatrix::Path path_for(int type);

/// Copies `t` (a matrix, or a 3-D tensor seen as ne1 * ne2 rows) to the current device. `bytes_out` adds the
/// device bytes used. `allow_convert` false turns a type without a GPU kernel into an error.
DevMatrix upload(const Tensor& t, size_t* bytes_out = nullptr, bool allow_convert = true);
void free_matrix(DevMatrix& m);

/// Bytes of q8_1 scratch for activations of up to `n_in` values.
size_t q8_scratch_bytes(int64_t n_in);

/// y (n_out) = W x (n_in). `q8` is scratch from q8_scratch_bytes(n_in).
void gemv(const DevMatrix& m, const float* x, float* y, void* q8, void* stream);
/// The same activation through two matrices of the same n_in and type (quantized once).
void gemv2(const DevMatrix& a, const DevMatrix& b, const float* x, float* ya, float* yb, void* q8, void* stream);
/// x quantized the way gemv quantizes it for `m` (nothing for the float paths), into `q8`
/// (q8_scratch_bytes(m.n_in)), for gemv_pre on any matrix that quantizes_alike(m, ...).
void quantize_for(const DevMatrix& m, const float* x, void* q8, void* stream);
/// gemv on an activation quantize_for already put in `q8` (the float paths read x): the same values as gemv.
void gemv_pre(const DevMatrix& m, const float* x, const void* q8, float* y, void* stream);
/// Whether a and b take the same quantized activation (same kernel path and n_in).
bool quantizes_alike(const DevMatrix& a, const DevMatrix& b);
/// `m` holds `n_heads` matrices of m.n_out / n_heads rows one after another; head h: y + h * rows = W_h (x + h *
/// m.n_in). The same values as gemv on m.rows(h * rows, rows) for each head, in one quantization and one launch
/// where the type has the kernel (native_mmvq_heads). `q8` is scratch from q8_scratch_bytes(n_heads * m.n_in).
void gemv_heads(const DevMatrix& m, int n_heads, const float* x, float* y, void* q8, void* stream);

/// A small tensor (a norm weight, a bias) to the device as f32.
float* upload_f32(const Tensor& t, size_t* bytes_out = nullptr);

}  // namespace strata::glm::gpu
