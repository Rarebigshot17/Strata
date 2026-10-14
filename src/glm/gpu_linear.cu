// src/glm/gpu_linear.cu - y = W x for every weight type (gpu_linear.hpp).
#include "strata/glm/gpu_linear.hpp"

#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::glm::gpu {

namespace {

void cuda_ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) { return __bfloat162float(v); }

// warp per row
template <class T>
__global__ void k_gemv(const T* __restrict__ w, const float* __restrict__ x, float* __restrict__ y, int64_t n_in,
                       int64_t n_out) {
    const int lane = threadIdx.x & 31;
    const int64_t r = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (r >= n_out) return;
    const T* row = w + r * n_in;
    float acc = 0.0f;
    for (int64_t i = lane; i < n_in; i += 32) acc += to_f(row[i]) * x[i];
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[r] = acc;
}

template <class T>
void launch_gemv(const void* w, const float* x, float* y, int64_t n_in, int64_t n_out, void* stream) {
    const int rows_per_block = 8;
    k_gemv<T><<<(unsigned) ((n_out + rows_per_block - 1) / rows_per_block), 32 * rows_per_block, 0,
                (cudaStream_t) stream>>>((const T*) w, x, y, n_in, n_out);
}

}  // namespace

DevMatrix DevMatrix::rows(int64_t r0, int64_t n) const {
    DevMatrix m = *this;
    m.w = static_cast<uint8_t*>(w) + (size_t) r0 * row_bytes;
    m.n_out = n;
    return m;
}

DevMatrix upload(const Tensor& t, size_t* bytes_out, bool allow_convert) {
    DevMatrix m;
    m.n_in = t.ne[0];
    m.n_out = t.rows();
    m.type = (int) t.type;
    const void* src = t.data;
    std::vector<uint16_t> converted;
    if (t.type == GGML_TYPE_F32) {
        m.path = DevMatrix::F32;
    } else if (t.type == GGML_TYPE_F16) {
        m.path = DevMatrix::F16;
    } else if (t.type == GGML_TYPE_BF16) {
        m.path = DevMatrix::BF16;
    } else if (strata::kernels::native_mmvq_supported((int) t.type)) {
        m.path = DevMatrix::Mmvq;
    } else if (strata::kernels::iq_supported((int) t.type)) {
        m.path = DevMatrix::IqMmvq;
    } else {
        if (!allow_convert) throw std::runtime_error("no GPU kernel for the type of " + t.name);
        const auto* tr = ggml_get_type_traits((ggml_type) t.type);
        if (!tr || !tr->to_float) throw std::runtime_error("cannot convert the type of " + t.name);
        converted.resize((size_t) t.elements());
        std::vector<float> row((size_t) m.n_in);
        for (int64_t r = 0; r < m.n_out; ++r) {
            tr->to_float(t.row(r), row.data(), m.n_in);
            for (int64_t i = 0; i < m.n_in; ++i) converted[(size_t) (r * m.n_in + i)] = ggml_fp32_to_fp16(row[(size_t) i]);
        }
        src = converted.data();
        m.type = GGML_TYPE_F16;
        m.path = DevMatrix::F16;
    }
    m.row_bytes = ggml_row_size((ggml_type) m.type, m.n_in);
    const size_t bytes = m.row_bytes * (size_t) m.n_out;
    cuda_ok(cudaMalloc(&m.w, bytes), ("cudaMalloc for " + t.name).c_str());
    cuda_ok(cudaMemcpy(m.w, src, bytes, cudaMemcpyHostToDevice), ("upload of " + t.name).c_str());
    if (bytes_out) *bytes_out += bytes;
    return m;
}

void free_matrix(DevMatrix& m) {
    if (m.w) cudaFree(m.w);
    m = DevMatrix{};
}

size_t q8_scratch_bytes(int64_t n_in) { return (size_t) ((n_in + 255) / 256 * 256 / 32) * 36 + 256; }

void gemv(const DevMatrix& m, const float* x, float* y, void* q8, void* stream) {
    switch (m.path) {
        case DevMatrix::F32: launch_gemv<float>(m.w, x, y, m.n_in, m.n_out, stream); break;
        case DevMatrix::F16: launch_gemv<__half>(m.w, x, y, m.n_in, m.n_out, stream); break;
        case DevMatrix::BF16: launch_gemv<__nv_bfloat16>(m.w, x, y, m.n_in, m.n_out, stream); break;
        case DevMatrix::Mmvq:
            strata::kernels::native_quantize_q8_1(x, q8, (int) m.n_in, 1, stream);
            strata::kernels::native_mmvq(m.type, m.w, q8, y, (int) m.n_in, (int) m.n_out, 1, stream);
            break;
        case DevMatrix::IqMmvq:
            strata::kernels::quantize_q8_1_rows(x, 1, m.n_in, q8, stream);
            strata::kernels::iq_mmvq(m.type, m.w, q8, y, (int) m.n_in, (int) m.n_out, 1, stream);
            break;
        default: throw std::runtime_error("gemv on an empty matrix");
    }
    cuda_ok(cudaGetLastError(), "gemv");
}

void gemv2(const DevMatrix& a, const DevMatrix& b, const float* x, float* ya, float* yb, void* q8, void* stream) {
    if (a.path == DevMatrix::Mmvq && b.path == DevMatrix::Mmvq && a.n_in == b.n_in) {
        strata::kernels::native_quantize_q8_1(x, q8, (int) a.n_in, 1, stream);
        strata::kernels::native_mmvq(a.type, a.w, q8, ya, (int) a.n_in, (int) a.n_out, 1, stream);
        strata::kernels::native_mmvq(b.type, b.w, q8, yb, (int) b.n_in, (int) b.n_out, 1, stream);
        cuda_ok(cudaGetLastError(), "gemv2");
        return;
    }
    gemv(a, x, ya, q8, stream);
    gemv(b, x, yb, q8, stream);
}

float* upload_f32(const Tensor& t, size_t* bytes_out) {
    std::vector<float> host((size_t) t.elements());
    if (t.type == GGML_TYPE_F32) {
        std::memcpy(host.data(), t.data, host.size() * sizeof(float));
    } else {
        const auto* tr = ggml_get_type_traits((ggml_type) t.type);
        if (!tr || !tr->to_float) throw std::runtime_error("cannot dequantize " + t.name);
        for (int64_t r = 0; r < t.rows(); ++r) tr->to_float(t.row(r), host.data() + r * t.ne[0], t.ne[0]);
    }
    float* d = nullptr;
    cuda_ok(cudaMalloc(&d, host.size() * sizeof(float)), ("cudaMalloc for " + t.name).c_str());
    cuda_ok(cudaMemcpy(d, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice),
            ("upload of " + t.name).c_str());
    if (bytes_out) *bytes_out += host.size() * sizeof(float);
    return d;
}

}  // namespace strata::glm::gpu
