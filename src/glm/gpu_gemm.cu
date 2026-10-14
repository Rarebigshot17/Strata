// src/glm/gpu_gemm.cu - the prompt path's products (gpu_gemm.hpp).
#include "strata/glm/gpu_gemm.hpp"

#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml.h"

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace strata::glm::gpu {

namespace {

void ok(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("CUDA ") + what + ": " + cudaGetErrorString(e));
}
void ok(cublasStatus_t s, const char* what) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string("cuBLAS ") + what + " failed (" +
                                                             std::to_string((int) s) + ")");
}

// ggml's block_q6_K: ql[128], qh[64], scales[16] (int8), d (f16) = 210 bytes per 256 values
__global__ void k_dequant_q6_k(const uint8_t* __restrict__ src, __half* __restrict__ dst) {
    const int64_t b = blockIdx.x;
    const uint8_t* blk = src + b * 210;
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = (const int8_t*) (blk + 192);
    const float d = __half2float(*(const __half*) (blk + 208));
    __half* y = dst + b * 256;
    const int tid = threadIdx.x;          // 64 threads: half n (0/1) x l (0..31)
    const int n = tid / 32, l = tid % 32;
    const uint8_t* qlh = ql + 64 * n;
    const uint8_t* qhh = qh + 32 * n;
    const int8_t* sch = sc + 8 * n;
    const int is = l / 16;
    const int q1 = (int) ((qlh[l] & 0xF) | (((qhh[l] >> 0) & 3) << 4)) - 32;
    const int q2 = (int) ((qlh[l + 32] & 0xF) | (((qhh[l] >> 2) & 3) << 4)) - 32;
    const int q3 = (int) ((qlh[l] >> 4) | (((qhh[l] >> 4) & 3) << 4)) - 32;
    const int q4 = (int) ((qlh[l + 32] >> 4) | (((qhh[l] >> 6) & 3) << 4)) - 32;
    __half* yh = y + 128 * n;
    yh[l + 0] = __float2half(d * sch[is + 0] * q1);
    yh[l + 32] = __float2half(d * sch[is + 2] * q2);
    yh[l + 64] = __float2half(d * sch[is + 4] * q3);
    yh[l + 96] = __float2half(d * sch[is + 6] * q4);
}

__global__ void k_f32_to_f16(const float* __restrict__ x, __half* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half(x[i]);
}

__global__ void k_f32_to_bf16(const float* __restrict__ x, __nv_bfloat16* __restrict__ y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(x[i]);
}

unsigned grid(int64_t n, int t) { return (unsigned) ((n + t - 1) / t); }

}  // namespace

GemmCtx::GemmCtx(int device) : device_(device) {
    ok(cudaSetDevice(device), "cudaSetDevice");
    cublasHandle_t h = nullptr;
    ok(cublasCreate(&h), "cublasCreate");
    handle_ = h;
}

GemmCtx::~GemmCtx() {
    cudaSetDevice(device_);
    if (handle_) cublasDestroy((cublasHandle_t) handle_);
    if (wbuf_) cudaFree(wbuf_);
    if (xbuf_) cudaFree(xbuf_);
    if (q8_) cudaFree(q8_);
}

void GemmCtx::reserve(size_t elements, size_t act) {
    if (elements * 2 > wcap_) {
        if (wbuf_) cudaFree(wbuf_);
        wcap_ = elements * 2;
        ok(cudaMalloc(&wbuf_, wcap_), "cudaMalloc gemm weights");
    }
    if (act * 2 > xcap_) {
        if (xbuf_) cudaFree(xbuf_);
        xcap_ = act * 2;
        ok(cudaMalloc(&xbuf_, xcap_), "cudaMalloc gemm activations");
    }
    const size_t q8 = act / 32 * 36 + 256;
    if (q8 > q8cap_) {
        if (q8_) cudaFree(q8_);
        q8cap_ = q8;
        ok(cudaMalloc(&q8_, q8cap_), "cudaMalloc gemm q8");
    }
}

void* GemmCtx::weight_f16(const DevMatrix& w, void* stream) {
    const int64_t n = w.n_in * w.n_out;
    if (w.path == DevMatrix::F16 || w.path == DevMatrix::BF16) return w.w;
    reserve((size_t) n, 0);
    cudaStream_t s = (cudaStream_t) stream;
    if (w.path == DevMatrix::F32) {
        k_f32_to_f16<<<grid(n, 256), 256, 0, s>>>((const float*) w.w, (__half*) wbuf_, n);
    } else if (w.type == GGML_TYPE_Q6_K) {
        k_dequant_q6_k<<<(unsigned) (n / 256), 64, 0, s>>>((const uint8_t*) w.w, (__half*) wbuf_);
    } else if (strata::kernels::iq_supported(w.type) && n % 256 == 0) {
        strata::kernels::iq_dequant_f16(w.type, w.w, n, (uint16_t*) wbuf_, stream);
    } else {
        throw std::runtime_error("no f16 dequantizer for ggml type " + std::to_string(w.type));
    }
    ok(cudaGetLastError(), "dequant");
    return wbuf_;
}

void GemmCtx::gemm(const DevMatrix& w, const float* X, int T, float* Y, void* stream) {
    ok(cudaSetDevice(device_), "cudaSetDevice");
    if (T <= 8 && (w.path == DevMatrix::Mmvq || w.path == DevMatrix::IqMmvq)) {
        reserve(0, (size_t) w.n_in * T);
        if (w.path == DevMatrix::Mmvq) {
            strata::kernels::native_quantize_q8_1(X, q8_, (int) w.n_in, T, stream);
            strata::kernels::native_mmvq(w.type, w.w, q8_, Y, (int) w.n_in, (int) w.n_out, T, stream);
        } else {
            strata::kernels::quantize_q8_1_rows(X, T, w.n_in, q8_, stream);
            strata::kernels::iq_mmvq(w.type, w.w, q8_, Y, (int) w.n_in, (int) w.n_out, T, stream);
        }
        ok(cudaGetLastError(), "gemm mmvq");
        return;
    }
    cublasHandle_t h = (cublasHandle_t) handle_;
    ok(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    const float one = 1.0f, zero = 0.0f;
    const int m = (int) w.n_out, n = T, k = (int) w.n_in;
    if (w.path == DevMatrix::F32) {                   // small f32 matrices (mHC, norms): plain SGEMM
        ok(cublasSgemm(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, (const float*) w.w, k, X, k, &zero, Y, m), "Sgemm");
        return;
    }
    const void* A = weight_f16(w, stream);
    reserve(0, (size_t) k * T);
    const int64_t nx = (int64_t) k * T;
    const bool bf = w.path == DevMatrix::BF16;
    if (bf) k_f32_to_bf16<<<grid(nx, 256), 256, 0, (cudaStream_t) stream>>>(X, (__nv_bfloat16*) xbuf_, nx);
    else k_f32_to_f16<<<grid(nx, 256), 256, 0, (cudaStream_t) stream>>>(X, (__half*) xbuf_, nx);
    const cudaDataType_t dt = bf ? CUDA_R_16BF : CUDA_R_16F;
    ok(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, A, dt, k, xbuf_, dt, k, &zero, Y, CUDA_R_32F, m,
                    CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "GemmEx");
}

void GemmCtx::gemm_heads(const DevMatrix& w_all, int batch, const float* X, int ldx, int64_t x_stride, int T,
                         float* Y, int ldy, int64_t y_stride, void* stream) {
    ok(cudaSetDevice(device_), "cudaSetDevice");
    cublasHandle_t h = (cublasHandle_t) handle_;
    ok(cublasSetStream(h, (cudaStream_t) stream), "cublasSetStream");
    const float one = 1.0f, zero = 0.0f;
    const int m = (int) (w_all.n_out / batch), n = T, k = (int) w_all.n_in;
    const long long sa = (long long) m * k;
    if (w_all.path == DevMatrix::F32) {
        ok(cublasSgemmStridedBatched(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, (const float*) w_all.w, k, sa, X, ldx,
                                     x_stride, &zero, Y, ldy, y_stride, batch), "SgemmStridedBatched");
        return;
    }
    const void* A = weight_f16(w_all, stream);
    const int64_t nx = (int64_t) ldx * T;
    reserve(0, (size_t) nx);
    const bool bf = w_all.path == DevMatrix::BF16;
    if (bf) k_f32_to_bf16<<<grid(nx, 256), 256, 0, (cudaStream_t) stream>>>(X, (__nv_bfloat16*) xbuf_, nx);
    else k_f32_to_f16<<<grid(nx, 256), 256, 0, (cudaStream_t) stream>>>(X, (__half*) xbuf_, nx);
    const cudaDataType_t dt = bf ? CUDA_R_16BF : CUDA_R_16F;
    ok(cublasGemmStridedBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one, A, dt, k, sa, xbuf_, dt, ldx, x_stride,
                                  &zero, Y, CUDA_R_32F, ldy, y_stride, batch, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT), "GemmStridedBatchedEx");
}

}  // namespace strata::glm::gpu
