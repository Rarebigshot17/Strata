// src/glm/gpu_kernels.cu - GLM-5.3-Flash's own CUDA kernels (gpu_kernels.hpp).
//
// Written for clarity first: every kernel follows the loop order of the matching block in src/glm/cpu_ref.cpp,
// so the two can be compared element by element. Speed comes later, measured.
#include "strata/glm/gpu_kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace strata::glm::gpu {

namespace {

inline cudaStream_t S(void* s) { return (cudaStream_t) s; }

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "glm gpu kernel %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/// Sum over the block (blockDim.x a multiple of 32, at most 1024). Every thread gets the result.
__device__ float block_sum(float v) {
    __shared__ float part[32];
    __shared__ float total;
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) part[wid] = v;
    __syncthreads();
    if (wid == 0) {
        float t = lane < (int) (blockDim.x >> 5) ? part[lane] : 0.0f;
        t = warp_sum(t);
        if (lane == 0) total = t;
    }
    __syncthreads();
    const float r = total;
    __syncthreads();
    return r;
}

__device__ float block_max(float v) {
    __shared__ float part[32];
    __shared__ float total;
    v = warp_max(v);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) part[wid] = v;
    __syncthreads();
    if (wid == 0) {
        float t = lane < (int) (blockDim.x >> 5) ? part[lane] : -FLT_MAX;
        t = warp_max(t);
        if (lane == 0) total = t;
    }
    __syncthreads();
    const float r = total;
    __syncthreads();
    return r;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float siluf_(float x) { return x / (1.0f + expf(-x)); }

int blocks(int64_t n, int t) { return (int) ((n + t - 1) / t); }

// ---------------------------------------------------------------- norms and elementwise

__global__ void k_rms_norm(const float* x, float* y, int n, float eps, const float* w) {
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss);
    const float scale = rsqrtf(ss / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * scale * (w ? w[i] : 1.0f);
}

__global__ void k_layer_norm(const float* x, float* y, int n, float eps, const float* w, const float* b) {
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i];
    const float mean = block_sum(s) / (float) n;
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) v += (x[i] - mean) * (x[i] - mean);
    const float scale = rsqrtf(block_sum(v) / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = (x[i] - mean) * scale * w[i] + b[i];
}

__global__ void k_add(float* y, const float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

__global__ void k_scale(float* x, int n, float s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= s;
}

__global__ void k_axpy(float* y, const float* x, float a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

__global__ void k_swiglu(const float* g, const float* u, float* y, int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float gi = g[i], ui = u[i];
    if (limit > 1e-6f) {
        gi = fminf(gi, limit);
        ui = fminf(fmaxf(ui, -limit), limit);
    }
    y[i] = siluf_(gi) * ui;
}

__global__ void k_sigmoid(float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = sigmoidf_(x[i]);
}

// ---------------------------------------------------------------- mHC

// One thread does it all: hc is 4, and the order of the Sinkhorn sums is then exactly cpu_ref's.
__global__ void k_hc_weights(const float* mixes, const float* scale, const float* base, float* pc, int hc, int iters,
                             float eps) {
    if (threadIdx.x != 0) return;
    float* pre = pc;
    float* post = pc + hc;
    float* comb = pc + 2 * hc;
    for (int h = 0; h < hc; ++h) {
        pre[h] = sigmoidf_(mixes[h] * scale[0] + base[h]) + eps;
        post[h] = 2.0f * sigmoidf_(mixes[hc + h] * scale[1] + base[hc + h]);
    }
    for (int k = 0; k < hc * hc; ++k) comb[k] = mixes[2 * hc + k] * scale[2] + base[2 * hc + k];
    for (int src = 0; src < hc; ++src) {
        float* row = comb + src * hc;
        float mx = row[0];
        for (int d = 1; d < hc; ++d) mx = fmaxf(mx, row[d]);
        float sum = 0.0f;
        for (int d = 0; d < hc; ++d) {
            row[d] = expf(row[d] - mx);
            sum += row[d];
        }
        for (int d = 0; d < hc; ++d) row[d] = row[d] / sum + eps;
    }
    for (int it = 0; it < (iters > 1 ? iters : 1); ++it) {   // one column pass even when iters < 2, as ggml
        if (it > 0) {                                  // rows (over dst), then columns: iterations 2..iters
            for (int src = 0; src < hc; ++src) {
                float s = 0.0f;
                for (int d = 0; d < hc; ++d) s += comb[src * hc + d];
                s += eps;
                for (int d = 0; d < hc; ++d) comb[src * hc + d] /= s;
            }
        }
        for (int d = 0; d < hc; ++d) {                 // columns (over src)
            float s = 0.0f;
            for (int src = 0; src < hc; ++src) s += comb[src * hc + d];
            s += eps;
            for (int src = 0; src < hc; ++src) comb[src * hc + d] /= s;
        }
    }
}

__global__ void k_hc_mix(const float* R, const float* pc, float* x, int hc, int n) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    float acc = R[e] * pc[0];
    for (int h = 1; h < hc; ++h) acc += R[(size_t) h * n + e] * pc[h];
    x[e] = acc;
}

__global__ void k_hc_post(const float* y, const float* R, const float* pc, float* out, int hc, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= hc * n) return;
    const int d = idx / n, e = idx % n;
    const float* post = pc + hc;
    const float* comb = pc + 2 * hc;
    float acc = y[e] * post[d];
    for (int src = 0; src < hc; ++src) acc += R[(size_t) src * n + e] * comb[src * hc + d];
    out[idx] = acc;
}

__global__ void k_hc_mean(const float* R, float* x, int hc, int n) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    float acc = R[e];
    for (int h = 1; h < hc; ++h) acc += R[(size_t) h * n + e];
    x[e] = acc * (1.0f / (float) hc);
}

// ---------------------------------------------------------------- KDA

__global__ void k_kda_conv(const float* raw, float* state, const float* w, float* out, int n, int K) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= n) return;
    float acc = 0.0f;
    for (int k = 0; k < K - 1; ++k) acc += state[(size_t) k * n + ch] * w[(size_t) ch * K + k];
    const float r = raw[ch];
    acc += r * w[(size_t) ch * K + K - 1];
    out[ch] = siluf_(acc);
    for (int k = 0; k + 1 < K - 1; ++k) state[(size_t) k * n + ch] = state[(size_t) (k + 1) * n + ch];
    if (K > 1) state[(size_t) (K - 2) * n + ch] = r;
}

__global__ void k_kda_gate(float* g, const float* dt_b, const float* a, int H, int D, bool has_lb, float lb) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= H * D) return;
    const float ah = a[idx / D];
    const float gi = g[idx] + dt_b[idx];
    if (has_lb) g[idx] = lb * sigmoidf_(-(gi * ah));
    else g[idx] = (gi > 20.0f ? gi : log1pf(expf(gi))) * ah;
}

__global__ void k_l2_norm_heads(float* x, int D) {
    float* xh = x + (size_t) blockIdx.x * D;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += xh[i] * xh[i];
    ss = block_sum(ss);
    const float inv = 1.0f / sqrtf(ss + 1e-6f);
    for (int i = threadIdx.x; i < D; i += blockDim.x) xh[i] *= inv;
}

// One block per head, thread j owns value column j of S (S[i][j] at i * D + j: coalesced across j).
__global__ void k_kda_step(float* S, const float* q, const float* k, const float* v, const float* g,
                           const float* beta, float* o, int D, float scale) {
    extern __shared__ float sh[];
    float* eg = sh;
    float* ks = sh + D;
    float* qs = sh + 2 * D;
    const int h = blockIdx.x;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        eg[i] = expf(g[(size_t) h * D + i]);
        ks[i] = k[(size_t) h * D + i];
        qs[i] = q[(size_t) h * D + i];
    }
    __syncthreads();
    float* Sh = S + (size_t) h * D * D;
    const float bh = beta[h];
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float sk = 0.0f;
        for (int i = 0; i < D; ++i) {
            const float s = Sh[(size_t) i * D + j] * eg[i];
            Sh[(size_t) i * D + j] = s;
            sk += s * ks[i];
        }
        const float delta = (v[(size_t) h * D + j] - sk) * bh;
        float acc = 0.0f;
        for (int i = 0; i < D; ++i) {
            const float s = Sh[(size_t) i * D + j] + ks[i] * delta;
            Sh[(size_t) i * D + j] = s;
            acc += s * qs[i];
        }
        o[(size_t) h * D + j] = acc * scale;
    }
}

__global__ void k_kda_out_gate(const float* o, const float* o_norm, const float* g2, float* gated, int D, float eps) {
    const size_t base = (size_t) blockIdx.x * D;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) ss += o[base + i] * o[base + i];
    ss = block_sum(ss);
    const float scale = rsqrtf(ss / (float) D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        gated[base + i] = o[base + i] * scale * o_norm[i] * sigmoidf_(g2[base + i]);
}

// ---------------------------------------------------------------- MLA

__global__ void k_store_f16_row(const float* x, __half* cache, int64_t t, int KL) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < KL) cache[(size_t) t * KL + c] = __float2half_rn(x[c]);
}

__global__ void k_round_f16(float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __half2float(__float2half_rn(x[i]));
}

// warp per (cell, head): blockIdx.y = head, 4 warps per block over cells
__global__ void k_mla_scores(const float* q_abs, const __half* cache, const int32_t* sel, int n_sel, float* scores,
                             int KL, float scale) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int i = blockIdx.x * 4 + w, h = blockIdx.y;
    if (i >= n_sel) return;
    const float* q = q_abs + (size_t) h * KL;
    const __half* kc = cache + (size_t) sel[i] * KL;
    float acc = 0.0f;
    for (int c = lane; c < KL; c += 32) acc += q[c] * __half2float(kc[c]);
    acc = warp_sum(acc);
    if (lane == 0) scores[(size_t) h * n_sel + i] = acc * scale;
}

__global__ void k_mla_softmax(float* scores, int n_sel) {
    float* s = scores + (size_t) blockIdx.x * n_sel;
    float mx = -FLT_MAX;
    for (int i = threadIdx.x; i < n_sel; i += blockDim.x) mx = fmaxf(mx, s[i]);
    mx = block_max(mx);
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n_sel; i += blockDim.x) {
        const float e = expf(s[i] - mx);
        s[i] = e;
        sum += e;
    }
    sum = block_sum(sum);
    for (int i = threadIdx.x; i < n_sel; i += blockDim.x) s[i] = __half2float(__float2half_rn(s[i] / sum));
}

__global__ void k_mla_out(const float* p, const __half* cache, const int32_t* sel, int n_sel, float* out, int KL) {
    const int h = blockIdx.x, c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c >= KL) return;
    const float* ph = p + (size_t) h * n_sel;
    float acc = 0.0f;
    for (int i = 0; i < n_sel; ++i) acc += ph[i] * __half2float(cache[(size_t) sel[i] * KL + c]);
    out[(size_t) h * KL + c] = acc;
}

// ---------------------------------------------------------------- indexer

__global__ void k_kpool(const float* keys, const float* gates, const float* ape, float* pooled, int P, int ID) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= ID) return;
    float mx = -FLT_MAX;
    for (int m = 0; m < P; ++m) mx = fmaxf(mx, gates[(size_t) m * ID + d] + ape[(size_t) m * ID + d]);
    float sum = 0.0f;
    for (int m = 0; m < P; ++m) sum += expf(gates[(size_t) m * ID + d] + ape[(size_t) m * ID + d] - mx);
    float acc = 0.0f;
    for (int m = 0; m < P; ++m)
        acc += expf(gates[(size_t) m * ID + d] + ape[(size_t) m * ID + d] - mx) / sum * keys[(size_t) m * ID + d];
    pooled[d] = acc;
}

// warp per pool; the indexer query (IH x ID) in shared memory
__global__ void k_kpool_scores(const float* q, const float* pooled, const float* w, float* score, int n_pool, int IH,
                               int ID) {
    extern __shared__ float qs[];
    for (int i = threadIdx.x; i < IH * ID; i += blockDim.x) qs[i] = q[i];
    __syncthreads();
    const int lane = threadIdx.x & 31, wi = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int p = blockIdx.x * nw + wi;
    if (p >= n_pool) return;
    const float* pk = pooled + (size_t) p * ID;
    float sc = 0.0f;
    for (int h = 0; h < IH; ++h) {
        float dot = 0.0f;
        for (int d = lane; d < ID; d += 32) dot += qs[h * ID + d] * pk[d];
        dot = warp_sum(dot);
        sc += fmaxf(dot, 0.0f) * w[h];
    }
    if (lane == 0) score[p] = sc;
}

// ---------------------------------------------------------------- routing

__global__ void k_route(const float* logits, const float* bias, int n_expert, int k, int gating, bool norm,
                        float scale, int32_t* ids, float* weights) {
    __shared__ float probs[1024];
    __shared__ float selp[1024];
    __shared__ float red_v[32];
    __shared__ int red_i[32];
    const int t = threadIdx.x;
    float lg = t < n_expert ? logits[t] : -FLT_MAX;
    if (gating == 1) {
        const float mx = block_max(lg);
        const float e = t < n_expert ? expf(lg - mx) : 0.0f;
        const float sum = block_sum(e);
        if (t < n_expert) probs[t] = e / sum;
    } else if (t < n_expert) {
        probs[t] = sigmoidf_(lg);
    }
    __syncthreads();
    if (t < n_expert) selp[t] = probs[t] + bias[t];
    __syncthreads();
    float wsum = 0.0f;
    for (int r = 0; r < k; ++r) {
        // argmax of selp; ties go to the lower index (cpu_ref's partial_sort order)
        float v = t < n_expert ? selp[t] : -FLT_MAX;
        int idx = t;
        for (int o = 16; o > 0; o >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, v, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if ((t & 31) == 0) { red_v[t >> 5] = v; red_i[t >> 5] = idx; }
        __syncthreads();
        if (t == 0) {
            float bv = red_v[0];
            int bi = red_i[0];
            for (int w = 1; w < (int) (blockDim.x >> 5); ++w)
                if (red_v[w] > bv || (red_v[w] == bv && red_i[w] < bi)) { bv = red_v[w]; bi = red_i[w]; }
            ids[r] = bi;
            weights[r] = probs[bi];
            selp[bi] = -FLT_MAX;
        }
        __syncthreads();
    }
    if (t == 0) {
        for (int r = 0; r < k; ++r) wsum += weights[r];
        if (norm) {
            wsum = fmaxf(wsum, 6.103515625e-5f);
            for (int r = 0; r < k; ++r) weights[r] /= wsum;
        }
        if (scale != 0.0f && scale != 1.0f)
            for (int r = 0; r < k; ++r) weights[r] *= scale;
    }
}

}  // namespace

// ---------------------------------------------------------------- launchers

void rms_norm(const float* x, float* y, int n, float eps, const float* w, void* stream) {
    k_rms_norm<<<1, 1024, 0, S(stream)>>>(x, y, n, eps, w);
    check("rms_norm");
}

void rms_norm_rows(const float* x, float* y, int rows, int n, float eps, const float* w, void* stream) {
    k_rms_norm<<<rows, n >= 256 ? 256 : 128, 0, S(stream)>>>(x, y, n, eps, w);
    check("rms_norm_rows");
}

void layer_norm(const float* x, float* y, int n, float eps, const float* w, const float* b, void* stream) {
    k_layer_norm<<<1, 256, 0, S(stream)>>>(x, y, n, eps, w, b);
    check("layer_norm");
}

void add(float* y, const float* x, int n, void* stream) {
    k_add<<<blocks(n, 256), 256, 0, S(stream)>>>(y, x, n);
    check("add");
}

void scale(float* x, int n, float s, void* stream) {
    k_scale<<<blocks(n, 256), 256, 0, S(stream)>>>(x, n, s);
    check("scale");
}

void axpy(float* y, const float* x, float a, int n, void* stream) {
    k_axpy<<<blocks(n, 256), 256, 0, S(stream)>>>(y, x, a, n);
    check("axpy");
}

void swiglu(const float* g, const float* u, float* y, int n, float limit, void* stream) {
    k_swiglu<<<blocks(n, 256), 256, 0, S(stream)>>>(g, u, y, n, limit);
    check("swiglu");
}

void sigmoid_inplace(float* x, int n, void* stream) {
    k_sigmoid<<<blocks(n, 256), 256, 0, S(stream)>>>(x, n);
    check("sigmoid");
}

void hc_weights(const float* mixes, const float* scale, const float* base, float* pc, int hc, int iters, float eps,
                void* stream) {
    k_hc_weights<<<1, 32, 0, S(stream)>>>(mixes, scale, base, pc, hc, iters, eps);
    check("hc_weights");
}

void hc_mix(const float* R, const float* pc, float* x, int hc, int n_embd, void* stream) {
    k_hc_mix<<<blocks(n_embd, 256), 256, 0, S(stream)>>>(R, pc, x, hc, n_embd);
    check("hc_mix");
}

void hc_post(const float* y, const float* R, const float* pc, float* R_out, int hc, int n_embd, void* stream) {
    k_hc_post<<<blocks((int64_t) hc * n_embd, 256), 256, 0, S(stream)>>>(y, R, pc, R_out, hc, n_embd);
    check("hc_post");
}

void hc_mean(const float* R, float* x, int hc, int n_embd, void* stream) {
    k_hc_mean<<<blocks(n_embd, 256), 256, 0, S(stream)>>>(R, x, hc, n_embd);
    check("hc_mean");
}

void kda_conv(const float* raw, float* state, const float* w, float* out, int n, int K, void* stream) {
    k_kda_conv<<<blocks(n, 256), 256, 0, S(stream)>>>(raw, state, w, out, n, K);
    check("kda_conv");
}

void kda_gate(float* g, const float* dt_b, const float* a, int H, int D, bool has_lb, float lb, void* stream) {
    k_kda_gate<<<blocks((int64_t) H * D, 256), 256, 0, S(stream)>>>(g, dt_b, a, H, D, has_lb, lb);
    check("kda_gate");
}

void l2_norm_heads(float* x, int H, int D, void* stream) {
    k_l2_norm_heads<<<H, D >= 256 ? 256 : 128, 0, S(stream)>>>(x, D);
    check("l2_norm_heads");
}

void kda_step(float* S_, const float* q, const float* k, const float* v, const float* g, const float* beta, float* o,
              int H, int D, void* stream) {
    const int threads = D >= 128 ? 128 : 32 * ((D + 31) / 32);
    k_kda_step<<<H, threads, (size_t) 3 * D * sizeof(float), S(stream)>>>(S_, q, k, v, g, beta, o, D,
                                                                          1.0f / sqrtf((float) D));
    check("kda_step");
}

void kda_out_gate(const float* o, const float* o_norm, const float* g2, float* gated, int H, int D, float eps,
                  void* stream) {
    k_kda_out_gate<<<H, D >= 256 ? 256 : 128, 0, S(stream)>>>(o, o_norm, g2, gated, D, eps);
    check("kda_out_gate");
}

void store_f16_row(const float* x, uint16_t* cache, int64_t t, int KL, void* stream) {
    k_store_f16_row<<<blocks(KL, 256), 256, 0, S(stream)>>>(x, (__half*) cache, t, KL);
    check("store_f16_row");
}

void round_f16(float* x, int n, void* stream) {
    k_round_f16<<<blocks(n, 256), 256, 0, S(stream)>>>(x, n);
    check("round_f16");
}

void mla_attend(const float* q_abs, const uint16_t* cache, const int32_t* sel, int n_sel, float* scores, float* out,
                int H, int KL, float scale, void* stream) {
    const __half* c = (const __half*) cache;
    k_mla_scores<<<dim3((unsigned) blocks(n_sel, 4), (unsigned) H), 128, 0, S(stream)>>>(q_abs, c, sel, n_sel, scores,
                                                                                         KL, scale);
    k_mla_softmax<<<H, 256, 0, S(stream)>>>(scores, n_sel);
    k_mla_out<<<dim3((unsigned) H, (unsigned) blocks(KL, 256)), 256, 0, S(stream)>>>(scores, c, sel, n_sel, out, KL);
    check("mla_attend");
}

void kpool(const float* keys, const float* gates, const float* ape, float* pooled, int P, int ID, void* stream) {
    k_kpool<<<blocks(ID, 128), 128, 0, S(stream)>>>(keys, gates, ape, pooled, P, ID);
    check("kpool");
}

void kpool_scores(const float* q, const float* pooled, const float* w, float* score, int n_pool, int IH, int ID,
                  void* stream) {
    if (n_pool <= 0) return;
    k_kpool_scores<<<blocks(n_pool, 8), 256, (size_t) IH * ID * sizeof(float), S(stream)>>>(q, pooled, w, score,
                                                                                            n_pool, IH, ID);
    check("kpool_scores");
}

void route(const float* logits, const float* bias, int n_expert, int k, int gating, bool norm, float scale,
           int32_t* ids, float* weights, void* stream) {
    if (n_expert > 1024) {
        std::fprintf(stderr, "glm route: %d experts, at most 1024 supported\n", n_expert);
        std::exit(1);
    }
    k_route<<<1, 1024, 0, S(stream)>>>(logits, bias, n_expert, k, gating, norm, scale, ids, weights);
    check("route");
}

}  // namespace strata::glm::gpu
