// src/glm/gpu_prefill.cu - kernels for a chunk of prompt tokens (gpu_prefill.hpp).
#include "strata/glm/gpu_prefill.hpp"

#include <cub/device/device_scan.cuh>
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
        std::fprintf(stderr, "glm prefill kernel %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

unsigned grid(int64_t n, int t) { return (unsigned) ((n + t - 1) / t); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
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

// ---------------------------------------------------------------- mHC (one block per token; thread 0 as decode)

__global__ void k_hc_weights_rows(const float* mixes, const float* scale, const float* base, float* pc, int hc,
                                  int iters, float eps) {
    if (threadIdx.x != 0) return;
    const int mix = (2 + hc) * hc, pcd = 2 * hc + hc * hc;
    const float* mx_ = mixes + (size_t) blockIdx.x * mix;
    float* pre = pc + (size_t) blockIdx.x * pcd;
    float* post = pre + hc;
    float* comb = pre + 2 * hc;
    for (int h = 0; h < hc; ++h) {
        pre[h] = sigmoidf_(mx_[h] * scale[0] + base[h]) + eps;
        post[h] = 2.0f * sigmoidf_(mx_[hc + h] * scale[1] + base[hc + h]);
    }
    for (int k = 0; k < hc * hc; ++k) comb[k] = mx_[2 * hc + k] * scale[2] + base[2 * hc + k];
    for (int src = 0; src < hc; ++src) {
        float* row = comb + src * hc;
        float m = row[0];
        for (int d = 1; d < hc; ++d) m = fmaxf(m, row[d]);
        float sum = 0.0f;
        for (int d = 0; d < hc; ++d) {
            row[d] = expf(row[d] - m);
            sum += row[d];
        }
        for (int d = 0; d < hc; ++d) row[d] = row[d] / sum + eps;
    }
    for (int it = 0; it < (iters > 1 ? iters : 1); ++it) {
        if (it > 0) {
            for (int src = 0; src < hc; ++src) {
                float s = 0.0f;
                for (int d = 0; d < hc; ++d) s += comb[src * hc + d];
                s += eps;
                for (int d = 0; d < hc; ++d) comb[src * hc + d] /= s;
            }
        }
        for (int d = 0; d < hc; ++d) {
            float s = 0.0f;
            for (int src = 0; src < hc; ++src) s += comb[src * hc + d];
            s += eps;
            for (int src = 0; src < hc; ++src) comb[src * hc + d] /= s;
        }
    }
}

__global__ void k_hc_mix_rows(const float* R, const float* pc, float* x, int T, int hc, int n) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t) T * n) return;
    const int64_t t = idx / n, e = idx % n;
    const float* Rt = R + t * hc * n;
    const float* pre = pc + t * (2 * hc + hc * hc);
    float acc = Rt[e] * pre[0];
    for (int h = 1; h < hc; ++h) acc += Rt[(size_t) h * n + e] * pre[h];
    x[idx] = acc;
}

__global__ void k_hc_post_rows(const float* y, const float* R, const float* pc, float* out, int T, int hc, int n) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t) T * hc * n) return;
    const int64_t t = idx / ((int64_t) hc * n), rem = idx % ((int64_t) hc * n);
    const int d = (int) (rem / n), e = (int) (rem % n);
    const float* Rt = R + t * hc * n;
    const float* p = pc + t * (2 * hc + hc * hc);
    float acc = y[t * n + e] * p[hc + d];
    for (int src = 0; src < hc; ++src) acc += Rt[(size_t) src * n + e] * p[2 * hc + src * hc + d];
    out[idx] = acc;
}

// ---------------------------------------------------------------- KDA

// thread per channel, sequential over the chunk's tokens: the window slides exactly as decode's does
__global__ void k_kda_conv_rows(const float* raw, float* state, const float* w, float* out, int T, int n, int K) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= n) return;
    float win[16];                                   // K - 1 <= 15
    for (int k = 0; k < K - 1; ++k) win[k] = state[(size_t) k * n + ch];
    for (int t = 0; t < T; ++t) {
        const float r = raw[(size_t) t * n + ch];
        float acc = 0.0f;
        for (int k = 0; k < K - 1; ++k) acc += win[k] * w[(size_t) ch * K + k];
        acc += r * w[(size_t) ch * K + K - 1];
        out[(size_t) t * n + ch] = siluf_(acc);
        for (int k = 0; k + 1 < K - 1; ++k) win[k] = win[k + 1];
        if (K > 1) win[K - 2] = r;
    }
    for (int k = 0; k < K - 1; ++k) state[(size_t) k * n + ch] = win[k];
}

__global__ void k_kda_gate_rows(float* g, const float* dt_b, const float* a, int64_t total, int H, int D, bool has_lb,
                                float lb) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const int64_t inner = idx % ((int64_t) H * D);
    const float ah = a[inner / D];
    const float gi = g[idx] + dt_b[inner];
    if (has_lb) g[idx] = lb * sigmoidf_(-(gi * ah));
    else g[idx] = (gi > 20.0f ? gi : log1pf(expf(gi))) * ah;
}

// block per head; thread j owns column j of S; the tokens one after another, as decode
__global__ void k_kda_scan(float* S, const float* q, const float* k, const float* v, const float* g, const float* beta,
                           float* o, int T, int H, int D, float scale) {
    extern __shared__ float sh[];
    float* eg = sh;
    float* ks = sh + D;
    float* qs = sh + 2 * D;
    const int h = blockIdx.x;
    float* Sh = S + (size_t) h * D * D;
    for (int t = 0; t < T; ++t) {
        const size_t base = ((size_t) t * H + h) * D;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            eg[i] = expf(g[base + i]);
            ks[i] = k[base + i];
            qs[i] = q[base + i];
        }
        __syncthreads();
        const float bh = beta[(size_t) t * H + h];
        for (int j = threadIdx.x; j < D; j += blockDim.x) {
            float sk = 0.0f;
            for (int i = 0; i < D; ++i) {
                const float s = Sh[(size_t) i * D + j] * eg[i];
                Sh[(size_t) i * D + j] = s;
                sk += s * ks[i];
            }
            const float delta = (v[base + j] - sk) * bh;
            float acc = 0.0f;
            for (int i = 0; i < D; ++i) {
                const float s = Sh[(size_t) i * D + j] + ks[i] * delta;
                Sh[(size_t) i * D + j] = s;
                acc += s * qs[i];
            }
            o[base + j] = acc * scale;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------- MLA

__global__ void k_store_f16_rows(const float* kv, __half* cache, int64_t t0, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) cache[t0 + i] = __float2half_rn(kv[i]);
}

__global__ void k_to_f16(const float* x, __half* y, int64_t n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2half_rn(x[i]);
}

// one block per row (t, h): the decode path's softmax over cells 0 .. t0 + t, then f16 probabilities
__global__ void k_causal_softmax_f16(const float* Sc, __half* P, int H, int n_ctx, int64_t t0) {
    const int64_t row = blockIdx.x;
    const int64_t t = row / H;
    const int64_t last = t0 + t;                     // the last cell this query sees
    const float* s = Sc + row * n_ctx;
    __half* p = P + row * n_ctx;
    float mx = -FLT_MAX;
    for (int64_t i = threadIdx.x; i <= last; i += blockDim.x) mx = fmaxf(mx, s[i]);
    mx = block_max(mx);
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i <= last; i += blockDim.x) sum += expf(s[i] - mx);
    sum = block_sum(sum);
    for (int64_t i = threadIdx.x; i < n_ctx; i += blockDim.x)
        p[i] = __float2half_rn(i <= last ? expf(s[i] - mx) / sum : 0.0f);
}

// block per (query, cell): the query's selected latent rows side by side, zeros past its count
__global__ void k_gather_cells(const __half* cache, const int32_t* sel, const int32_t* n_sel, int max_sel, int KL,
                               __half* Kg) {
    const int j = blockIdx.x, b = blockIdx.y;
    __half* dst = Kg + ((size_t) b * max_sel + j) * KL;
    if (j < n_sel[b]) {
        const __half* src = cache + (size_t) sel[(size_t) b * max_sel + j] * KL;
        for (int c = threadIdx.x; c < KL; c += blockDim.x) dst[c] = src[c];
    } else {
        for (int c = threadIdx.x; c < KL; c += blockDim.x) dst[c] = __float2half_rn(0.0f);
    }
}

// one block per row (query b, head h): k_causal_softmax_f16 over the query's n_sel[b] gathered cells
__global__ void k_sparse_softmax_f16(const float* Sc, __half* P, const int32_t* n_sel, int H, int max_sel) {
    const int64_t row = blockIdx.x;
    const int n = n_sel[row / H];
    const float* s = Sc + row * max_sel;
    __half* p = P + row * max_sel;
    float mx = -FLT_MAX;
    for (int i = threadIdx.x; i < n; i += blockDim.x) mx = fmaxf(mx, s[i]);
    mx = block_max(mx);
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) sum += expf(s[i] - mx);
    sum = block_sum(sum);
    for (int i = threadIdx.x; i < max_sel; i += blockDim.x)
        p[i] = __float2half_rn(i < n ? expf(s[i] - mx) / sum : 0.0f);
}

// ---------------------------------------------------------------- indexer

// acc[t][p] (+)= max(head h's dot, 0) * w[t][h], the decode path's k_kpool_scores one head at a time
__global__ void k_pool_scores_accum(const float* dots, const float* w, int h, int IH, float* acc, int64_t T,
                                    int64_t NP, bool first) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * NP) return;
    const float v = fmaxf(dots[i], 0.0f) * w[(i / NP) * IH + h];
    acc[i] = first ? 0.0f + v : acc[i] + v;
}

/// A float's bits as an unsigned key that sorts the same way (-0 taken as +0, which compares equal to it).
__device__ __forceinline__ uint32_t order_key(float x) {
    if (x == 0.0f) x = 0.0f;
    const uint32_t u = __float_as_uint(x);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/// Over the block's threads in order: how many before this one have `f` set; the block's count in *total.
/// Every thread of the block must call it.
__device__ int block_excl_count(bool f, int* total) {
    __shared__ int wt[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = (int) (blockDim.x >> 5);
    const unsigned b = __ballot_sync(0xffffffffu, f);
    if (lane == 0) wt[wid] = __popc(b);
    __syncthreads();
    int before = 0, tot = 0;
    for (int w = 0; w < nw; ++w) {
        before += w < wid ? wt[w] : 0;
        tot += wt[w];
    }
    __syncthreads();
    *total = tot;
    return before + __popc(b & ((1u << lane) - 1u));
}

// one block per query row: the decode path's selection. Of the nv = (t+1)/P complete pools, the n_top with the
// highest scores, ties to the lower pool (std::stable_sort descending), found by a radix select on the scores'
// bits; every pool when nv <= n_top. Their cells in pool order, then the tail cells t, t-1, ... of the pool
// being filled.
__global__ void k_topk_pools(const float* acc, int64_t NP, int64_t t0, int P, int n_top, bool tail, int max_sel,
                             int32_t* sel, int32_t* n_sel) {
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_prefix, s_remaining;
    const int i = blockIdx.x;
    const int64_t t = t0 + i;
    const int nv = (int) ((t + 1) / P);
    const float* s = acc + (size_t) i * NP;
    int32_t* out = sel + (size_t) i * max_sel;
    const bool all = nv <= n_top, none = n_top <= 0;

    uint32_t prefix = 0, mask = 0, remaining = (uint32_t) (n_top > 0 ? n_top : 0);
    if (!all && !none) {
        for (int shift = 24; shift >= 0; shift -= 8) {
            for (int b = threadIdx.x; b < 256; b += blockDim.x) hist[b] = 0;
            __syncthreads();
            for (int j = threadIdx.x; j < nv; j += blockDim.x) {
                const uint32_t key = order_key(s[j]);
                if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 0xFFu], 1u);
            }
            __syncthreads();
            if (threadIdx.x == 0) {               // the digit holding the remaining-th largest, from the top
                uint32_t above = 0;
                int d = 255;
                for (; d > 0 && above + hist[d] < remaining; --d) above += hist[d];
                s_prefix = prefix | ((uint32_t) d << shift);
                s_remaining = remaining - above;
            }
            __syncthreads();
            prefix = s_prefix;
            remaining = s_remaining;
            mask |= 0xFFu << shift;
            __syncthreads();
        }
    }
    // prefix is now the n_top-th largest key; `remaining` of the pools equal to it are taken, the lowest first
    int eq_before = 0, taken = 0;
    for (int j0 = 0; j0 < nv; j0 += (int) blockDim.x) {
        const int j = j0 + (int) threadIdx.x;
        const bool in = j < nv && !none;
        const uint32_t key = in ? order_key(s[j]) : 0u;
        const bool gt = in && (all || key > prefix);
        const bool eq = in && !all && key == prefix;
        int n_eq = 0, n_take = 0;
        const int r = eq_before + block_excl_count(eq, &n_eq);
        const bool take = gt || (eq && r < (int) remaining);
        const int pos = taken + block_excl_count(take, &n_take);
        if (take)
            for (int m = 0; m < P; ++m) out[(size_t) pos * P + m] = j * P + m;
        eq_before += n_eq;
        taken += n_take;
    }
    int n = taken * P;
    const int tl = tail ? (int) ((t + 1) % P) : 0;
    for (int k = threadIdx.x; k < tl; k += blockDim.x) out[n + k] = (int32_t) (t - k);
    if (threadIdx.x == 0) n_sel[i] = n + tl;
}

// block per pool, thread per channel: decode's k_kpool on rows first + p * P ...
__global__ void k_kpool_rows(const float* keys, const float* gates, const float* ape, float* pooled, int64_t p0,
                             int first, int P, int ID) {
    const int p = blockIdx.x;
    const int d = threadIdx.x;
    if (d >= ID) return;
    const size_t r0 = (size_t) first + (size_t) p * P;
    float mx = -FLT_MAX;
    for (int m = 0; m < P; ++m) mx = fmaxf(mx, gates[(r0 + m) * ID + d] + ape[(size_t) m * ID + d]);
    float sum = 0.0f;
    for (int m = 0; m < P; ++m) sum += expf(gates[(r0 + m) * ID + d] + ape[(size_t) m * ID + d] - mx);
    float acc = 0.0f;
    for (int m = 0; m < P; ++m)
        acc += expf(gates[(r0 + m) * ID + d] + ape[(size_t) m * ID + d] - mx) / sum * keys[(r0 + m) * ID + d];
    pooled[(size_t) (p0 + p) * ID + d] = acc;
}

__global__ void k_layer_norm_rows(const float* x, float* y, int n, float eps, const float* w, const float* b) {
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i];
    const float mean = block_sum(s) / (float) n;
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) v += (xr[i] - mean) * (xr[i] - mean);
    const float scale = rsqrtf(block_sum(v) / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = (xr[i] - mean) * scale * w[i] + b[i];
}

// ---------------------------------------------------------------- routing: decode's k_route, one block per token

__global__ void k_route_rows(const float* logits, const float* bias, int n_expert, int k, int gating, bool norm,
                             float scale, int32_t* ids, float* weights) {
    __shared__ float probs[1024];
    __shared__ float selp[1024];
    __shared__ float red_v[32];
    __shared__ int red_i[32];
    const int t = threadIdx.x;
    const float* lg_row = logits + (size_t) blockIdx.x * n_expert;
    int32_t* id_row = ids + (size_t) blockIdx.x * k;
    float* w_row = weights + (size_t) blockIdx.x * k;
    const float lg = t < n_expert ? lg_row[t] : -FLT_MAX;
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
    for (int r = 0; r < k; ++r) {
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
            id_row[r] = bi;
            w_row[r] = probs[bi];
            selp[bi] = -FLT_MAX;
        }
        __syncthreads();
    }
    if (t == 0) {
        float wsum = 0.0f;
        for (int r = 0; r < k; ++r) wsum += w_row[r];
        if (norm) {
            wsum = fmaxf(wsum, 6.103515625e-5f);
            for (int r = 0; r < k; ++r) w_row[r] /= wsum;
        }
        if (scale != 0.0f && scale != 1.0f)
            for (int r = 0; r < k; ++r) w_row[r] *= scale;
    }
}

// ---------------------------------------------------------------- experts

__global__ void k_gather_rows(const float* src, const int32_t* idx, float* X, int64_t m, int n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * n) return;
    X[i] = src[(size_t) idx[i / n] * n + i % n];
}

__global__ void k_scatter_rows(const float* Y, const int32_t* slot, float* O, int64_t m, int n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * n) return;
    O[(size_t) slot[i / n] * n + i % n] = Y[i];
}

__global__ void k_scatter_add_w(const float* Y, const int32_t* tok, const float* w, float* y, int64_t m, int n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * n) return;
    const int64_t r = i / n;
    y[(size_t) tok[r] * n + i % n] += w[r] * Y[i];
}

__global__ void k_combine(const float* O, const float* w, const float* shared, float* y, int64_t T, int k, int n) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * n) return;
    const int64_t t = i / n, e = i % n;
    float acc = 0.0f;
    for (int j = 0; j < k; ++j) acc += w[t * k + j] * O[((size_t) t * k + j) * n + e];
    if (shared) acc += shared[i];
    y[i] = acc;
}

// ---------------------------------------------------------------- the latent cache in host RAM (--kv-in-ram)

__global__ void k_mark_cells(const int32_t* sel, const int32_t* n_sel, int max_sel, int32_t* flags) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y;
    if (j < n_sel[b]) flags[sel[(size_t) b * max_sel + j]] = 1;
}

__global__ void k_compact_cells(const int32_t* flags, const int32_t* pos, int64_t n, int32_t* U, int32_t* count) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (flags[i]) U[pos[i]] = (int32_t) i;
    if (i == n - 1) *count = pos[i] + flags[i];
}

__global__ void k_remap_cells(const int32_t* sel, const int32_t* n_sel, int max_sel, const int32_t* pos,
                              int32_t* local) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x, b = blockIdx.y;
    const size_t at = (size_t) b * max_sel + j;
    if (j < n_sel[b]) local[at] = pos[sel[at]];
}

// one block per row: rows idx[i] of `src` (host memory, read over PCIe) into dst row i, for i < *count (or n)
template <class V>
__global__ void k_gather_host_rows(const V* src, const int32_t* idx, const int32_t* count, int n, int row_v,
                                   V* dst) {
    const int i = blockIdx.x;
    if (i >= (count ? *count : n)) return;
    const V* s = src + (size_t) idx[i] * row_v;
    V* d = dst + (size_t) i * row_v;
    for (int c = threadIdx.x; c < row_v; c += blockDim.x) d[c] = s[c];
}

}  // namespace

void hc_weights_rows(const float* mixes, const float* scale, const float* base, float* pc, int T, int hc, int iters,
                     float eps, void* stream) {
    k_hc_weights_rows<<<T, 32, 0, S(stream)>>>(mixes, scale, base, pc, hc, iters, eps);
    check("hc_weights_rows");
}

void hc_mix_rows(const float* R, const float* pc, float* x, int T, int hc, int n_embd, void* stream) {
    k_hc_mix_rows<<<grid((int64_t) T * n_embd, 256), 256, 0, S(stream)>>>(R, pc, x, T, hc, n_embd);
    check("hc_mix_rows");
}

void hc_post_rows(const float* y, const float* R, const float* pc, float* R_out, int T, int hc, int n_embd,
                  void* stream) {
    k_hc_post_rows<<<grid((int64_t) T * hc * n_embd, 256), 256, 0, S(stream)>>>(y, R, pc, R_out, T, hc, n_embd);
    check("hc_post_rows");
}

void kda_conv_rows(const float* raw, float* state, const float* w, float* out, int T, int n, int K, void* stream) {
    if (K - 1 > 16) {
        std::fprintf(stderr, "kda_conv_rows: conv kernel %d, at most 17 supported\n", K);
        std::exit(1);
    }
    k_kda_conv_rows<<<grid(n, 128), 128, 0, S(stream)>>>(raw, state, w, out, T, n, K);
    check("kda_conv_rows");
}

void kda_gate_rows(float* g, const float* dt_b, const float* a, int T, int H, int D, bool has_lb, float lb,
                   void* stream) {
    const int64_t total = (int64_t) T * H * D;
    k_kda_gate_rows<<<grid(total, 256), 256, 0, S(stream)>>>(g, dt_b, a, total, H, D, has_lb, lb);
    check("kda_gate_rows");
}

void kda_scan(float* S_, const float* q, const float* k, const float* v, const float* g, const float* beta, float* o,
              int T, int H, int D, void* stream) {
    const int threads = D >= 128 ? 128 : 32 * ((D + 31) / 32);
    k_kda_scan<<<H, threads, (size_t) 3 * D * sizeof(float), S(stream)>>>(S_, q, k, v, g, beta, o, T, H, D,
                                                                         1.0f / sqrtf((float) D));
    check("kda_scan");
}

void store_f16_rows(const float* kv, uint16_t* cache, int64_t t0, int T, int KL, void* stream) {
    const int64_t n = (int64_t) T * KL;
    k_store_f16_rows<<<grid(n, 256), 256, 0, S(stream)>>>(kv, (__half*) cache, t0 * KL, n);
    check("store_f16_rows");
}

void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    k_to_f16<<<grid(n, 256), 256, 0, S(stream)>>>(x, (__half*) y, n);
    check("to_f16");
}

void causal_softmax_f16(const float* Sc, uint16_t* P, int T, int H, int n_ctx, int64_t t0, void* stream) {
    k_causal_softmax_f16<<<T * H, 256, 0, S(stream)>>>(Sc, (__half*) P, H, n_ctx, t0);
    check("causal_softmax_f16");
}

void gather_cells(const uint16_t* cache, const int32_t* sel, const int32_t* n_sel, int B, int max_sel, int KL,
                  uint16_t* Kg, void* stream) {
    k_gather_cells<<<dim3((unsigned) max_sel, (unsigned) B), 128, 0, S(stream)>>>((const __half*) cache, sel, n_sel,
                                                                                 max_sel, KL, (__half*) Kg);
    check("gather_cells");
}

void sparse_softmax_f16(const float* Sc, uint16_t* P, const int32_t* n_sel, int B, int H, int max_sel, void* stream) {
    k_sparse_softmax_f16<<<B * H, 256, 0, S(stream)>>>(Sc, (__half*) P, n_sel, H, max_sel);
    check("sparse_softmax_f16");
}

void pool_scores_accum(const float* dots, const float* w, int h, int IH, float* acc, int T, int64_t NP, bool first,
                       void* stream) {
    k_pool_scores_accum<<<grid((int64_t) T * NP, 256), 256, 0, S(stream)>>>(dots, w, h, IH, acc, T, NP, first);
    check("pool_scores_accum");
}

void topk_pools_rows(const float* acc, int64_t NP, int T, int64_t t0, int P, int n_top, bool tail, int max_sel,
                     int32_t* sel, int32_t* n_sel, void* stream) {
    k_topk_pools<<<T, 1024, 0, S(stream)>>>(acc, NP, t0, P, n_top, tail, max_sel, sel, n_sel);
    check("topk_pools_rows");
}

void kpool_rows(const float* keys, const float* gates, const float* ape, float* pooled, int64_t p0, int n_pools,
                int first, int P, int ID, void* stream) {
    if (n_pools <= 0) return;
    k_kpool_rows<<<n_pools, ID, 0, S(stream)>>>(keys, gates, ape, pooled, p0, first, P, ID);
    check("kpool_rows");
}

void layer_norm_rows(const float* x, float* y, int rows, int n, float eps, const float* w, const float* b,
                     void* stream) {
    k_layer_norm_rows<<<rows, 128, 0, S(stream)>>>(x, y, n, eps, w, b);
    check("layer_norm_rows");
}

void route_rows(const float* logits, const float* bias, int T, int n_expert, int k, int gating, bool norm,
                float scale, int32_t* ids, float* weights, void* stream) {
    k_route_rows<<<T, 1024, 0, S(stream)>>>(logits, bias, n_expert, k, gating, norm, scale, ids, weights);
    check("route_rows");
}

void gather_rows(const float* src, const int32_t* idx, float* X, int m, int n, void* stream) {
    k_gather_rows<<<grid((int64_t) m * n, 256), 256, 0, S(stream)>>>(src, idx, X, m, n);
    check("gather_rows");
}

void scatter_rows(const float* Y, const int32_t* slot, float* O, int m, int n, void* stream) {
    k_scatter_rows<<<grid((int64_t) m * n, 256), 256, 0, S(stream)>>>(Y, slot, O, m, n);
    check("scatter_rows");
}

void scatter_add_w_rows(const float* Y, const int32_t* tok, const float* w, float* y, int m, int n, void* stream) {
    if (m <= 0) return;
    k_scatter_add_w<<<grid((int64_t) m * n, 256), 256, 0, S(stream)>>>(Y, tok, w, y, m, n);
    check("scatter_add_w_rows");
}

void combine_experts(const float* O, const float* w, const float* shared, float* y, int T, int k, int n,
                     void* stream) {
    k_combine<<<grid((int64_t) T * n, 256), 256, 0, S(stream)>>>(O, w, shared, y, T, k, n);
    check("combine_experts");
}

size_t cell_union_scratch(int64_t n_cells) {
    size_t bytes = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, bytes, (const int32_t*) nullptr, (int32_t*) nullptr, (int) n_cells);
    return bytes;
}

void cell_union(const int32_t* sel, const int32_t* n_sel, int B, int max_sel, int64_t n_cells, int32_t* flags,
                int32_t* pos, void* tmp, size_t tmp_bytes, int32_t* U, int32_t* count, int32_t* local, void* stream) {
    cudaMemsetAsync(flags, 0, (size_t) n_cells * sizeof(int32_t), S(stream));
    const dim3 g(grid(max_sel, 128), (unsigned) B);
    k_mark_cells<<<g, 128, 0, S(stream)>>>(sel, n_sel, max_sel, flags);
    cub::DeviceScan::ExclusiveSum(tmp, tmp_bytes, flags, pos, (int) n_cells, S(stream));
    k_compact_cells<<<grid(n_cells, 256), 256, 0, S(stream)>>>(flags, pos, n_cells, U, count);
    k_remap_cells<<<g, 128, 0, S(stream)>>>(sel, n_sel, max_sel, pos, local);
    check("cell_union");
}

void gather_host_rows(const uint16_t* src, const int32_t* idx, const int32_t* count, int n, int KL, uint16_t* dst,
                      void* stream) {
    if (n <= 0) return;
    if ((KL * 2) % 16 == 0)
        k_gather_host_rows<int4><<<(unsigned) n, 64, 0, S(stream)>>>((const int4*) src, idx, count, n, KL * 2 / 16,
                                                                      (int4*) dst);
    else
        k_gather_host_rows<uint16_t><<<(unsigned) n, 64, 0, S(stream)>>>(src, idx, count, n, KL, dst);
    check("gather_host_rows");
}

}  // namespace strata::glm::gpu
