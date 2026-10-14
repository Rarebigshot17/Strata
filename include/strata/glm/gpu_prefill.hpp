// include/strata/glm/gpu_prefill.hpp - kernels for a chunk of T prompt tokens at once (src/glm/gpu_prefill.cu).
//
// Each one is the batched form of a decode kernel in gpu_kernels.hpp, computing the same values per token in the
// same order: row-major [T][...] activations, one row per token.
#pragma once

#include <cstdint>

namespace strata::glm::gpu {

// mHC, per token
void hc_weights_rows(const float* mixes, const float* scale, const float* base, float* pc, int T, int hc, int iters,
                     float eps, void* stream);
void hc_mix_rows(const float* R, const float* pc, float* x, int T, int hc, int n_embd, void* stream);
void hc_post_rows(const float* y, const float* R, const float* pc, float* R_out, int T, int hc, int n_embd,
                  void* stream);

// KDA
/// The causal conv over T tokens: the window state ((K-1) x n) holds the inputs before the chunk and is left
/// holding the chunk's last K-1 inputs.
void kda_conv_rows(const float* raw, float* state, const float* w, float* out, int T, int n, int K, void* stream);
/// kda_gate for T tokens of H x D.
void kda_gate_rows(float* g, const float* dt_b, const float* a, int T, int H, int D, bool has_lb, float lb,
                   void* stream);
/// The recurrence over T tokens, token by token: S[h] (D x D), q/k/v/g [T][H][D], beta [T][H] -> o [T][H][D].
void kda_scan(float* S, const float* q, const float* k, const float* v, const float* g, const float* beta, float* o,
              int T, int H, int D, void* stream);

// MLA
/// Rows t0 .. t0+T-1 of the f16 latent cache from kv [T][KL].
void store_f16_rows(const float* kv, uint16_t* cache, int64_t t0, int T, int KL, void* stream);
/// [rows][n] f32 -> f16 (the absorbed queries, as the decode path rounds them).
void to_f16(const float* x, uint16_t* y, int64_t n, void* stream);
/// Causal softmax over the dense scores S [T*H][n_ctx] (row t*H + h sees cells 0 .. t0 + t), written as f16
/// probabilities P [T*H][n_ctx] with the cells it does not see at 0.
void causal_softmax_f16(const float* S, uint16_t* P, int T, int H, int n_ctx, int64_t t0, void* stream);

// the k-pool indexer
/// Pools `n_pools` complete pools whose P members are consecutive rows of keys/gates [rows][ID] starting at row
/// `first`, into pooled + p0 * ID.
void kpool_rows(const float* keys, const float* gates, const float* ape, float* pooled, int64_t p0, int n_pools,
                int first, int P, int ID, void* stream);

// LayerNorm per row
void layer_norm_rows(const float* x, float* y, int rows, int n, float eps, const float* w, const float* b,
                     void* stream);

// routing
void route_rows(const float* logits, const float* bias, int T, int n_expert, int k, int gating, bool norm,
                float scale, int32_t* ids, float* weights, void* stream);

// experts
/// X [m][n] = src rows idx[0..m)
void gather_rows(const float* src, const int32_t* idx, float* X, int m, int n, void* stream);
/// O[slot[i]] = Y[i] for m rows of n (slot = token * k + position in its routing list)
void scatter_rows(const float* Y, const int32_t* slot, float* O, int m, int n, void* stream);
/// y[t] = sum_j w[t][j] * O[t][j] (j in routing order) + (shared ? shared[t] : 0)
void combine_experts(const float* O, const float* w, const float* shared, float* y, int T, int k, int n,
                     void* stream);

}  // namespace strata::glm::gpu
