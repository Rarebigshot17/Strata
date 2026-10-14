// include/strata/glm/gpu_kernels.hpp - GLM-5.3-Flash's own CUDA kernels (src/glm/gpu_kernels.cu).
//
// One token at a time, sizes from the model's Config at run time. Each kernel mirrors one block of
// src/glm/cpu_ref.cpp, which is the reference it is tested against (tools/glm53_gpu_check). All pointers are
// device pointers on the current device; every call is enqueued on `stream` and does not synchronize.
#pragma once

#include <cstdint>

namespace strata::glm::gpu {

// ---- norms and elementwise ----
/// y = x / sqrt(mean(x^2) + eps) * (w ? w : 1); in place allowed. One block.
void rms_norm(const float* x, float* y, int n, float eps, const float* w, void* stream);
/// The same over `rows` independent rows of `n` (the KDA heads).
void rms_norm_rows(const float* x, float* y, int rows, int n, float eps, const float* w, void* stream);
/// LayerNorm with weight and bias. One block.
void layer_norm(const float* x, float* y, int n, float eps, const float* w, const float* b, void* stream);
void add(float* y, const float* x, int n, void* stream);
void scale(float* x, int n, float s, void* stream);
/// y[i] = silu(min(g, L)) * clamp(u, -L, L), or silu(g) * u when L <= 1e-6 (ggml_swiglu_clamp / swiglu).
void swiglu(const float* g, const float* u, float* y, int n, float limit, void* stream);

// ---- mHC (build_hc_pre / build_hc_post) ----
/// From the hc_fn product `mixes` ((2 + hc) * hc floats): pre (hc), post (hc) and the Sinkhorn-normalized comb
/// (hc * hc, comb[src * hc + dst]) into `pc` = [pre | post | comb]. One block.
void hc_weights(const float* mixes, const float* scale, const float* base, float* pc, int hc, int iters, float eps,
                void* stream);
/// x[e] = sum_h pre[h] * R[h][e]
void hc_mix(const float* R, const float* pc, float* x, int hc, int n_embd, void* stream);
/// R_out[d][e] = y[e] * post[d] + sum_src R[src][e] * comb[src][d]   (R_out must not alias R)
void hc_post(const float* y, const float* R, const float* pc, float* R_out, int hc, int n_embd, void* stream);
/// x[e] = mean over the hc streams of R
void hc_mean(const float* R, float* x, int hc, int n_embd, void* stream);

// ---- KDA (build_kda_layer) ----
/// Causal conv over the last K inputs plus SiLU, for one of q/k/v: out = silu(sum_k window[k] * w[ch][k]);
/// the window state ((K-1) x n rows, oldest first) moves by one row.
void kda_conv(const float* raw, float* state, const float* w, float* out, int n, int K, void* stream);
/// g = f_b(f_a(x)) + dt_b, then lb * sigmoid(-g * a[h]) (has_lb) or softplus(g) * a[h]; in place, H x D.
void kda_gate(float* g, const float* dt_b, const float* a, int H, int D, bool has_lb, float lb, void* stream);
/// L2 norm per head, eps 1e-6 (build_gdn_l2_norm), in place, H x D.
void l2_norm_heads(float* x, int H, int D, void* stream);
void sigmoid_inplace(float* x, int n, void* stream);
/// The recurrence (ggml's fused gated_delta_net, KDA form) for one token: S[h][i][j] with i the key channel.
/// S[i][:] *= exp(g[i]); delta = beta (v - S^T k); S += k delta^T; o = S^T q / sqrt(D).
void kda_step(float* S, const float* q, const float* k, const float* v, const float* g, const float* beta, float* o,
              int H, int D, void* stream);
/// gated = rms_norm(o_h) * o_norm * sigmoid(g2), per head.
void kda_out_gate(const float* o, const float* o_norm, const float* g2, float* gated, int H, int D, float eps,
                  void* stream);

// ---- MLA (build_dsa_layer) ----
/// Stores the latent of position `t` in the f16 cache (row t of `cache`, KL halves per row).
void store_f16_row(const float* x, uint16_t* cache, int64_t t, int KL, void* stream);
/// Rounds a vector to f16 precision in place (the f16 K cache's activation, as in cpu_ref).
void round_f16(float* x, int n, void* stream);
/// Attention of H absorbed queries (H x KL) over the cells `sel[0..n_sel)` of the latent cache:
/// p = softmax(q . K[cell] * scale) (rounded to f16), out[h] = sum p * K[cell]. `scores` is H x n_sel scratch.
void mla_attend(const float* q_abs, const uint16_t* cache, const int32_t* sel, int n_sel, float* scores, float* out,
                int H, int KL, float scale, void* stream);

// ---- the k-pool indexer (build_kpool_select) ----
/// Pools `P` member keys/gates (P x ID each) into one key: per channel d, softmax over members of
/// (gate[m][d] + ape[m][d]), weighted sum of key[m][d].
void kpool(const float* keys, const float* gates, const float* ape, float* pooled, int P, int ID, void* stream);
/// score[p] = sum_h w[h] * relu(q_h . pooled[p]) for p < n_pool.
void kpool_scores(const float* q, const float* pooled, const float* w, float* score, int n_pool, int IH, int ID,
                  void* stream);

// ---- routing (build_moe_ffn) ----
/// probs = sigmoid(logits) (gating 2) or softmax (gating 1); top-k of probs + bias; weights = probs of the
/// selected, normalized (norm) and scaled (scale not 0 or 1). One block; n_expert <= 1024.
void route(const float* logits, const float* bias, int n_expert, int k, int gating, bool norm, float scale,
           int32_t* ids, float* weights, void* stream);

}  // namespace strata::glm::gpu
