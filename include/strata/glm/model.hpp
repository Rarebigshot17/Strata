// include/strata/glm/model.hpp - GLM-5.3-Flash (`glm5-next`): the configuration and the tensors of one GGUF.
//
// Everything here is read from the file. Nothing about the model's size is compiled in: the tiny random models
// the tests build and the real 320B model go through the same code. The semantics follow llama.cpp's
// `src/models/glm5-next.cpp` (pinned in tools/glm53_ref/build.sh), which is the reference this engine is
// checked against.
#pragma once

#include "strata/artifact/gguf_reader.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::glm {

/// One tensor of the GGUF: its bytes (mapped from the file), its ggml type and its shape in GGUF order
/// (`ne[0]` varies fastest; a matrix `[ne0, ne1]` has `ne1` rows of `ne0` elements).
struct Tensor {
    const void* data = nullptr;
    uint32_t type = 0;
    int64_t ne[4] = {1, 1, 1, 1};
    std::string name;

    bool ok() const { return data != nullptr; }
    int64_t rows() const { return ne[1] * ne[2] * ne[3]; }
    int64_t elements() const { return ne[0] * ne[1] * ne[2] * ne[3]; }
    /// Bytes of one row of `ne[0]` elements.
    size_t row_bytes() const;
    /// Pointer to row `r` (counting over ne1, ne2, ne3).
    const uint8_t* row(int64_t r) const { return static_cast<const uint8_t*>(data) + (size_t) r * row_bytes(); }
    /// The 2-D matrix `i2` of a 3-D tensor (one expert, or one head).
    Tensor slice2(int64_t i2) const;
};

struct Config {
    std::string arch;                 ///< "glm5-next" (llama.cpp master) or "glm5next" (some converters)
    int64_t n_layer_all = 0;          ///< block_count: the trunk plus the MTP layers
    int64_t n_layer = 0;              ///< the trunk
    int64_t n_layer_nextn = 0;
    int64_t n_embd = 0;
    int64_t n_vocab = 0;
    int64_t n_ctx_train = 0;
    float rms_eps = 1e-5f;
    float norm_eps = 1e-5f;           ///< LayerNorm (the indexer's key norm)

    // per layer
    std::vector<int64_t> n_head;      ///< query heads: KDA heads on a KDA layer, MLA heads on an MLA layer
    std::vector<int64_t> n_head_kv;   ///< 0 = KDA layer, else MLA layer
    std::vector<int64_t> n_ff_exp;    ///< expert FFN width
    std::vector<float> clamp_exp;     ///< SwiGLU clamp of the routed experts, 0 = none
    std::vector<float> clamp_shexp;   ///< SwiGLU clamp of the shared expert and the dense FFN, 0 = none
    std::vector<uint8_t> indexer_full;///< 1 = the layer runs its own indexer, 0 = reuses the previous selection

    // MoE
    int64_t n_ff = 0;                 ///< dense FFN width (the leading dense layers)
    int64_t n_layer_dense_lead = 0;
    int64_t n_expert = 0;
    int64_t n_expert_used = 0;
    int64_t n_expert_shared = 0;
    float expert_weights_scale = 0;   ///< 0 or 1 = no scaling
    bool expert_weights_norm = false;
    int expert_gating_func = 2;       ///< llama.cpp's enum: 1 softmax, 2 sigmoid

    // KDA
    int64_t kda_head_dim = 0;
    int64_t ssm_d_conv = 0;
    float kda_gate_lower_bound = 0;   ///< used when has_gate_lower_bound
    bool has_gate_lower_bound = false;

    // MLA
    int64_t q_lora_rank = 0;
    int64_t kv_lora_rank = 0;
    int64_t head_k_mla = 0;           ///< qk_nope + qk_rope; rope is 0 here
    int64_t head_v_mla = 0;
    int64_t n_rot = 0;

    // k-pool indexer
    int64_t indexer_n_head = 0;
    int64_t indexer_head_size = 0;
    int64_t indexer_top_k = 0;
    int64_t indexer_kpool = 0;
    bool indexer_select_tail = true;  ///< read_config's default too: the released GGUF lacks the key

    // mHC
    int64_t hc = 0;
    int64_t hc_sinkhorn_iters = 0;
    float hc_eps = 1e-6f;

    bool is_kda(int64_t il) const { return n_head_kv[(size_t) il] == 0; }
    bool is_dense(int64_t il) const { return il < n_layer_dense_lead; }
    int64_t hc_mix_dim() const { return (2 + hc) * hc; }
};

struct KdaWeights {
    Tensor wq, wk, wv;                ///< [n_embd, d_inner]
    Tensor conv_q, conv_k, conv_v;    ///< [d_conv, 1, d_inner]
    Tensor f_a, f_b;                  ///< decay gate, low rank
    Tensor beta;                      ///< [n_embd, n_head]
    Tensor a;                         ///< [n_head], holds -exp(A_log)
    Tensor dt_b;                      ///< [d_inner]
    Tensor g_a, g_b;                  ///< output gate, low rank
    Tensor o_norm;                    ///< [head_dim]
};

struct MlaWeights {
    Tensor q_a, q_a_norm, q_b;
    Tensor kv_a, kv_a_norm;
    Tensor k_b;                       ///< [qk_nope, kv_lora, n_head]
    Tensor v_b;                       ///< [kv_lora, head_v, n_head]
    // indexer (absent on a layer that reuses the previous selection)
    Tensor idx_k_norm, idx_k_norm_b, idx_proj, idx_attn_k, idx_attn_q_b, idx_kpool_gate, idx_kpool_ape;
};

struct LayerWeights {
    Tensor attn_norm, ffn_norm;
    Tensor hc_attn_fn, hc_attn_base, hc_attn_scale;
    Tensor hc_ffn_fn, hc_ffn_base, hc_ffn_scale;
    KdaWeights kda;
    MlaWeights mla;
    Tensor wo;
    // dense FFN
    Tensor ffn_gate, ffn_up, ffn_down;
    // MoE
    Tensor gate_inp, exp_probs_b;
    Tensor gate_exps, up_exps, down_exps;   ///< [n_embd, n_ff_exp, n_expert] / [n_ff_exp, n_embd, n_expert]
    Tensor gate_shexp, up_shexp, down_shexp;
};

/// The MTP block's own tensors (llama.cpp's NextN: `blk.<n_layer>.nextn.*`). Its attention and FFN are a LayerWeights
/// like a trunk layer's (MLA with its own indexer, routed experts and the shared expert; no mHC). The optional ones
/// fall back to the trunk's: shared_head_norm to output_norm, shared_head_head to output, embed_tokens to token_embd.
struct NextnWeights {
    Tensor eh_proj;                   ///< [2 * n_embd, n_embd]: the normed embedding and hidden state, concatenated
    Tensor enorm, hnorm;              ///< [n_embd]
    Tensor shared_head_norm, shared_head_head, embed_tokens;   ///< optional
};

/// A GLM-5.3-Flash GGUF, mapped. Construction reads the configuration and resolves every tensor the forward pass
/// needs, and throws with the tensor's name and the expected shape when one is missing or has another shape.
class Model {
public:
    explicit Model(const std::string& first_shard);

    const Config& config() const { return cfg_; }
    /// Layer `il` of the trunk, or with il == config().n_layer the MTP block (when has_mtp()).
    const LayerWeights& layer(int64_t il) const { return layers_[(size_t) il]; }
    /// The file carries the MTP block (nextn_predict_layers >= 1 and its tensors): layer(n_layer) and nextn().
    bool has_mtp() const { return has_mtp_; }
    const NextnWeights& nextn() const { return nextn_; }
    const Tensor& tok_embd() const { return tok_embd_; }
    const Tensor& output_norm() const { return output_norm_; }
    const Tensor& output() const { return output_; }
    const strata::GgufModel& gguf() const { return *gguf_; }
    /// The routed experts' bytes (most of the file).
    size_t expert_bytes() const;
    /// Reads every routed expert once, so they sit in the page cache: the experts are mapped from the file, and
    /// the first prompt after a start would otherwise fault them in from disk while it runs. `progress(done,
    /// total)` is called every couple of seconds from the calling thread.
    void warm_experts(int threads, const std::function<void(size_t, size_t)>& progress) const;
    /// Points layer `il`'s routed experts at a copy of the same bytes elsewhere (pinned_experts.hpp copies them into
    /// page-locked RAM); everything that reads them through layer() then reads the copy, not the mapped file.
    void move_experts(int64_t il, const void* gate, const void* up, const void* down);

private:
    Tensor get(const std::string& name, std::vector<int64_t> shape, bool required = true) const;
    void read_config();
    void resolve();

    std::unique_ptr<strata::GgufModel> gguf_;
    Config cfg_;
    std::vector<LayerWeights> layers_;
    Tensor tok_embd_, output_norm_, output_;
    bool has_mtp_ = false;
    NextnWeights nextn_;
};

}  // namespace strata::glm
