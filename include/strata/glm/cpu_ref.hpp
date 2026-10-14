// include/strata/glm/cpu_ref.hpp - GLM-5.3-Flash on the CPU, one token at a time: the engine's reference.
//
// This is not the fast path. It exists so every GPU kernel has something to be checked against that is itself
// checked against llama.cpp (tools/glm53_compare.py). Every product goes through ggml-cpu's own dot products with
// the activation quantized to the weight's `vec_dot_type`, which is what llama.cpp's CPU backend does, so the
// two agree to float rounding on any quant type.
//
// The op order is llama.cpp's `src/models/glm5-next.cpp` (fused `gated_delta_net` op for the KDA recurrence).
#pragma once

#include "strata/glm/model.hpp"
#include "strata/glm/snapshot.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace strata::glm {

class CpuExperts;

/// A fixed set of worker threads for `parallel_for`.
class ThreadPool {
public:
    explicit ThreadPool(int n_threads);
    ~ThreadPool();
    int size() const { return n_; }
    /// Calls `fn(begin, end)` on disjoint chunks covering [0, n), and returns when all are done.
    void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int n_;
};

/// Per-token tensors written while evaluating, for bisecting against llama.cpp: the residual streams after every
/// layer (`hc * n_embd` floats per layer).
struct Trace {
    std::vector<std::vector<float>> layer_out;
};

class CpuRef {
public:
    /// `max_ctx`: the longest sequence `eval` will be called for (sizes the MLA and indexer caches).
    CpuRef(const Model& model, int64_t max_ctx, int n_threads);
    ~CpuRef();

    /// Clears every cache and recurrent state: the next `eval` is position 0.
    void reset();
    /// Runs `token` at the next position and returns its logits (`n_vocab` floats).
    const std::vector<float>& eval(int32_t token, Trace* trace = nullptr);
    int64_t pos() const { return pos_; }
    /// Saves the state (snapshot.hpp); `restore` puts one back, pos() included.
    std::shared_ptr<Snapshot> snapshot(bool full) const;
    void restore(const Snapshot& snap);

private:
    struct LayerState;
    void matvec(const Tensor& w, const float* x, float* y);
    void hc_pre(const Tensor& fn, const Tensor& scale, const Tensor& base, const float* R, float* x, float* post,
                float* comb);
    void hc_post(const float* y, float* R, const float* post, const float* comb);
    void kda_layer(int64_t il, const float* x, float* y);
    void mla_layer(int64_t il, const float* x, float* y);
    void ffn(int64_t il, const float* x, float* y);
    void swiglu_ffn(const Tensor& gate, const Tensor& up, const Tensor& down, float limit, const float* x, float* y);

    const Model& m_;
    const Config& c_;
    int64_t max_ctx_;
    int64_t pos_ = 0;
    std::unique_ptr<ThreadPool> pool_;
    std::unique_ptr<CpuExperts> experts_;
    std::vector<std::unique_ptr<LayerState>> st_;
    std::vector<int32_t> sel_;        ///< the selected cells of the last full indexer layer, this token
    std::vector<float> logits_;
};

}  // namespace strata::glm
