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

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::glm {

class CpuExperts;

/// A fixed set of worker threads for `parallel_for`. The threads take chunks from a shared counter, so one slow
/// thread does not hold up the rest, and they spin for a few milliseconds after each job before they sleep: the
/// engine calls it twice or three times per layer, about every millisecond, and a wake-up through the kernel costs
/// about as much as the work.
class ThreadPool {
public:
    explicit ThreadPool(int n_threads);
    ~ThreadPool();
    int size() const { return n_; }
    /// Calls `fn(begin, end)` on disjoint chunks covering [0, n), and returns when all are done.
    void parallel_for(int64_t n, const std::function<void(int64_t, int64_t)>& fn);
    /// The same with chunks of `grain` items (the last one shorter), or [0, n) at once on one thread or when
    /// n <= grain.
    void parallel_for(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)>& fn);
    /// The same, and the calling thread runs `side` first, while the other threads start on the chunks (the engine
    /// launches its GPU work there). An exception from `side` is rethrown once the job is done.
    void parallel_for(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t)>& fn,
                      const std::function<void()>& side);
    /// The CPU cores (not hardware threads) Linux reports, or the hardware threads where it does not say.
    static int physical_cores();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int n_;
};

/// Per-token tensors written while evaluating, for bisecting against llama.cpp: the residual streams after every
/// layer (`hc * n_embd` floats per layer, llama.cpp's l_out). The CPU reference also keeps the points inside each
/// layer that llama.cpp's graph names: the streams after the attention (hc_attn_post, `hc * n_embd`), and the
/// mixer's input, the normed input and the output of the attention half and the FFN half (`n_embd` each).
struct Trace {
    std::vector<std::vector<float>> layer_out;     ///< l_out
    std::vector<std::vector<float>> attn_out;      ///< hc_attn_post
    std::vector<std::vector<float>> hc_attn_pre, attn_norm, hc_ffn_pre, ffn_norm, ffn_out;
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
    /// The hidden state the last eval left (`n_embd` floats): the streams' mean after output_norm, what the output head
    /// and the MTP block read (llama.cpp's h_nextn).
    const std::vector<float>& h() const { return h_; }
    /// The MTP block (Model::has_mtp) at its next position, mtp_pos(): `token` there and `h_prev`, the trunk's h() at
    /// the position before it (zeros at position 0), as llama.cpp's MTP drafting pairs them. Returns its logits, the
    /// next token's draft. Its latent cache and indexer are its own; snapshots leave them out (restore() and reset()
    /// empty them).
    const std::vector<float>& mtp_eval(int32_t token, const float* h_prev);
    int64_t mtp_pos() const { return mtp_pos_; }
    /// Counts layers run, from any thread (serve_loop's heartbeat).
    uint64_t ticks() const { return ticks_.load(std::memory_order_relaxed); }
    /// Saves the state (snapshot.hpp); `restore` puts one back, pos() included.
    std::shared_ptr<Snapshot> snapshot(bool full) const;
    size_t snapshot_bytes(bool full) const;   ///< what snapshot(full) would hold now, logits not included
    void restore(const Snapshot& snap);

private:
    struct LayerState;
    void matvec(const Tensor& w, const float* x, float* y);
    void hc_pre(const Tensor& fn, const Tensor& scale, const Tensor& base, const float* R, float* x, float* post,
                float* comb);
    void hc_post(const float* y, float* R, const float* post, const float* comb);
    void kda_layer(int64_t il, const float* x, float* y);
    void mla_layer(int64_t il, const float* x, float* y, int64_t t);
    void ffn(int64_t il, const float* x, float* y);
    void swiglu_ffn(const Tensor& gate, const Tensor& up, const Tensor& down, float limit, const float* x, float* y);

    const Model& m_;
    const Config& c_;
    int64_t max_ctx_;
    int64_t pos_ = 0;
    std::atomic<uint64_t> ticks_{0};
    std::unique_ptr<ThreadPool> pool_;
    std::unique_ptr<CpuExperts> experts_;
    std::vector<std::unique_ptr<LayerState>> st_;
    std::vector<int32_t> sel_;        ///< the selected cells of the last full indexer layer, this token
    std::vector<float> logits_;
    std::vector<float> h_;            ///< h(): the last eval's hidden state
    int64_t mtp_pos_ = 0;
    std::vector<float> mtp_logits_, enorm_, hnorm_, head_norm_;
    std::string debug_pools_;         ///< STRATA_GLM_SEL_DEBUG: the MTP block's best pools, printed with its selection
};

}  // namespace strata::glm
