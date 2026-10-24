// include/strata/glm/engine.hpp - GLM-5.3-Flash on the GPUs, one token at a time.
//
// The layers are split over the GPUs in order. Everything except the routed experts lives in VRAM; the routed
// experts stay in RAM (the mapped GGUF) and the CPU computes them while the GPU computes the shared expert
// (Strata's split). The op order is src/glm/cpu_ref.cpp's, kernel by kernel, so the two can be compared layer
// by layer (strata-glm --check-cpu).
#pragma once

#include "strata/glm/cpu_ref.hpp"
#include "strata/glm/expert_cache.hpp"
#include "strata/glm/model.hpp"
#include "strata/glm/snapshot.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::glm {

struct EngineOptions {
    int64_t max_ctx = 32768;
    std::vector<int> devices;        ///< CUDA device ids, in layer order; empty = every visible device
    std::vector<double> split;       ///< share of the layers per device; empty = by free memory
    int cpu_threads = 0;             ///< 0 = one per CPU core (ThreadPool::physical_cores)
    bool verbose = true;
    int prefill_chunk = 256;         ///< prompt tokens read at once (0: one at a time)
    int prefill_rows = 0;            ///< of those, tokens at a time outside the routed experts (0: by memory, <= 512)
    bool kv_in_ram = false;          ///< the f16 latent cache in page-locked host RAM, not VRAM (--kv-in-ram)
    bool prefill_spread = false;     ///< a chunk's streamed experts computed on every card (--prefill-spread)
    int decode_help = 0;             ///< a token's missed experts per layer the GPUs take from the CPU (--decode-help)
    bool decode_graphs = true;       ///< a decoded token's fixed runs of ops as one CUDA graph per layer (--no-decode-graphs)
    bool lend_prompt_buffers = true; ///< between prompts the prompt buffers hold expert-cache slots (--no-lend-prompt-buffers)
    int lend_min_prompt = 64;        ///< with that, a shorter prompt runs token by token (--lend-min-prompt)
    bool mtp = false;                ///< a token and its draft in one pass, and the model's MTP block (--mtp)
    ExpertCacheOptions cache;
};

class GpuEngine {
public:
    GpuEngine(const Model& model, const EngineOptions& opt);
    ~GpuEngine();
    GpuEngine(const GpuEngine&) = delete;
    GpuEngine& operator=(const GpuEngine&) = delete;

    void reset();
    /// Runs `token` at the next position; returns the logits (n_vocab floats, host memory).
    const std::vector<float>& eval(int32_t token, Trace* trace = nullptr);
    /// --mtp: runs `a` and `b` at the next two positions in one pass (b is a draft of the token after a); returns
    /// both positions' logits, a's then b's (2 x n_vocab floats, host memory). Each is bitwise what eval would give.
    const float* eval_pair(int32_t a, int32_t b);
    /// Right after eval_pair: takes its second token back (pos() goes back by one), as if only `a` had run.
    void drop_last();
    /// --mtp on a model with an MTP block (Model::has_mtp): mtp_draft works, and every pass (eval, eval_pair, the
    /// prompt's chunks) writes the block's rows for its positions, from its tokens and the trunk's hidden states.
    bool mtp_ready() const;
    /// The MTP block at pos(): `token` (sampled from the last logits, not run yet) with the hidden state the trunk
    /// left at pos() - 1. Returns the block's most likely next token, the draft for eval_pair(token, draft); its
    /// logits in mtp_logits(). The pass over `token` at pos() keeps the row this wrote.
    int32_t mtp_draft(int32_t token);
    const std::vector<float>& mtp_logits() const;
    /// The cells the MTP block's indexer selected at the last mtp_draft, kept from mtp_keep_selection(true) on (the
    /// checks compare it with the CPU reference's).
    const std::vector<int32_t>& mtp_selection() const;
    void mtp_keep_selection(bool on);
    /// Runs `n` prompt tokens in chunks; returns the last one's logits. `progress(done)` after each chunk; false
    /// from it stops early (the state then holds the tokens read so far, pos() says how many).
    const std::vector<float>& prefill(const int32_t* ids, int64_t n,
                                      const std::function<bool(int64_t)>& progress = nullptr);
    int64_t pos() const { return pos_; }
    /// Copies the state to host memory (snapshot.hpp); `restore` copies one back, pos() included.
    std::shared_ptr<Snapshot> snapshot(bool full);
    size_t snapshot_bytes(bool full) const;   ///< what snapshot(full) would hold now, logits not included
    void restore(const Snapshot& snap);
    /// The layers each device holds, for the log: "0: layers 0-21 (8.1 GB)".
    const std::vector<std::string>& placement() const { return placement_; }
    /// The VRAM expert cache's hit rate so far, and a one-line description.
    double cache_hit_rate() const;
    int64_t cache_hits() const;
    int64_t cache_lookups() const;
    std::string cache_summary() const;
    int64_t cache_slots() const;          ///< expert slots over every device
    size_t cache_bytes() const;           ///< the VRAM they take
    /// Counts layers run, from any thread: it moves while the engine works (serve_loop's heartbeat).
    uint64_t ticks() const { return ticks_.load(std::memory_order_relaxed); }
    /// Writes the expert use profile (when --expert-profile names a file).
    void save_profile() const;
    /// The expert cache's copies land, and it then keeps what it holds (--check-pairs: two runs place experts alike);
    /// freeze_cache(false) lets it change again.
    void freeze_cache(bool on = true);

private:
    /// eval and eval_pair: `n` tokens (1 or 2) at the next positions
    void eval_n(const int32_t* tok, int n, Trace* trace);
    /// eval_n's MLA attention: layer `il` (or the MTP block, il == n_layer) on device `ci`, `n` tokens at positions
    /// t..t+n-1, from the device's normed input xn to its y
    void mla_decode(int64_t il, int ci, int n, int64_t t);
    /// --mtp: the MTP block's input at one position into its `cur`: eh_proj of [enorm(token's embedding), hnorm(h)],
    /// then attn_norm into the output device's xn; `slot` (< kNMax) is the embedding's staging row
    void mtp_input(int32_t token, const float* h, int slot);
    /// --mtp: the MTP block's rows (its latent row, its indexer's key and gate) at positions t..t+n-1 of a decode
    /// pass, the hidden state before position t being `h_first` and before t + j the pass's row j - 1
    void mtp_rows(const int32_t* tok, int n, int64_t t, const float* h_first, const float* h_rows);
    /// --mtp: the same for a prompt chunk of T tokens at t0 (engine_prefill.cu), from the streams it left
    void mtp_rows_chunk(const int32_t* ids, int T, int64_t t0);
    /// T prompt tokens; `sparse`: past the indexer's dense range, each query attends its own selection
    bool run_chunk(const int32_t* ids, int T, bool sparse);
    void setup_prefill();                        ///< the chunk buffers, before the expert cache takes the rest
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int64_t pos_ = 0;
    std::atomic<uint64_t> ticks_{0};
    std::vector<std::string> placement_;
};

}  // namespace strata::glm
