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
    int cpu_threads = 0;             ///< 0 = all hardware threads
    bool verbose = true;
    int prefill_chunk = 256;         ///< prompt tokens read at once (0: one at a time)
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
    /// Writes the expert use profile (when --expert-profile names a file).
    void save_profile() const;

private:
    /// T prompt tokens; `sparse`: past the indexer's dense range, each query attends its own selection
    bool run_chunk(const int32_t* ids, int T, bool sparse);
    void setup_prefill();                        ///< the chunk buffers, before the expert cache takes the rest
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int64_t pos_ = 0;
    std::vector<std::string> placement_;
};

}  // namespace strata::glm
