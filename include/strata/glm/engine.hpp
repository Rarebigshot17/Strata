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

#include <cstdint>
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
    int64_t pos() const { return pos_; }
    /// The layers each device holds, for the log: "0: layers 0-21 (8.1 GB)".
    const std::vector<std::string>& placement() const { return placement_; }
    /// The VRAM expert cache's hit rate so far, and a one-line description.
    double cache_hit_rate() const;
    std::string cache_summary() const;
    /// Writes the expert use profile (when --expert-profile names a file).
    void save_profile() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    int64_t pos_ = 0;
    std::vector<std::string> placement_;
};

}  // namespace strata::glm
