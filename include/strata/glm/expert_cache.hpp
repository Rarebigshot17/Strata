// include/strata/glm/expert_cache.hpp - the routed experts most used, held in VRAM.
//
// Strata's design: every routed expert stays in RAM, and each GPU keeps a cache of the ones its layers use most,
// so those run on the GPU and only the rest on the CPU. A slot holds one expert of one layer as raw GGUF blocks,
// [gate rows | up rows | down rows], on the device that runs the layer. Each layer gets the same number of slots
// (every layer routes the same number of experts per token).
//
// What to cache: every expert of a layer has a use score that decays per token (half-life ~700 tokens by
// default). A missed expert takes a free slot, or replaces the lowest-scored cached one when its own score is
// clearly higher. The swaps per token are capped, so a change of topic moves the cache over a few dozen tokens
// rather than stalling one. The scores can be saved and loaded (--expert-profile), so a restart begins with the
// cache already filled.
#pragma once

#include "strata/glm/gpu_linear.hpp"
#include "strata/glm/model.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::glm {

struct ExpertCacheOptions {
    double vram_reserve_gb = 1.0;   ///< left free on each device after the cache
    int max_swaps_per_token = 8;    ///< experts copied in per token, all layers together
    double decay = 0.999;           ///< per-token decay of the use scores
    std::string profile;            ///< scores file: loaded at start, written by save_profile()
    bool disabled = false;
};

struct ExpertSlots {
    bool cacheable = false;
    int dev = -1;
    int n_slots = 0;
    size_t blob = 0, up_off = 0, down_off = 0;
    uint8_t* d = nullptr;                    ///< n_slots * blob device bytes
    std::vector<int32_t> slot_of;            ///< per expert: its slot, or -1
    std::vector<int32_t> expert_in;          ///< per slot: its expert, or -1
    gpu::DevMatrix gate, up, down;           ///< the shapes and kernels; `w` set per slot by matrices()
};

class ExpertCache {
public:
    ExpertCache(const Model& model, const ExpertCacheOptions& opt);

    /// Carves the slots of the MoE layers on `dev` out of what is free there (minus the reserve). `layers` are
    /// the layers that device runs. Returns the device bytes used.
    size_t allocate(int dev, const std::vector<int64_t>& layers);
    /// Fills the slots from the loaded profile (if any): the highest-scored experts of each layer.
    void warm_start();

    /// The slot holding expert `e` of layer `il`, or -1.
    int slot(int64_t il, int e) const;
    /// The gate, up and down matrices of the expert in `slot` of layer `il`.
    void matrices(int64_t il, int slot, gpu::DevMatrix& gate, gpu::DevMatrix& up, gpu::DevMatrix& down) const;
    const ExpertSlots& layer(int64_t il) const { return L_[(size_t) il]; }

    /// A free slot of layer `il` for expert `e` (now mapped to it), or -1. The caller fills it (the prompt path
    /// copies the expert there while it streams it in).
    int claim_free(int64_t il, int e);
    uint8_t* slot_ptr(int64_t il, int slot) const { return L_[(size_t) il].d + (size_t) slot * L_[(size_t) il].blob; }

    /// Counts this token's routing of layer `il` (k experts).
    void record(int64_t il, const int32_t* ids, int k);
    /// After a token: decays the scores and copies in the misses that earned a slot (synchronous copies).
    void maintain();
    /// Writes the scores to the profile path (no-op without one).
    void save_profile() const;

    int64_t hits() const { return hits_; }
    int64_t lookups() const { return lookups_; }
    std::string summary() const;
    int64_t total_slots() const;
    size_t total_bytes() const;

private:
    void load(int64_t il, int e, int s);

    const Model& m_;
    ExpertCacheOptions opt_;
    std::vector<ExpertSlots> L_;
    std::vector<std::vector<float>> score_;           ///< [layer][expert]
    std::vector<std::vector<int32_t>> missed_;        ///< this token's misses per layer
    int64_t hits_ = 0, lookups_ = 0, tick_ = 0;
    bool loaded_profile_ = false;
};

}  // namespace strata::glm
