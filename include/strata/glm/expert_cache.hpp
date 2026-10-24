// include/strata/glm/expert_cache.hpp - the routed experts most used, held in VRAM.
//
// Strata's design: every routed expert stays in RAM, and each GPU keeps a cache of the ones its layers use most,
// so those run on the GPU and only the rest on the CPU. A slot holds one expert of one layer as raw GGUF blocks,
// [gate rows | up rows | down rows], on the device that runs the layer. Each layer gets the same number of slots
// (every layer routes the same number of experts per token).
//
// What to cache: every expert of a layer has a use score that decays per token (half-life ~14 tokens by
// default: while the model writes an answer it keeps coming back to a small set of experts, and a cache that
// follows it holds about two thirds of each token's experts, against about a third with a half-life of ~700). A missed expert takes a free slot, or replaces the lowest-scored cached one when its own score is
// clearly higher. The swaps per token are capped, so a change of topic moves the cache over a few dozen tokens
// rather than stalling one. A swap's copy runs on a thread of its own while decoding goes on: the old expert leaves
// its slot at once, and the new one takes it at the first token boundary after its copy is done. The scores can be saved and loaded (--expert-profile), so a restart begins with the
// cache already filled.
#pragma once

#include "strata/glm/gpu_linear.hpp"
#include "strata/glm/model.hpp"

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::glm {

struct ExpertCacheOptions {
    double vram_reserve_gb = 1.0;   ///< left free on each device after the cache
    int max_swaps_per_token = 32;   ///< experts copied in per token, all layers together
    double decay = 0.95;            ///< per-token decay of the use scores
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
    std::vector<int32_t> expert_in;          ///< per slot: its expert, -1 when free, kLoading while it is copied
    gpu::DevMatrix gate, up, down;           ///< the shapes and kernels; `w` set per slot by matrices()
};

class ExpertCache {
public:
    static constexpr int32_t kLoading = -2;   ///< expert_in of a slot whose copy is in flight

    ExpertCache(const Model& model, const ExpertCacheOptions& opt);
    ~ExpertCache();

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
    /// After a token: puts the copies that have finished in their slots, decays the scores, and queues the copies
    /// of the misses that earned a slot (the copy thread does them).
    void maintain();
    /// Waits for the queued copies and puts them in their slots (before the prompt path claims free slots).
    void drain();
    /// Writes the scores to the profile path (no-op without one).
    void save_profile() const;

    /// STRATA_GLM_VERIFY: the bytes in `slot` of layer `il` against expert `e` in RAM; prints the first difference.
    bool check_slot(int64_t il, int slot, int e, const char* what) const;
    bool verifying() const { return verify_; }

    int64_t hits() const { return hits_; }
    int64_t lookups() const { return lookups_; }
    std::string summary() const;
    int64_t total_slots() const;
    size_t total_bytes() const;

private:
    struct Copy {
        int64_t il;
        int e, slot;
    };
    void load(int64_t il, int e, int s);
    void copy_expert(int64_t il, int e, int slot, void* stream);   ///< the bytes only; the maps are the caller's
    void copier();                                                 ///< the copy thread
    void apply_done();                                             ///< finished copies into their slots

    const Model& m_;
    ExpertCacheOptions opt_;
    std::vector<ExpertSlots> L_;
    std::vector<std::vector<float>> score_;           ///< [layer][expert]
    std::vector<std::vector<int32_t>> missed_;        ///< this token's misses per layer
    int64_t hits_ = 0, lookups_ = 0, tick_ = 0;
    bool verify_ = false;                             ///< STRATA_GLM_VERIFY=1: every copy and, every 512 tokens,
                                                      ///< every cached slot checked against RAM
    bool sync_swaps_ = false;                         ///< STRATA_GLM_SYNC_SWAPS=1: a token's swaps finish before
                                                      ///< maintain() returns (a test setting)
    bool loaded_profile_ = false;
    // the copy thread: copies queued by maintain(), done ones handed back; all under mu_
    std::thread copier_;
    std::mutex mu_;
    std::condition_variable cv_todo_, cv_idle_;
    std::deque<Copy> todo_;
    std::vector<Copy> done_;
    int busy_ = 0;
    bool stop_ = false;
};

}  // namespace strata::glm
