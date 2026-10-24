// include/strata/glm/pinned_experts.hpp - the routed experts in page-locked RAM (strata-glm --pin-experts).
//
// The experts are mapped from the GGUF, so they live in the page cache, which CUDA cannot copy from directly: a copy
// to a GPU goes through a staging buffer, and the prompt path's streaming (every expert a chunk uses that the VRAM
// cache does not hold) copies each expert twice, the first time on the CPU. Here every MoE layer's experts are copied
// once, at start, into memory that is then page-locked for every GPU to read by DMA, and the model's expert tensors
// are pointed at the copies (Model::move_experts): the CPU experts, the VRAM cache's copies and the prompt path read
// them there. The VRAM cache stays the first place an expert runs from. The memory is anonymous (2 MiB pages where
// the kernel allows), filled by all the threads, then locked with cudaHostRegister (cudaHostRegisterPortable), a
// layer's block per thread: cudaHostAlloc, which locks every page before the copy, ran at 0.65 GB/s on napster-ai
// (168 s for 109.9 GB). Each piece's pages of the file are let go once copied, so the copies do not push the pieces
// still to come out of the page cache, or themselves into swap.
//
// The memory stays locked while the engine runs (about 110 GB for the 3-bit file): the rest of the PC can neither
// use nor swap it. Layers are moved while MemAvailable, less what the layers before took, stays above `keep_free`
// after each; the others stay mapped, as does a layer whose block cannot be locked.
// The copy reads every expert once, so it also does what the warm-up (Model::warm_experts) does.
#pragma once

#include "strata/glm/model.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace strata::glm {

class PinnedExperts {
public:
    /// Copies the experts of the MoE layers, `threads` threads at a time, and points the model at the copies.
    /// `progress(done, total)` every couple of seconds, from the calling thread.
    PinnedExperts(Model& model, int threads, size_t keep_free, const std::function<void(size_t, size_t)>& progress);
    ~PinnedExperts();
    PinnedExperts(const PinnedExperts&) = delete;
    PinnedExperts& operator=(const PinnedExperts&) = delete;

    size_t bytes() const { return bytes_; }         ///< page-locked bytes held
    int layers() const { return layers_; }          ///< MoE layers moved
    int left() const { return left_; }              ///< MoE layers left mapped
    const std::string& why() const { return why_; } ///< why the first of those was left, if any

private:
    struct Watched {
        const uint8_t* p;
        size_t n;
        int64_t il;
        uint64_t hash;
    };
    void watch();                                    ///< STRATA_GLM_VERIFY: rehashes the copies every minute

    std::vector<std::pair<void*, size_t>> blocks_;   ///< each moved layer's block and its length
    // STRATA_GLM_VERIFY=1: the copies are compared with the file once they are locked, then hashed, and a thread
    // hashes them again every minute and says which pieces changed
    std::vector<Watched> watched_;
    std::thread watcher_;
    std::mutex wmu_;
    std::condition_variable wcv_;
    bool wstop_ = false;
    size_t bytes_ = 0;
    int layers_ = 0, left_ = 0;
    std::string why_;
};

/// MemAvailable from /proc/meminfo in bytes, 0 when it cannot be read.
size_t mem_available();

}  // namespace strata::glm
