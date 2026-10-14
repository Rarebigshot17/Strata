// include/strata/glm/snapshot.hpp - a saved engine state, for parking conversations (serve_loop.hpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::glm {

/// The state after `pos` tokens, and the logits after the last one (filled in by the serve loop).
///
/// A full snapshot holds everything. A light one (a checkpoint) holds only what later tokens overwrite: the KDA
/// layers' recurrent state and conv windows, and the indexer's pool being filled. Restoring a light one relies on
/// the MLA cache rows (and pooled keys) below `pos` still being the same conversation's.
struct Snapshot {
    int64_t pos = 0;
    bool full = false;
    size_t bytes = 0;                 ///< host memory held, logits included
    std::vector<float> logits;
    virtual ~Snapshot() = default;
};

}  // namespace strata::glm
