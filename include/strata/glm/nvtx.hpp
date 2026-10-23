// include/strata/glm/nvtx.hpp - named ranges for Nsight Systems (nsys), and a capture window for prompt chunks.
//
// NvtxRange marks a stretch of the calling thread in an nsys timeline (the NVTX headers come with the CUDA toolkit;
// without them it compiles to nothing, and with no profiler attached a range costs next to nothing).
// STRATA_GLM_PROFILE=FROM,N makes the prompt path call cudaProfilerStart at the first chunk that starts at or past
// position FROM and cudaProfilerStop after N chunks, so `nsys profile --capture-range=cudaProfilerApi` records only
// those chunks: a long prompt's deep chunks without the hours before them.
#pragma once

#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define STRATA_HAVE_NVTX 1
#endif
#endif

namespace strata::glm {

/// A range opened and closed by hand, for one that does not end with a C++ scope.
inline void nvtx_push(const char* name) {
#ifdef STRATA_HAVE_NVTX
    nvtxRangePushA(name);
#else
    (void) name;
#endif
}
inline void nvtx_pop() {
#ifdef STRATA_HAVE_NVTX
    nvtxRangePop();
#endif
}

class NvtxRange {
public:
    explicit NvtxRange(const char* name) {
#ifdef STRATA_HAVE_NVTX
        nvtxRangePushA(name);
#else
        (void) name;
#endif
    }
    ~NvtxRange() {
#ifdef STRATA_HAVE_NVTX
        nvtxRangePop();
#endif
    }
    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;
};

}  // namespace strata::glm
