// include/strata/glm/serve_loop.hpp - the engine side of serve/'s line protocol, for GLM-5.3-Flash.
//
// The engine stays resident and takes requests on stdin; serve/glm_server.py (and Strata's own server) drive it:
//
//   -> GEN <max_new> [key=value ...] <id,id,...>     generate after these prompt tokens
//   <- RESUME <n>                                    n prompt tokens were already in the state and are not read again
//   <- PP <done> <total> <ms> <tok/s>                prompt progress, one line per chunk
//   <- T <id>                                        a generated token
//   <- DONE <generated> <prompt> <prompt ms> <decode ms> <finish> 0 0 <reused> <hits> <lookups> 0 0 0.0 <read> 0
//   <- ERR <reason>
//   -> STOP                                          end the current request (finish "cancel")
//   -> QUIT
//
// At start it prints `INFO key=value ...` lines and `READY <context> stop`. A prompt that continues the tokens the
// state holds (the previous prompt and its reply, the usual chat turn) reads only what is new; any other prompt
// starts from position 0 (the KDA layers' state cannot be rewound).
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace strata::glm {

/// What the serve loop needs from an engine: GpuEngine, or CpuRef for tests without a GPU.
class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual void reset() = 0;
    virtual const std::vector<float>& eval(int32_t token) = 0;
    /// The prompt tokens; `progress(done)` after each chunk, false from it stops early. Returns the last logits.
    virtual const std::vector<float>& prefill(const int32_t* ids, int64_t n,
                                              const std::function<bool(int64_t)>& progress) {
        const std::vector<float>* last = nullptr;
        for (int64_t i = 0; i < n; ++i) {
            last = &eval(ids[i]);
            if ((i + 1) % 64 == 0 || i + 1 == n)
                if (progress && !progress(i + 1)) break;
        }
        return *last;
    }
    /// Tokens the state holds.
    virtual int64_t pos() const = 0;
    virtual int64_t hits() const { return 0; }
    virtual int64_t lookups() const { return 0; }
};

struct ServeOptions {
    int64_t max_ctx = 0;
    std::vector<int32_t> eos_ids;
    std::vector<std::string> info;   ///< extra INFO key=value pairs
};

/// Runs until QUIT or the end of stdin. Returns the process exit code.
int serve_loop(Evaluator& ev, const ServeOptions& opt);

}  // namespace strata::glm
