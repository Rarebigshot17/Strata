// include/strata/glm/serve_loop.hpp - the engine side of serve/'s line protocol, for GLM-5.3-Flash.
//
// The engine stays resident and takes requests on stdin; serve/server.py drives it (docs/GLM53_SERVER_PORT.md):
//
//   -> GEN <max_new> [cp=<n>] [end=id,...] [logit_bias=id:v,...] [key=value ...] <id,id,...>
//                                                    generate after these prompt tokens (sampler.hpp's keys); end=:
//                                                    more ids that end the request as an end of turn does
//   <- RESUME <n>                                    n prompt tokens were already in the state and are not read again
//   <- PP <done> <total> <ms> <tok/s>                prompt progress, at most 4 a second and once at the end
//   <- HB                                            still working (a long prompt chunk): only while it progresses
//   <- T <id>                                        a generated token (an end-of-turn id too, then DONE stop)
//   <- DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts> <taken> <reused> <hits> <lookups> 0 0 0.0
//      <read> 0                                      hits/lookups: the expert cache during decode; drafts/taken:
//                                                    the MTP block's drafts and those that were the next token
//   <- ERR <reason>                                  the request is refused; no DONE follows
//   <- ERR internal: <what>                          the engine failed; the state is discarded; no DONE follows
//   -> STOP                                          end the last GEN's request, even still queued (finish "cancel")
//   -> QUIT                                          read between requests
//
// At start it prints `INFO engine=<version> eos=<n> park_mib=<n> logit_bias=1 ...` and `READY <context> stop`.
//
// CONVERSATIONS. The KDA layers' state cannot be rewound, so a prompt can only continue a state whose tokens are
// its prefix. Several clients (agents, a chat, a bot) take turns, so the loop keeps their conversations: switching
// away from one parks its whole state in host memory, and a prompt that continues a parked conversation restores
// it and reads only what is new. A prompt also leaves a checkpoint (snapshot.hpp) at `cp=<n>` from the GEN line
// (cp=0: none), or at its end without one, when that is at least checkpoint_min tokens in: the server puts it
// before the generation prompt, so the next turn (which re-renders the last reply) or a regenerated reply restarts
// there rather than at 0. Least recently used conversations, then old checkpoints, are dropped beyond park_bytes.
// A request's last printed token is never taken into the state (it ended there: max_new, an end of turn, an end= id,
// a STOP), so a prompt of the request's prompt and every token it printed continues the state.
//
// DRAFTS. An engine that can draft (GpuEngine with --mtp and the model's MTP block) decodes a sampled token with its
// draft of the next one in one pass. The next token is then sampled from the pass's first row of logits, as it would
// be from the token's own pass: when it is the draft, the second row is the logits after it, and it costs no pass of
// its own; else the draft is taken back. Every token is sampled once, from the same logits and history as without
// drafts, so a reply is the same token for token (the logits of a pass are bitwise one token's at a time).
#pragma once

#include "strata/glm/snapshot.hpp"

#include <cstdint>
#include <functional>
#include <memory>
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
    /// A counter that moves while the engine works, read from another thread (0 forever: no heartbeats).
    virtual uint64_t ticks() const { return 0; }
    /// Saving and restoring the state (snapshot.hpp). Without them only the live conversation is kept.
    virtual bool can_snapshot() const { return false; }
    /// What snapshot(full) would hold now, logits not included (so the loop can skip one that cannot be kept).
    virtual size_t snapshot_bytes(bool /*full*/) const { return 0; }
    virtual std::shared_ptr<Snapshot> snapshot(bool /*full*/) { return nullptr; }
    virtual void restore(const Snapshot& /*s*/) {}
    /// Drafting: draft(token) proposes the token after `token` (sampled, not run yet); eval_pair(a, b) runs both at
    /// the next two positions and returns a's logits then b's (n_vocab floats each); drop_last() takes b back.
    virtual bool can_draft() const { return false; }
    virtual int32_t draft(int32_t /*token*/) { return -1; }
    virtual const float* eval_pair(int32_t /*a*/, int32_t /*b*/) { return nullptr; }
    virtual void drop_last() {}
};

#ifndef STRATA_VERSION
#define STRATA_VERSION "0.0.0"
#endif

struct ServeOptions {
    int64_t max_ctx = 0;
    int64_t n_vocab = 0;                    ///< token ids must be below it (0: not checked)
    std::string version = STRATA_VERSION;   ///< INFO engine=
    size_t park_bytes = (size_t) 8 << 30;   ///< host memory for parked conversations and checkpoints; 0: none
    int64_t checkpoint_min = 256;           ///< no checkpoint closer than this to the start
    int max_checkpoints = 2;                ///< per conversation, the latest kept
    std::vector<int32_t> eos_ids;
    std::vector<std::string> info;   ///< extra INFO key=value pairs (backend, expert cache figures)
};

/// Runs until QUIT or the end of stdin. Returns the process exit code.
int serve_loop(Evaluator& ev, const ServeOptions& opt);

}  // namespace strata::glm
