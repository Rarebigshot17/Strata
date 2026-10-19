# Strata's server on GLM-5.3-Flash: the port plan

Decided by the owner: Strata's full server (`serve/server.py` with its web app, OpenAI / Anthropic / Responses
APIs, MCP, structured output, metrics) drives the GLM engine (`strata-glm`), and the Qwen engine and HIP go now.
This file is the plan and its record: each step says what is done, so work can resume from here after any
interruption. It was written from a read of the whole server, the engine protocol, the tests, the build and
`setup.py` (seven maps and a critique, October 2026).

Status legend: [ ] to do, [x] done (with the commit), [~] partly.

## Decisions (and why)

| Kept | Removed | Added |
|---|---|---|
| OpenAI chat, Anthropic Messages, Responses, MCP, structured output, the web app and Monitor, /metrics and Prometheus, lazy load / unload / idle unload, API keys, CORS, host checks, shared settings, stop strings, reasoning budget, same-token stop (#606), loop detection (stop mode), non-finite retry (#879) | The Qwen engine and every Qwen-only feature: batch slots (BGEN...), vision (GENI, strata-vision), replicas (one engine needs all three GPUs and 120-170 GB of RAM), the VRAM command, slot save/restore (SAVE/RESTORE: the route answers 501), MTP drafts, the speed projection (cvec), effort_end (`--tail-role-token`), focused recovery (Qwen's effort sentence), `pin=`/`ckpt=`/`strata_tune` keys, the top_k <= 64 cap; HIP/AMD everywhere; the Windows launchers (the GLM engine is POSIX-only); the prebuilt-engine download | `cp=<n>` checkpoints from the server (multi-turn chat no longer re-reads the conversation); `logit_bias` in the GLM sampler; `/v1/completions`; requests served in arrival order; GLM's tokenizer and chat template read from the GGUF |

Defaults: port **8101** everywhere (Open WebUI 3000, Portainer 8000 and the UPS dashboard 8080 are taken on
napster-ai). Model name from the GGUF's `general.name`. Sampling defaults when a request sets none:
temperature 1.0, top_p 0.95 (GLM's recommendation). Thinking is on/off only: GLM's template has no effort
levels, so low/medium/high mean "on" and none/off mean "off".

## The GLM engine protocol (what the server speaks)

```
start   <- INFO engine=<version> backend=cuda|cpu-reference eos=<n> park_mib=<int> logit_bias=1
               [expert_slots=<n> expert_cache_mib=<int>] ...
        <- READY <ctx> stop
request -> GEN <max_new> [cp=<n>] [end=<id,...>] [logit_bias=id:bias,...] [temperature= top_p= top_k= min_p=
               penalty_repeat= penalty_freq= penalty_present= penalty_last_n= seed=] <id,id,...>
                                         (unknown keys ignored; end=: more ids that end the request as an end of
                                         turn does.  The last printed token is never taken into the state, so the
                                         prompt plus every printed token continues it)
        <- RESUME <n>                    n prompt tokens came from a kept state (only when n > 0)
        <- PP <done> <total> <ms> <tok/s>   at most 4 a second, always once at done == total
        <- HB                            the engine is working (a long prompt chunk); at most every few seconds,
                                         only while it makes progress
        <- T <id>                        each generated token; an end-of-turn id is printed, then DONE stop
        <- DONE <gen> <prompt> <prompt_ms> <decode_ms> <stop|length|cancel|nonfinite> 0 0 <reused> <hits>
                <lookups> 0 0 0.0 <read> 0        (hits/lookups: the expert cache during decode)
        <- ERR <reason>                  the request was refused (malformed, does not fit); no DONE follows
        <- ERR internal: <what>          the engine failed; the request's state is gone; no DONE follows.
                                         The engine may exit after it (a CUDA fault).
control -> STOP                          ends the last GEN's request, even a queued one
        -> QUIT                          (send STOP first: QUIT is read between requests)
```

Engine options (strata-glm): `-m <first shard> --ctx N --eos-ids a,b,c [--devices 0,1,2] [--split w,w,w]
[--threads N] [--vram-reserve GB] [--expert-profile FILE] [--max-swaps N] [--no-expert-cache]
[--prefill-chunk N] [--park-mib N] [--checkpoint-min N] [--no-warm]`. It exits 2 on any other option.
strata-glm-ref takes `-m --ctx --eos-ids --threads --park-mib --checkpoint-min`.

## Steps

### A. Engine (C++)

Done: A1-A3 in f8efa24, A4 in 29056d3 (ctest, the selftest, the park and server checks pass).

- [x] A1 Sampler: no full-vocabulary sort (top-p by growing partial sorts, min-p by threshold, plain sampling
  in O(n)); `logit_bias` (added before penalties, -100 or less bans); seed comment.
- [x] A2 Serve loop: INFO keys above (integer `park_mib`, `logit_bias=1`, version, expert cache figures);
  `ERR internal:` for engine failures; `HB` heartbeats gated on progress; DONE hits/lookups from decode only.
- [x] A3 Expert warm-up before READY (reads the routed experts into the page cache, with progress on stderr;
  `--no-warm` skips it): the first request after a start no longer faults 120-170 GB in from disk while the
  server's silence watchdog runs.
- [x] A4 CMake: GLM-only (ggml, `strata_glm`, a 3-file `strata_glm_kernels`, `strata_glm_gpu`, the two
  executables). Delete the Qwen engine tree, HIP, `tests/`, the C++ benches, `ref/`, Qwen `data/`.

### B. Server (Python)

In progress on work-in-progress branches, one per part, each pushed after every commit and merged into
glm53-flash once reviewed: `glm53-flash-wip-frontend` (B1), `glm53-flash-wip-server` (B2),
`glm53-flash-wip-web` (B4 and the small serve modules), `glm53-flash-wip-setup` (C1), `glm53-flash-wip-tools`
(C2). The tests (B3), the docs (C3) and the review (D1) follow the merge.

State of the branches (October 19): A, B, C1, C2 and D1 are merged into glm53-flash; C3 (the docs) is what is
left. Bugs the port's tests found and fixed: a call cut off by the end of a reply was still returned as a call to
run in non-streamed OpenAI and Anthropic answers (#211); a namespace tool written as `ns__tool` came back as text on
/v1/responses; a call quoted on a "> " line inside the reasoning could be run. Added: OpenAI's
`stream_options.include_usage`. The review (D1) found 44 issues in the server and its modules, and a fifth group
reviewed the engine's serve loop and sampler itself; the real ones were reproduced by a test first, fixed and
reviewed again (groups g1-g5, merged in 7b44860 ... 7f69bc9), and the last one after them: a thinking
budget's wrap-up or a forced call's opening made the engine read the whole thinking again when the server read its
tokens more slowly than the engine wrote them (now each pass ends where the server goes on from it: `end=`).

- [x] B1 (merged in fd94fb6) `serve/frontend.py`: a GLM `OutputParser` with the same interface (events reasoning / content /
  tool_start / tool_args / tool_call; `finish`, `state`, `buf`, `pending`, `rescued`, `refused`), GLM call grammar
  `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`, values verbatim (schema "string"
  raw, else JSON), no Qwen quoted-`</think>` rule, a reply that opens its own `<think>`; `forced_call` GLM
  openings; `ChatTemplate` from a source string with `strftime_now` and GLM capability probes; literal marking
  of every type-3/4 token the tokenizer has; `clear_thinking` and other template kwargs passed through; tool
  message content flattened to text.
- [x] B2 (merged in f639844) `serve/server.py`: tokenizer and template from the GGUF (`pre` from the GGUF, default glm5);
  `stop_ids` and `--eos-ids` = `<|endoftext|>`, `<|user|>`, `<|observation|>` + GGUF eos; `cp=` computed in
  `prepare` (also for MCP rounds); engine args for strata-glm; STOP before QUIT; `ERR internal:` as a server
  error (5xx); `HB`; Monitor figures from GLM's INFO and its `conversations:` log lines; ordered FIFO;
  `/v1/completions`; removals listed under Decisions; GLM ByteTokenizer / MockEngine for tests.
- [x] B3 (parts a-e merged and reviewed; part f in f067bc1) Tests: GLM fixture kit (template = GLM, GLM
  ByteTokenizer, MockEngine ending in `<|user|>`, a GLM fake engine process); port the fixture-swap tests, rewrite
  the format tests, delete the Qwen-only ones; fold `test_glm_server.py` into them and retire
  `serve/glm_server.py`; `tools/glm53_server_check.sh` drives `serve/server.py`.
- [x] B4 (merged in aa7d1bd) `serve/runconfig.py` and the web app: GLM keys (`--vram-reserve` GB, `--park-mib`), no top_k cap,
  thinking on/off, no speed projection / MTP / vision text, GLM facts in the Monitor.

### C. Install and launch

- [x] C1 (merged in 52bb36a) `setup.py` as a GLM installer for Linux + NVIDIA: checks, 3-bit or 4-bit, a local
  build of strata-glm, the shards from `unsloth/GLM-5.3-Flash-GGUF` (file names listed from the repository at run
  time), the run config, the run script, start on 8101. `setup.sh`, `update.sh`, Docker, `chat.py`, requirements.
- [x] C2 (merged in c4d97d0) `tools/`: keep the GLM tools and generic HTTP tools; delete Qwen and HIP tooling with
  its tests.
- [~] C3 (Qwen/AMD pages and translated READMEs removed, HOW_IT_WORKS and LOGIT_BIAS rewritten) Docs: README for GLM, delete Qwen/HIP docs and the translated READMEs, AGENTS.md.

### D. Review

- [x] D1 (merged in 7b44860 ... 7f69bc9, and the passes' `end=` after them) A review pass over the whole diff
  (correctness first), fixes, all Python tests and the GLM checks (`glm53_selftest.sh`, `glm53_park_check.py`,
  `glm53_server_check.sh`) green on the CPU engine.

## Known risks to check on napster-ai

- The real GLM-5.3 template (the tests use GLM-4.7-Flash's): its generation prompt, effort or thinking kwargs,
  `strftime_now`, the tool dict shape. `tools/glm53_inspect.py` prints it; the server reads it from the GGUF.
- Token types of `<think>`, `<tool_call>`, `<arg_key>` ... in the real GGUF (type 3 or 4).
- Whether GLM writes tool calls inside its thinking.
