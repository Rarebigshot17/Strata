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
| OpenAI chat, Anthropic Messages, Responses, MCP, structured output, the web app and Monitor, /metrics and Prometheus, lazy load / unload / idle unload, API keys, CORS, host checks, shared settings, stop strings, reasoning budget, same-token stop (#606), loop detection (stop mode), non-finite retry (#879) | The Qwen engine and every Qwen-only feature: batch slots (BGEN...), vision (GENI, strata-vision), replicas (one engine needs all three GPUs and 110-146 GB of RAM for its experts: the 3-bit file's count, the 4-bit's derived), the VRAM command, slot save/restore (SAVE/RESTORE: the route answers 501), MTP drafts, the speed projection (cvec), effort_end (`--tail-role-token`), focused recovery (Qwen's effort sentence), `pin=`/`ckpt=`/`strata_tune` keys, the top_k <= 64 cap; HIP/AMD everywhere; the Windows launchers (the GLM engine is POSIX-only); the prebuilt-engine download; `tools/strata_mcp.py` and `docs/MCP_SERVER.md` | `cp=<n>` checkpoints from the server (multi-turn chat no longer re-reads the conversation); `end=<ids>` (a reply's later passes end where the server goes on from them); `logit_bias` in the GLM sampler; `/v1/completions`; OpenAI's `stream_options.include_usage`; requests served in arrival order; GLM's tokenizer and chat template read from the GGUF; the expert warm-up before READY |

Defaults: port **8101** everywhere (Open WebUI 3000, Portainer 8000 and the UPS dashboard 8080 are taken on
napster-ai). Model name: the run config's `"model_name"` (setup writes `glm-5.3-flash`), else the GGUF's
`general.name`, else `glm-5.3-flash`. Sampling defaults when a request sets none: temperature 1.0, top_p 0.95
(GLM's recommendation). Thinking follows the real template (G below): three levels, low < high < max (the default),
and thinking off, which the server makes itself.

Behaviour the port and its review (D1) settled:

- **Tool calls and `</think>` come from the model's special tokens only**, when the tokenizer has all of GLM's tags
  (`<think>`, `</think>`, `<tool_call>`, `</tool_call>`, `<arg_key>`, ...) as special tokens (`server.output_tags`,
  `frontend.ORDINARY_LT` / `OUTPUT_TAGS`). A call copied as plain text from a tool result, a web page or the user
  stays text and is never run. With no tools declared, or `tool_choice` `"none"`, nothing becomes a call. When the
  tokenizer lacks one of the tags, the parser reads them from the text (see the known risks).
- **MCP tools run only for Strata's own page.** An API client's request with `"strata_mcp": true` is a 403 unless the
  run config sets `"mcp_for_api": true` (default false). MCP server addresses are shown and logged without user
  name, password, query string or fragment (`mcp.shown_url`).
- **`stream_options.include_usage`**: with it, one more chunk with empty `choices` carries the usage, and no other
  chunk does.
- **Sampling values out of range** (NaN, Infinity, a repetition penalty <= 0, temperature < 0, top_p or min_p
  outside 0..1, integers too large for the engine) are a 400 that names the field; the shared settings (a 400) and
  the run config's `"sampling"` block (the server does not start) are checked too, each with its own wording (a
  shared temperature may go up to 2).
- **The start**: strata-glm reads the routed experts into the page cache before READY, with progress on stderr
  (`--no-warm` skips it). The server's start watchdog (`STRATA_ENGINE_READY_S`, 900 s) counts from the last step the
  engine reported (a new start line: the layers on each GPU, the expert cache; a warm-up line that moved on), so a
  slow disk that keeps reading is not ended.
- **Request bodies**: the size limit stays (256 MiB, `STRATA_MAX_BODY_MIB`), and at most two bodies over 16 MiB are
  held at once (a third is a 503 with `Retry-After`). Reading a request's headers and body times out after 120 s
  without data; the answer has no limit, so a long stream is never cut off.
- **`response_format` patterns** run with a time limit (2 s for all of an answer's patterns) and a size limit
  (10,000 parts with the repeats written out), in `serve/structured.py`.

## The GLM engine protocol (what the server speaks)

```
start   <- INFO engine=<version> eos=<number of --eos-ids> park_mib=<int> logit_bias=1 backend=cuda|cpu-reference
               [expert_slots=<n> expert_cache_mib=<int>]        (strata-glm: the warm-up's lines on stderr first)
        <- READY <ctx> stop
request -> GEN <max_new> [cp=<n>] [end=<id,...>] [logit_bias=id:bias,...] [temperature= top_p= top_k= min_p=
               penalty_repeat= penalty_freq= penalty_present= penalty_last_n= seed= drawn=] <id,id,...>
                                         (unknown keys ignored; cp=: where the prompt is checkpointed, 0: nowhere,
                                         no key: at its end; end=: more ids that end the request as an end of
                                         turn does.  The last printed token is never taken into the state, so the
                                         prompt plus every printed token continues it.  drawn=: the seed's draws
                                         the reply's earlier passes made, one a sampled token, which this one skips)
        <- RESUME <n>                    n prompt tokens came from a kept state (only when n > 0)
        <- PP <done> <total> <ms> <tok/s>   at most 4 a second, always once at done == total
        <- HB                            the engine is working (a long prompt chunk): after 3 s without a line,
                                         only while it makes progress
        <- T <id>                        each generated token; an end-of-turn id is printed, then DONE stop
        <- DONE <gen> <prompt> <prompt_ms> <decode_ms> <stop|length|cancel|nonfinite> <drafts> <taken> <reused>
                <hits> <lookups> 0 0 0.0 <read> 0 (hits/lookups: the expert cache during decode; drafts/taken:
                                         the MTP block's drafts and those that were the next token, 0 0
                                         without --mtp)
        <- ERR <reason>                  the request was refused (a malformed line, a sampling value the sampler
                                         cannot use, an id outside the vocabulary, a prompt that does not fit);
                                         no DONE follows
        <- ERR internal: <what>          the engine failed; the request's state is gone; no DONE follows.
                                         The engine may exit after it (a CUDA fault); a failed kernel launch
                                         exits at once with `glm gpu kernel <name>: <error>` on stderr
control -> STOP                          ends the last GEN's request, even a queued one
        -> QUIT                          (send STOP first: QUIT is read between requests); strata-glm saves
                                         its expert profile (--expert-profile) and exits
        -> anything else                 <- ERR unsupported command: <word>  (the server's PING probe after an
                                         ERR internal: an engine that still answers is in step)
        (end of input)                   as QUIT: the server is gone
```

The server starts the engine in a session of its own: Ctrl+C and a closed terminal's SIGHUP reach only the server,
which ends the engine with STOP and QUIT (main(), StrataEngine.shutdown). strata-glm in --serve mode
(include/strata/glm/serve_signals.hpp) ignores SIGINT, SIGHUP and SIGTERM while the server that started it in a
session of its own runs (a systemd stop signals every process of the service), ends on them once that server is gone,
ends at once when the server goes while it still starts (before READY: the parent-death signal), and ends at a broken
stdout (SIGPIPE, restored even when the launcher ignored it). Run by hand from a shell or by a tool, it is not in a
session of its own, and Ctrl+C or a `kill` ends it as before.

Engine options (strata-glm --serve): `-m <first shard> --ctx N --eos-ids a,b,c [--devices 0,1,2]
[--split w,w,w] [--threads N] [--vram-reserve GB] [--expert-profile FILE] [--max-swaps N] [--cache-decay D]
[--no-expert-cache] [--prefill-chunk N] [--park-mib N] [--checkpoint-min N] [--no-warm]`. It exits 2 on any other option. The server
adds `--serve`, `--eos-ids` (the end-of-turn ids) and `--split` (the config's `"split"`) to the config's `"args"`,
and names the GPUs with `CUDA_VISIBLE_DEVICES` (the config's `"gpu"`), not `--devices`. strata-glm-ref takes
`-m --ctx --eos-ids --threads --park-mib --checkpoint-min` and accepts `--no-warm` (it has nothing to warm).

## Steps

### A. Engine (C++)

Done: A1-A3 in f8efa24, A4 in 29056d3 (ctest, the selftest, the park and server checks pass).

- [x] A1 Sampler: no full-vocabulary sort (top-p by growing partial sorts, min-p by threshold, plain sampling
  in O(n)); `logit_bias` (added before penalties, -100 or less bans); seed comment.
- [x] A2 Serve loop: INFO keys above (integer `park_mib`, `logit_bias=1`, version, expert cache figures);
  `ERR internal:` for engine failures; `HB` heartbeats gated on progress; DONE hits/lookups from decode only.
- [x] A3 Expert warm-up before READY (reads the routed experts into the page cache, with progress on stderr;
  `--no-warm` skips it): the first request after a start no longer faults 110-146 GB in from disk (the 3-bit
  file's experts as the inspector counted them, the 4-bit's derived) while the server's silence watchdog runs.
- [x] A4 CMake: GLM-only (ggml, `strata_glm`, a 3-file `strata_glm_kernels`, `strata_glm_gpu`, the two
  executables). Delete the Qwen engine tree, HIP, the Qwen `tests/`, the C++ benches, `ref/`, Qwen `data/`. The
  GLM engine's unit tests are in `tests/glm` (ctest).

### B. Server (Python)

Done on work-in-progress branches, one per part, each pushed after every commit and merged into glm53-flash once
reviewed: `glm53-flash-wip-frontend` (B1), `glm53-flash-wip-server` (B2), `glm53-flash-wip-web` (B4 and the small
serve modules), `glm53-flash-wip-setup` (C1), `glm53-flash-wip-tools` (C2). The tests (B3), the docs (C3) and the
review (D1) followed the merge.

State of the branches (October 19): A, B, C, D1 and E1 are merged into glm53-flash; what is left needs napster-ai
(the steps in [GLM53_PORT.md](GLM53_PORT.md#what-to-run-on-napster-ai)). Bugs the port's tests found and fixed: a
call cut off by the end of a reply was still returned as a call to run in non-streamed OpenAI and Anthropic answers
(#211); a namespace tool written as `ns__tool` came back as text on /v1/responses; a call quoted on a "> " line
inside the reasoning could be run. Added: OpenAI's
`stream_options.include_usage`. The review (D1) found 44 issues in the server and its modules, and a fifth group
reviewed the engine's serve loop and sampler itself; the real ones were reproduced by a test first, fixed and
reviewed again (groups g1-g5, merged in 7b44860 ... 7f69bc9), and the last one after them: a thinking
budget's wrap-up or a forced call's opening made the engine read the whole thinking again when the server read its
tokens more slowly than the engine wrote them (now each pass ends where the server goes on from it: `end=`).

- [x] B1 (merged in fd94fb6) `serve/frontend.py`: a GLM `OutputParser` with the same interface (events reasoning /
  content / tool_start / tool_args / tool_call; `finish`, `state`, `buf`, `pending`, `rescued`, `refused`), GLM call
  grammar `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`, values verbatim (schema "string"
  raw, else JSON), no Qwen quoted-`</think>` rule, a reply that opens its own `<think>`; `forced_call` GLM
  openings; `ChatTemplate` from a source string with `strftime_now` and GLM capability probes; literal marking
  of every type-3/4 token the tokenizer has; `clear_thinking` and other template kwargs passed through; tool
  message content flattened to text.
- [x] B2 (merged in f639844) `serve/server.py`: tokenizer and template from the GGUF (`pre` from the GGUF, default
  glm5); `stop_ids` and `--eos-ids` = `<|endoftext|>`, `<|user|>`, `<|observation|>` + GGUF eos; `cp=` computed in
  `prepare` (also for MCP rounds); engine args for strata-glm; STOP before QUIT; `ERR internal:` as a server
  error (5xx); `HB`; Monitor figures from GLM's INFO and its `conversations:` log lines; ordered FIFO;
  `/v1/completions`; removals listed under Decisions; GLM ByteTokenizer / MockEngine for tests.
- [x] B3 (parts a-e merged and reviewed; part f in f067bc1) Tests: GLM fixture kit (template = GLM, GLM
  ByteTokenizer, MockEngine ending in `<|user|>`, a GLM fake engine process); port the fixture-swap tests, rewrite
  the format tests, delete the Qwen-only ones; fold `test_glm_server.py` into them and retire
  `serve/glm_server.py`; `tools/glm53_server_check.sh` drives `serve/server.py`.
- [x] B4 (merged in aa7d1bd) `serve/runconfig.py` and the web app: GLM keys (`--vram-reserve` GB, `--park-mib`), no
  top_k cap, thinking on/off, no speed projection / MTP / vision text, GLM facts in the Monitor.

### C. Install and launch

- [x] C1 (merged in 52bb36a) `setup.py` as a GLM installer for Linux + NVIDIA: checks, 3-bit or 4-bit, a local
  build of strata-glm, the shards from `unsloth/GLM-5.3-Flash-GGUF` (file names listed from the repository at run
  time), the run config, the run script, start on 8101. `setup.sh`, `update.sh`, Docker, `chat.py`, requirements.
- [x] C2 (merged in c4d97d0) `tools/`: keep the GLM tools and generic HTTP tools; delete Qwen and HIP tooling with
  its tests.
- [x] C3 (written and fact-checked against the code, merged in 02b0c89 ... 3d381ee) Docs: README for GLM, AGENTS.md, SECURITY.md, INSTALL, AI_SETUP, DETAILS, TROUBLESHOOTING,
  HOW_IT_WORKS, LOGIT_BIAS, LLAMA_SWAP, GLM53_PORT and this page; the Qwen/HIP docs, the translated READMEs and
  `docs/MCP_SERVER.md` (its `tools/strata_mcp.py` was removed in C1) deleted.

### D. Review

- [x] D1 (merged in 7b44860 ... 7f69bc9, and the passes' `end=` after them) A review pass over the whole diff
  (correctness first), fixes, all Python tests and the GLM checks (`glm53_selftest.sh`, `glm53_park_check.py`,
  `glm53_server_check.sh`) green on the CPU engine.
- [x] E1 (merged in d8d45a7, 45eae16, d008e81 and 7a4d94b) The code problems the docs' fact-checkers found, in three
  groups each reproduced by tests, fixed and reviewed: stopping (Ctrl+C, a closed window, SIGTERM and a systemd or
  Docker stop end the engine with QUIT, so it saves its expert profile; `./setup.sh` execs the server; 120 s for the
  engine to end; any start progress counts for the start watchdog), install (RAM in GiB on both sides, the driver
  the chosen CUDA toolkit needs, `--yes` never asks, the Settings view through Docker's config link, a warning for
  a keyless server on every interface), and the server (a start line saying whether calls are read from special
  tokens, the page's Thinking On sent, `response_format` patterns and `logit_bias` checked before generating, the
  free-VRAM check and the Monitor on every card when the config names none).
- [x] The repository is the owner's personal project: the upstream project's public-project files (the issue forms,
  the pull request template, SECURITY.md's reporting process, the funding link, AGENTS.md's contribution rules) are
  removed; what SECURITY.md said about the server's protections is in [DETAILS.md](DETAILS.md#security).

### G. The real chat template (napster-ai, 2026-10-20)

`tools/glm53_template_probe.py` read the real UD-IQ3_XXS GGUF's tokenizer and chat template. What it showed and what
the server does with it:

- [x] G1 `serve/chat_template.jinja` is the real template, byte for byte (10648 characters); it was GLM-4.7-Flash's.
  Nearly every serve test renders with it, `test_glm_engine_protocol` too. The capability probe and the checkpoint
  read it as before. Its calls and tool results are written as the parser and `text_tool_results` expect.
- [x] G2 Thinking off. The template has no `enable_thinking` switch: the prompt always ends `<|assistant|><think>`, so
  `none`, Anthropic's disabled thinking and `"anthropic_thinking": "on_request"` still thought. The server now
  closes it (`server.close_thinking`): the prompt ends `<|assistant|><think></think>`, as the template writes a past
  turn without thinking; a forced call's opening follows; the checkpoint stays before the generation prompt; the
  template still gets `enable_thinking`, and a template that honours it is left as it renders.
- [x] G3 Levels. The template writes `<|system|>Reasoning Effort: Low|High|Max` first: `low` and `high` as given,
  anything else Max. The server maps none/off/disabled/false to off, minimal/low to low, medium/high to high,
  xhigh/max/maximum to max, and passes nothing when no level is given; Anthropic's `budget_tokens` is low under
  2048, high under 8192, else max. The shared settings, the run config and the Settings view take none, low,
  medium, high and max. A change of level changes the prompt's first turn, so the conversation is read again.
- [x] G4 The web page's Thinking is Off / Low / High / Max (default Max); it was Off / On, and On sent `high`, less
  than the default. "Use for other apps too" still shares only Off (#278).
- [x] G5 `clear_thinking` is false by default: every earlier turn keeps its thinking, so the next turn's prompt starts
  with the prompt and the reply as the model wrote it. The docs said the opposite; DETAILS says what it costs in
  context and how to send `clear_thinking` true.

## Known risks to check on napster-ai

The real model has run on napster-ai (2026-10-22/23, [GLM53_PORT.md](GLM53_PORT.md)): through this server it
answered a chat, a question with thinking off and a tool call correctly. The rest is tested on the CPU engine with
small random models.

- The real GLM-5.3 template is known now (`tools/glm53_template_probe.py` on the real UD-IQ3_XXS GGUF, napster-ai,
  2026-10-20). `serve/chat_template.jinja` is that template, byte for byte, and the tests use it.
- Token types are known too: in the real GGUF GLM's tags (`<think>`, `<tool_call>`, `<arg_key>`, ...) are type 4
  and the turn tokens type 3 (the same probe), so the server reads calls and `</think>` from the special tokens
  only. The end-of-turn ids are 154820, 154827 and 154829 (`<|endoftext|>`, `<|user|>`, `<|observation|>`), and the
  server's tokenizer agreed with llama.cpp's on 21 of 21 test strings.
- How the model thinks at Low, High and Max, and how it answers with the thinking closed by the server: not run yet.
- Whether the next turn's prompt really starts with the model's own reply tokens: the template writes the answer
  stripped of spaces and newlines at its ends, and a reply tokenized again can differ from the tokens the model
  wrote. Where it differs, the engine goes on from the last checkpoint instead.
- Whether GLM writes tool calls inside its thinking.
- The start on napster-ai's SATA SSD: how long the mapping and the layer upload take before the first warm-up line
  (the start watchdog allows 900 s without progress), and how long the warm-up of 110-146 GB takes.
- The 4-bit model's RAM: about 52 GiB left beside its experts on 192 GB, shared by Linux, the server, Open WebUI and
  the parked conversations (8192 MiB). Its experts are derived from its download, not counted in its file.
