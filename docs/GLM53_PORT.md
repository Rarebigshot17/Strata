# Porting Strata to GLM-5.3-Flash

This branch (`glm53-flash`) turns Strata from a Qwen3.8-Flash-Next engine
([Niko1221/Strata](https://github.com/Niko1221/Strata), the upstream project) into a GLM-5.3-Flash engine. It keeps
how Strata works: every routed expert in system RAM, the most-used experts cached in VRAM, the rest computed on
the CPU, one model compiled for speed. GLM-5.3-Flash replaces Qwen3.8-Flash-Next. Nothing of the Qwen model
stays.

Target machine: napster-ai (Xeon E5-2699 v4 with AVX2, 192 GB DDR4 planned and 91 GiB for now, RTX 3090 Ti 24 GB +
RTX 3080 Ti 12 GB + RTX 3080 10 GB, all sm_86, Linux). CUDA only: the SYCL (Intel) engine and HIP (AMD) are removed
on this branch.

Estimated effort when the port was planned: about 5-6 months of engineering. Code is written off the machine and
tested on napster-ai by hand, so every milestone ends with commands to run and output to paste back.

**The real model has run** on napster-ai, 2026-10-22/23: the 3-bit UD-IQ3_XXS file, on a Xeon E5-2699 v4 (22 cores,
AVX2) with 184 GiB of RAM (MemTotal; 6 x 32 GB, one of the CPU's four memory channels empty), an RTX 3090 Ti 24 GB,
an RTX 3080 Ti 12 GB and an RTX 3080 10 GB (sm_86, driver 580.95.05, CUDA 13.2, gcc 14.2), the model on SSDs. One
run each unless it says otherwise:

- **It answers correctly** through the server (step 8): a greeting with its reasoning apart, "391" for 17 x 23 with
  thinking off, and a `get_weather` call for "What is the weather in Paris right now?".
- **It predicts text as well as llama.cpp.** On 1,953 tokens of text (`tools/glm53_ref/text.txt`, the start of the
  README), the CPU reference and llama.cpp's CPU path give a mean negative log-likelihood of 1.7211 and 1.7238 nats
  per token: a paired difference of -0.0028, 95% interval -0.0137 to +0.0091 (`tools/glm53_nll.py`). Their logits
  differ by a mean KL of 0.03 to 0.05, as llama.cpp's own CPU and GPU paths do (0.0375): every product rounds its
  activations to 8 bits, and a difference of 1e-7 flips that rounding for a few values ("Checking against
  llama.cpp"). The GPU engine was compared on 205 tokens only: +0.0217 nats per token against llama.cpp's CPU
  path, 95% interval -0.0156 to +0.0615, so no difference shown and none ruled out below about 0.06.
- **Decode speed** through the server, at `3fea91f`: 14.3 to 16.7 tokens/s on three fresh chat prompts (56% to 70%
  of the experts used read from VRAM), 17.0 to 17.7 on the second to fourth of four prompts started from the same
  expert profile. The same server at `04d51fc`, before the three decode commits: 6.5 tokens/s. Calculated from
  the measured RAM bandwidth (39.5 GB/s to read, about 27 GB/s through the expert kernels) and about 1 GB of
  experts read from RAM per token, this machine tops out near 18 to 20 tokens/s.
- **Start-up** with 131,072 tokens of context: the server was ready about 2 minutes after it started, with most of
  the shards already in the page cache; the warm-up read 103.6 of the experts' 109.9 GB at 1.67 GB/s. A start
  straight after a reboot (every expert read from the SSDs) was not timed. The layers went 0-22, 23-34 and 35-44
  (with the output head) to the three cards, and the expert cache took 31.1 GB (3,422 experts over 42 layers).
- **Not run yet on the real model:** a context longer than about 2,000 tokens (so the indexer's selection of pools
  has not been compared), long agent sessions, several clients at once, the prompt path's speed beyond short
  prompts, the 4-bit file, the systemd unit and the Docker image.

The small-model checks before that (2026-10-20 and 2026-10-22):

- the builds: llama.cpp (step 1) and this branch with CUDA (step 2);
- the CPU reference against llama.cpp on small random models (step 3): all models PASS, KL 0;
- the GPU check on the same models (step 4), three runs at `fa74d8a`: every model's decode and prompt path passed
  against the CPU reference and against llama.cpp, and conversation parking passed. Against the CPU reference the
  float models differ by under 1e-4 (relative), q4_0 by 0.016 to 0.027 (limit 0.05, mean KL at most 1.4e-6: the
  CUDA rounding llama.cpp's own GPU shows too, "Checking against llama.cpp") and q8_0 by at most 0.011. The small
  models' two layers went to the RTX 3090 Ti and the RTX 3080; the RTX 3080 Ti ran them alone in a later run
  (2026-10-22, every check PASS). Earlier runs found two bugs in the check, both fixed: the prefill check's CPU
  reference had room for the prompt only (5890fde), and its 2% limit was below the CUDA rounding on q4_0;
- the inspector and the template probe on the real 3-bit file (step 5).

Before that, without a GPU: conversation parking and the whole server on the CPU engine, and the Python and C++
tests. The memory figures on this page come from the real files where they say so (the repository's listing, the
inspector); the rest are calculated or derived, and say which.

## The model

From the transformers config defaults and llama.cpp's `src/models/glm5-next.cpp`. `tools/glm53_inspect.py` checks
each value against the real GGUF before code depends on it; what it found in the 3-bit file is under "The real file"
below.

| | GLM-5.3-Flash | Qwen3.8-Flash-Next (the upstream engine's model) |
|---|---|---|
| Size | 320B total, 18B active | 125B total, 6B active |
| Layers | 45: KDA linear attention in 34, MLA sparse attention in 11 (every 4th: 3, 7, ..., 43) | 48: GDN linear attention in 36, GQA sparse attention in 12 |
| Hidden size | 4096 | 2560 |
| Linear attention | KDA: 64 heads x 128, decay per channel, separate q/k/v short convs, low-rank gates | GDN: 16 key / 48 value heads, decay per head, one fused conv |
| Sparse attention | MLA without RoPE (latent 512, head dim 256, 64 heads); indexer of 32 heads that pools keys over 4 tokens (the real file's `kpool`), top 2048 | GQA with RoPE (24 q / 2 kv heads); indexer of 4 heads that pools over 4 tokens, top 2048 |
| Residual | mHC: 4 streams, Sinkhorn mixing (20 iterations) | 4-stream gated residual |
| Experts | 288 per layer, top 8, sigmoid routing with bias, x2.5 scale, 1 shared expert, layers 0-2 dense, SwiGLU clamp 10 | 512, top 10, softmax routing, gated shared expert, every layer MoE |
| Vocab | 154,880 | 248,320 |
| Context | 1,048,576 | 262,144 |
| Extras | 1 MTP layer (not used); thinking on by default | MTP layer, n-gram table, vision |

## Memory on napster-ai

The routed experts are about 304B parameters (288 experts x 3 matrices x 4096 x 2048 x 42 MoE layers). The sizes
below come from napster-ai on 2026-10-20: the downloads from the repository's listing, the 3-bit experts from
`tools/glm53_inspect.py` on the 3-bit file: 109.9 GB of routed experts in the 45 layers the engine runs, 7.7 GB of
everything else (on the GPUs), and 2.8 GB for the MTP layer (`blk.45`), which the engine does not load. (The
inspector's first run counted 112.5 GB of experts and 7.9 GB of the rest, with the MTP layer in them.) The 4-bit
experts are not measured: they are derived as its download less the 3-bit file's 7.7 GB and 2.8 GB, which takes the
4-bit file's other tensors and MTP layer as the same size as the 3-bit's (if they are bigger, its experts are
smaller). The RAM left is calculated from these.

| Option | Files | Download | Experts in RAM | RAM left beside them on 192 GB (188 GiB reported) |
|---|---|---|---|---|
| 3-bit | UD-IQ3_XXS, 4 files | 120.4 GB (listing) | 109.9 GB = 102.4 GiB (inspector) | about 86 GiB |
| 4-bit | UD-IQ4_XS, 5 files; UD-Q4_K_XL (6 files, 199.7 GB) when the repository has no UD-IQ4_XS | 156.8 GB (listing) | about 146 GB = 136 GiB (derived) | about 52 GiB |

The plan had calculated about 117 GB for the 3-bit experts (IQ3_XXS, 3.06 bits, for every one) and about 162 GB for
the 4-bit ones (IQ4_XS, 4.25 bits); the 3-bit file's experts mix IQ2_S, IQ3_S and IQ4_XS ("The real file" below),
and which types the 4-bit file holds is not read yet. The repository has no plain IQ4_XS file.

`setup.py` uses the same figures (its `MODELS` table, in GB) and prints what is left beside the experts in GiB,
taken from the RAM Linux reports (`/proc/meminfo`), which is a few GiB under the installed 192 GB: setup's tests
take 188 GiB. With less than 40 GiB left it calls a size tight and gives parked conversations 4096 MiB instead of
8192 (`--park-mib`); below 12 GiB it asks before it installs. With 192 GB it therefore recommends the 4-bit model
(about 52 GiB left), and keeps the 3-bit one when you choose it (about 86 GiB left). The 4-bit's margin over the
40 GiB rests on the derived figure. With 160 GB (about 157 GiB) it would recommend the 3-bit: the 4-bit would leave
about 21 GiB. For UD-Q4_K_XL setup takes the experts as the same share of its files as for UD-IQ4_XS.

For now napster-ai has 91 GiB of RAM (at the first run, 2026-10-20); 192 GB is planned. Neither size fits on 91 GiB:
setup wants at least 12 GiB left beside the experts, so about 114 GiB of RAM for the 3-bit size (142 GiB to call it a
fit) and about 148 GiB for the 4-bit size (176 GiB). On 91 GiB setup says the experts do not fit and asks before it
installs. The steps that run the real model (6-8) need the RAM in place: with less RAM than the experts, Linux reads
them from the disk while the model runs, which is very slow or does not start at all. They also need the Qwen service
stopped, as step 4 does, since it holds most of the VRAM and RAM while it runs. Step 5's download and inspector need
only the disk; the inspector prints this machine's RAM beside the experts it counts in the file.

The MLA cache stores a 512-wide latent per token in each of the 11 MLA layers. The engine keeps it in f16, as
llama.cpp does by default: 11 KB per token (an 8-bit cache would halve that). Each layer's indexer adds one pooled
128-wide key per 4 tokens (`attention.indexer.kpool` is 4 in the real file), kept in fp32: 128 bytes per token per
layer, 1.4 KB per token over the 11 layers. Together 12.4 KB per token, 1.7 GB of VRAM at 131,072 tokens and 13.3 GB
at 1M; `setup.py` uses this figure. On each card the prompt path's work area (its arena) also holds the pool scores
of a 256-token chunk (the default `--prefill-chunk`; 2 x 256 x 4 bytes per pool), so it grows with the context too:
the arena is about 0.32 GB per card up to about 200K tokens, 0.49 GB at 512K and 0.75 GB at 1M, beside the prompt
path's other buffers. The KDA layers hold a fixed-size state (64 heads x 128 x 128 per layer, about
0.15 GB over the 34 layers in fp32). All calculated from the engine's buffers (`src/glm/engine.cu`,
`src/glm/engine_prefill.cu`).

## How the engine is built

The Qwen engine's code was specialised to that model at every level (sizes compiled into the kernels, two layer
types, captured CUDA graphs built around them). Rewriting it in place would have meant changing every one of those
pieces with no way to test them in between. So the GLM engine is new code under `src/glm/` and
`include/strata/glm/`. It keeps Strata's design (every routed expert in RAM, the most used ones cached in VRAM,
the rest computed on the CPU) and reuses Strata's generic parts: the GGUF reader, the quantized GEMV kernels
(`native_mmvq`) and the server's engine protocol. The Qwen engine is deleted (M7).

Every piece is checked against llama.cpp, which has run GLM-5.3-Flash since its `glm5-next` support landed:

1. **CPU reference** (`strata-glm-ref`, `src/glm/cpu_ref.cpp`): the whole model on the CPU, one token at a time,
   with ggml-cpu's own dot products. It matches llama.cpp on small random GLM-5.3-Flash models to float rounding
   (`tools/glm53_selftest.sh`, which runs without a GPU or a download). On the real model it is only meant as the
   reference; its speed there is not measured.
2. **GPU engine** (`strata-glm`): each CUDA kernel is to be checked against the CPU reference, then whole-model
   logits against llama.cpp on the real GGUF (`tools/glm53_compare.py`). Written; on small random models its
   decode matches both on napster-ai's cards (step 4); not yet run on the real GGUF.

| # | Milestone | Status | Test |
|---|---|---|---|
| M0 | Remove SYCL; GGUF inspector; llama.cpp reference harness | done | inspector output, reference logits |
| M1 | CPU reference: config and tensors from the GGUF, mHC, KDA, MLA with the k-pool indexer, MoE with sigmoid routing and SwiGLU clamp | done | `tools/glm53_selftest.sh`: all variants PASS |
| M2 | CUDA: dense projections, mHC, KDA decode, MLA decode with the indexer, routing; layer split over the GPUs | run: on small random models (napster-ai, 2026-10-20) decode matched the CPU reference and llama.cpp within the CUDA rounding, on one card and split over two; on the real 3-bit file (2026-10-22/23) split over all three cards, 13.16 tokens/s one token at a time on 205 tokens of text, and it predicts that text as well as llama.cpp's CPU path within 0.06 nats per token (205 tokens, "The real model has run") | `tools/glm53_gpu_check.sh`; logits vs llama.cpp |
| M3 | Experts: all in RAM (the mapped GGUF), VRAM cache of the most used, CPU for the rest; the warm-up that reads them into the page cache before READY | run on the real 3-bit file (napster-ai, 2026-10-22/23): 31.1 GB of experts in VRAM at 131,072 tokens of context, the cache follows the conversation (`--cache-decay 0.95`, `--max-swaps 32`, copies in the background); 14.3 to 17.7 tokens/s through the server with 56% to 70% of the experts used read from VRAM | logits unchanged; tokens/s |
| M4 | Prompt processing: batched projections, chunked KDA, batched MLA (causal below 2,051 tokens of context, each query's own selection past that), experts streamed through VRAM | written; on small random models (napster-ai, 2026-10-20, chunks of 16 tokens) it matched the CPU reference within the CUDA rounding; on the real file it read the server's short prompts (up to 163 tokens), its speed beyond that not measured | prompt tokens/s; logits unchanged |
| M5 | Server: Strata's full server (`serve/server.py`: OpenAI, Anthropic and Responses APIs, the web app, MCP, structured output, metrics) on the GLM engine, with GLM's chat template, tool calls (`<arg_key>`/`<arg_value>`) and thinking on or off | ported and reviewed ([GLM53_SERVER_PORT.md](GLM53_SERVER_PORT.md)); tested on the CPU engine (`tools/glm53_server_check.sh`, the Python tests); on the real file (napster-ai, 2026-10-22) it answered a chat, a question with thinking off and a tool call correctly; Open WebUI and the agents not tried yet | Open WebUI and an agent work end to end (step 8) |
| M6 | Several clients: conversation parking and checkpoints | written, checked on the CPU engine (`tools/glm53_park_check.py`); batching and MTP drafting dropped | speed for 1, 2 and 4 clients |
| M7 | Remove the Qwen engine and HIP; docs | done: the Qwen engine, HIP and the Qwen docs are removed; the docs describe GLM-5.3-Flash | |

### Several clients on one engine

The agents, the chat and the bot take turns on one engine: requests queue in the server, one runs at a time, in the
order they arrived. The KDA layers' state is a running sum that cannot be rewound, so the engine can only continue
a state whose tokens are the start of the new prompt. To keep each client's conversation anyway, the engine
(`src/glm/serve_loop.cpp`) keeps several:

- **Parking.** When a prompt starts another conversation, the current one's whole state is copied to host RAM.
  A later prompt that continues it copies it back and reads only the new tokens.
- **Checkpoints.** The server asks for a checkpoint where the reply begins (`cp=` on the GEN line: the prompt
  without the `<|assistant|><think>` generation prompt). The next turn renders the previous reply again without
  its reasoning, so it parts from the engine's state there; it restarts from the checkpoint and reads the reply
  and the new message, not the whole conversation. A regenerated reply restarts there too.
- **Memory.** `--park-mib` (default 8192) caps what parked conversations and checkpoints use; the least recently
  used conversation goes first. A parked conversation is the KDA layers' state (about 0.15 GB) plus 12.4 KB per
  token, so a 100K-token conversation is about 1.4 GB; a checkpoint is about 0.15 GB (calculated). Each copy
  crosses PCIe 3.0; how long it takes is not measured yet.

Nothing about the answers changes: `tools/glm53_park_check.py` checks every resumed reply against a fresh
engine's, token for token.

Small random GLM-5.3-Flash models for tests that need no download: llama.cpp's
`test-llama-archs -a glm5-next -o DIR` writes a small `glm5-next-moe.gguf` with 2 layers (one MLA, one KDA).
`tools/glm53_tiny_variant.py` switches on the paths it leaves off (SwiGLU clamps, expert weight scaling, an
indexer top-k small enough to drop pools).

Two things found while writing the reference, both checked against llama.cpp's code:

- The KDA decay applies per **key** channel: `S[i][:] *= exp(g[i])`. That is what ggml's fused
  `gated_delta_net` op does, and llama.cpp uses that op by default. Its unfused one-token graph
  (`build_delta_net_autoregressive`) broadcasts the decay over the other axis.
- Below 2,048 tokens of context the indexer selects every pool, so attention is plain causal there. The
  indexer only changes results on longer contexts, which is why the test variants shrink its top-k.

The real file, as `tools/glm53_inspect.py` read the 3-bit GGUF (UD-IQ3_XXS) on napster-ai on 2026-10-20, differs
from what the plan took in these:

- `attention.indexer.kpool` is 4, not 16: the indexer pools its keys over 4 tokens, and picks 512 pools for its
  2,048 tokens. The engine reads the value from the file. The pooled keys take 4 times the VRAM the plan counted
  ("Memory on napster-ai").
- `attention.indexer.types` is missing. llama.cpp then gives every layer its own (full) indexer, and so does the
  engine.
- `attention.indexer.kpool_select_tail` is missing. llama.cpp then selects the incomplete tail pool (its default is
  true), and so does the engine. The small test models always carry the key (llama.cpp's model saver writes it), so
  the checks could not see which default the engine took.
- `block_count` is 46 with `nextn_predict_layers` 1: `blk.45` is the MTP layer. The engine runs the 45 layers before
  it and does not load it, so its Q2_K experts, the only Q2_K tensors in the file, need no kernel.
- The routed experts of the 45 layers are not IQ3_XXS throughout, as the plan took them: they are IQ2_S (gate and up,
  63.5 GB) and IQ3_S (down, 42.6 GB), with 3.9 GB of IQ4_XS. The GPU kernels take each. The MTP layer's experts are Q2_K
  and Q3_K. The inspector counted 109.9 GB of the 45 layers' experts, 7.7 GB of everything else and 2.8 GB for the MTP
  layer: the 120.4 GB the repository lists for the file. The plan had calculated about 117 GB of experts ("Memory on
  napster-ai").

### Checking against llama.cpp

The indexer's precision differs. Strata keeps the indexer's keys, gates and pooled keys in fp32, in the CPU reference
and in the GPU engine. llama.cpp keeps them in its KV cache type, f16 by default (`-ctk`). When two pools score nearly
the same, the two engines can then select different pools. With llama.cpp's default cache the selftest failed that way
on 5 of seeds 1-100, v1, v2 and v1-bare together (mean KL 4e-5 to 1.5e-4), and on 2 more a Q4_0 near tie failed it
(CPU builds of both on an x86 machine with AVX-512, not napster-ai, 2026-10-20). So the selftest and the GPU check run
llama.cpp's reference on the CPU with `-ctk f32 -ctv f32`. That keeps llama.cpp's MLA latent in f32 too, where Strata
rounds it to f16 as llama.cpp's default cache does. On the same seeds no selection then differs, and the logits differ
by at most 0.00025 (mean KL 4e-10) on the unquantized models and 0.0042 (mean KL 2.3e-7) on Q4_0. Two top tokens swap
places in 27 of the 100 runs, 25 of them Q4_0's, always within 0.0017 of each other on both sides; `glm53_compare.py`
counts such a row as a tie. With both changes the selftest passed all 100 seeds.

The GPU engine is held to one rule (`include/strata/glm/check.hpp`), against the CPU reference (`strata-glm
--check-cpu`, with or without `--prefill`) and against llama.cpp (step 4). Its relative difference is a row's largest
logit difference over that row's largest logit, for the worst row. Before a product with quantized weights the GPU
rounds the activations to q8_1, as llama.cpp's CUDA backend does. On napster-ai (2026-10-20, 36 seeds of step 4's v1,
Q8_0 and Q4_0 models, on one RTX 3080) that gave Q4_0 a relative difference of up to 0.032 (median 0.023) and a mean
KL up to 1.3e-6, and top-1 agreement fell to 37 of 40. llama.cpp's own CUDA backend differed from its CPU as much
(0.034, 37 of 40). Q8_0 reached 0.0077 (llama.cpp's own 0.014), v1 0.0001. That is rounding, not a bug, but the old
limits (0.02 and 98% top-1) failed it: Q4_0 at 0.0233 and 0.0305 when decoding and 0.027 on the prompt path, in GPU
check runs on napster-ai the same day. The rule now asks:

- a relative difference within 0.05 when some weights are quantized (1.45 times llama.cpp's worst), and within
  0.005 when all are F32 or F16, so that a 1% error there fails. The prompt path multiplies F16 weights in f16 (F32
  ones in f32) and keeps the attention's probabilities in f16. f16 products moved the logits by up to 0.001 on the
  CPU (strata-glm-ref on F16 copies of 20 seeds' v1, an x86 machine with AVX-512, 2026-10-20). The F32 models'
  prompt path has no GPU measurement yet;
- a mean KL within 1e-5, 7.6 times the rounding's worst. Bugs land far above it: the indexer's select_tail bug gave
  7.6e-4, a pool selected differently 4e-5 to 1.5e-4;
- no NaN or infinity.

Top-1 agreement is printed but not counted. When two engines' logits differ by at most d, they can only pick tokens
that are less than 2d apart, so the count adds nothing to the limit on d. It counts near ties, and the small models
have many. Against llama.cpp the two CPU engines' own difference adds to the GPU's: llama.cpp with an f32 cache and
strata-glm-ref differed by up to 0.015 on Q4_0 (seed 21), 0.014 on Q8_0 and 0.0008 on v1 (seeds 1-100, the same x86
machine). Added up that is at most 0.047, 0.022 and 0.0009 (calculated: two worst rows of different runs), within
the same limits; Q4_0's comes close to its 0.05. `glm53_compare.py`'s defaults would fail it: these models' largest
logits are about 0.3, so at 0.032 two tokens 0.02 apart can swap, which is no tie. The limits come from the small
random models.

On the real file the logits cannot be compared this way: they differ by far more than any rounding the small models
show, between any two paths. On step 6's 205 tokens (napster-ai, 2026-10-22/23) the mean KL was 0.052 between
Strata's CPU reference and llama.cpp's CPU path, 0.036 between Strata's GPU engine and that path, and 0.0375 and
0.0165 between llama.cpp's own CPU path and its runs with the GPUs (all three cards, and two cards with two layers'
experts moved to the CPU). Layer dumps of both CPU paths (`--dump-layers` on both tools,
`tools/glm53_layers_compare.py`) found where it starts. At the first token layer 0's attention half agreed to
3.2e-7; the mixer before its FFN, whose Q8_0 product rounds the activations to 8 bits, turned that into 9.3e-6;
and the dense FFN turned 2.6e-6 at its input into 1.4e-2 at its output. A float64 recomputation of that FFN from
the file's Q6_K weights reproduced each engine's output from its own input to 3e-7. One of the 4,096 values that
Q8_K rounding gives the input differed, and that alone made the whole 1.4e-2: the FFN's output is small next to its
input (RMS 0.0084 against a largest input of 0.55), and Q8_K rounding itself moves it 2.1% from exact arithmetic.
Through 45 layers such flips add up. So step 7 compares how well the engines predict the text, not their logits
(`tools/glm53_nll.py`): on 1,953 tokens Strata's CPU reference and llama.cpp's CPU path gave a mean NLL of 1.7211
and 1.7238 nats per token (paired difference -0.0028, 95% interval -0.0137 to +0.0091).

Step 7 sees no selection difference: up to 2,051 tokens of context the indexer selects every pool, so the precision
of its keys cannot change what is selected, and step 6's text is shorter. Step 6 runs llama.cpp with its default f16
cache, which keeps the latent as Strata does.

## What to run on napster-ai

Each step prints a few lines to paste back. Steps 1-4 need no model download. Steps 0-5 ran on napster-ai on
2026-10-20, steps 3 and 4 pass at `fa74d8a`, and steps 6-8 ran on 2026-10-22/23 ("The real model has run"). Run them
from the Strata folder.

0. **The checkout, the PC check and the tools the steps use.** `./setup.sh --check` checks the GPUs, the driver, the
   RAM, the CPU and the disk, and makes the `.venv` (it asks for sudo when Python's venv is missing); it stops before
   it installs anything else, so the Python packages the checks need (numpy, pyyaml, tqdm, regex, and cmake 3.24+)
   come from `requirements.txt`. Steps 1 and 2 compile CUDA code with CMake, which takes the `nvcc` that `CUDACXX`
   names, else the first one on `PATH` (a CUDA toolkit, 12.0 or newer). When `/usr/local/cuda` exists, the lines
   below point both at it: an older system nvcc can come first on `PATH` and fail CMake's CUDA compiler check with
   a newer gcc. On napster-ai Ubuntu's `/usr/bin/nvcc` (CUDA 12.2) did that with gcc 14; `/usr/local/cuda` is 13.2.
   Without `/usr/local/cuda` the nvcc on `PATH` is used; when there is none, [INSTALL.md](INSTALL.md) ("The
   toolkit") installs one. A build folder keeps the compiler CMake found first: after changing it, delete
   `~/llama.cpp/build` and `build` before steps 1 and 2. Setup (step 8) picks the newest toolkit itself.

   ```bash
   git clone -b glm53-flash https://github.com/Rarebigshot17/Strata ~/Strata && cd ~/Strata
   ./setup.sh --check
   .venv/bin/python -m pip install -r requirements.txt
   source .venv/bin/activate
   if [ -x /usr/local/cuda/bin/nvcc ]; then
       export CUDACXX=/usr/local/cuda/bin/nvcc PATH=/usr/local/cuda/bin:$PATH
   fi
   nvcc --version
   ```

1. **Build llama.cpp** (CUDA, pinned commit) with the reference tool and the small-model generator:

   ```bash
   tools/glm53_ref/build.sh ~/llama.cpp
   ```

2. **Build this branch** for the three Ampere cards (sm_86):

   ```bash
   cmake -B build -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
   cmake --build build -j"$(nproc)" --target strata-glm strata-glm-ref
   ```

3. **CPU reference vs llama.cpp** on small random models (paste the five PASS/FAIL lines). It needs no GPU: the
   script hides the GPUs from llama.cpp's CUDA build, so it runs while another program fills them. llama.cpp runs
   with an f32 cache ("Checking against llama.cpp"). A line that ends in `(100.00% with 1 tie)` passed: two tokens
   as likely as each other swapped places. Each run makes new random models; the first line gives the seed, and
   `SEED=N` before the command runs the same models again:

   ```bash
   PYTHONPATH=~/llama.cpp/gguf-py tools/glm53_selftest.sh ~/llama.cpp/build build
   ```

4. **GPU engine vs the CPU reference and llama.cpp** on the same small models, then conversation parking, with the
   layers split over the visible cards (paste the whole output). It needs the GPUs free: stop any other program that
   holds them first (`nvidia-smi` lists them). On napster-ai that is the Qwen service, started again afterwards.
   llama.cpp's reference runs on the CPU here too, with step 3's f32 cache: on CUDA, flash attention converts an f32
   cache to f16, and that path has not been checked:

   ```bash
   sudo systemctl stop strata-iq3s.service
   PYTHONPATH=~/llama.cpp/gguf-py tools/glm53_gpu_check.sh ~/llama.cpp/build build
   sudo systemctl start strata-iq3s.service
   ```

   Each model prints three lines that end in PASS: `check-cpu:` (40 tokens one at a time against the CPU
   reference), `prefill: check-prefill:` (the prompt in chunks, then 4 decoded tokens) and `vs llama.cpp:`. All
   three give the worst relative difference and the mean KL; the first two print the limits, the third names them
   when it fails and rounds the mean KL to 0.00000. The limits are 0.05 on q4_0 and q8_0 and 0.005 on the three F32
   models, and 1e-5 for the mean KL. Expect up to about 0.03 on q4_0 (0.04 against llama.cpp), less on q8_0, and
   0.001 or less on the F32 models. Top-1 agreement below 40/40 on q4_0 is rounding and passes ("Checking against
   llama.cpp"). The last line is the parking check's. It passed three runs on napster-ai at `fa74d8a`. The small
   models have two layers, so they use two cards (there: the RTX 3090 Ti and the RTX 3080); to run them on the
   RTX 3080 Ti as well, run the script once more with `CUDA_VISIBLE_DEVICES=1` in front.

5. **The real model.** Pick the 3-bit or the 4-bit GGUF (see "Memory on napster-ai"; Unsloth re-uploaded them on
   2026-10-05 with `glm5-next` metadata). List what the repository has, then download one quant's shards and
   check them against what the engine assumes (paste the whole output):

   ```bash
   pip install -U huggingface_hub
   python -c "from huggingface_hub import list_repo_files as l; print('\n'.join(f for f in l('unsloth/GLM-5.3-Flash-GGUF') if f.endswith('.gguf')))"
   # 3-bit, 120.4 GB (the repository's listing on 2026-10-20):
   hf download unsloth/GLM-5.3-Flash-GGUF --include "*UD-IQ3_XXS*" --local-dir ~/models/glm53
   # or 4-bit, 156.8 GB (the same listing): the UD-IQ4_XS shards (the repository has no plain IQ4_XS)
   hf download unsloth/GLM-5.3-Flash-GGUF --include "*UD-IQ4_XS*" --local-dir ~/models/glm53-q4
   python tools/glm53_inspect.py ~/models/glm53/<first .gguf shard>
   ```

   The inspector flags an expert type the GPU kernels cannot read only where a layer the engine runs uses it; the MTP
   layer (the last block, `blk.45` in the 3-bit file) is listed apart, since Strata does not load it. In the layers the
   engine runs, the 3-bit file's experts are IQ2_S, IQ3_S and IQ4_XS (its inspector output on napster-ai), and the
   kernels take each; its Q2_K and Q3_K experts are only in the MTP layer. On 192 GB (188 GiB reported) the 4-bit file
   leaves about 52 GiB of RAM beside its experts and the 3-bit file about 86 GiB (calculated from the figures in "Memory
   on napster-ai"; the 4-bit's are derived). Steps 6-8 need the RAM in place and the Qwen service stopped ("Memory on
   napster-ai").

6. **Reference logits from llama.cpp** on the real model, on the CPU, for `tools/glm53_ref/text.txt` (7,000 bytes:
   the start of the README as it was at `dba806b`, 1,954 tokens). On napster-ai it took 2 minutes 51 seconds:

   ```bash
   CUDA_VISIBLE_DEVICES= ~/llama.cpp/build/bin/llama-strata-ref -m ~/models/glm53/<first shard> \
       -f tools/glm53_ref/text.txt -c 4096 -b 256 -ub 256 \
       --out ~/glm53-ref/text-ref.bin --ids-out ~/glm53-ref/text-ids.txt
   ```

   llama.cpp on the GPUs works too (`-ngl 99 --n-cpu-moe 40`), but its default split puts the last layers, whose
   experts it keeps in VRAM, on the last card: on napster-ai that ran the RTX 3080 out of memory. Put the largest
   card last (`CUDA_VISIBLE_DEVICES=2,1,0` there).

7. **Strata's GPU engine on the real model**, compared with step 6 by how well each predicts the text (paste both
   outputs). It needs the GPUs free, so stop the Strata server first:

   ```bash
   build/strata-glm -m ~/models/glm53/<first shard> --tokens-file ~/glm53-ref/text-ids.txt --ctx 4096 \
       --out ~/glm53-ref/text-strata.bin --expert-profile ~/glm53-ref/profile.bin
   python tools/glm53_nll.py ~/glm53-ref/text-ids.txt ~/glm53-ref/text-ref.bin ~/glm53-ref/text-strata.bin
   ```

   It passes when Strata's mean negative log-likelihood is at most 0.02 nats per token above llama.cpp's, at the
   upper end of the 95% interval: about a 2% higher perplexity at most. The logits themselves differ between any
   two paths, llama.cpp's own included, by a mean KL of 0.03 to 0.05 on this file, so `tools/glm53_compare.py`'s
   limits fail every pair ("Checking against llama.cpp"). The CPU reference passes this step (`build/strata-glm-ref
   ... -t 16` in place of `strata-glm`: 13 minutes on napster-ai, -0.0028, interval -0.0137 to +0.0091). The GPU
   engine has run it on 205 tokens only, too few for an interval that narrow; the full text has not run yet.

   This compares one token at a time. The VRAM expert cache starts empty and fills from what the text uses, so the
   first run is the slowest; `--expert-profile` saves the use counts for the next start.

8. **The server**, once step 7 passes. Setup writes the run config and the start script for the
   shards step 5 downloaded (`--gguf-dir` finds them by their quantization's name, so nothing is downloaded
   again), compiles the engine into `build/` with its own settings (it records the build in `build/BUILD.json`),
   and uses port 8101, which keeps clear of Open WebUI (3000), Portainer (8000) and the UPS dashboard (8080).
   Anything beyond this machine's loopback needs an API key:

   ```bash
   KEY="$(openssl rand -hex 24)"; echo "$KEY"          # keep it: the clients need it
   ./setup.sh --model UD-IQ3_XXS --gguf-dir ~/models/glm53 --context 131072 \
       --host 0.0.0.0 --api-key "$KEY" --no-browser --no-start
   ./run-glm-ud-iq3_xxs.sh
   ```

   For the 4-bit model: `--model UD-IQ4_XS --gguf-dir ~/models/glm53-q4`, and the script is
   `run-glm-ud-iq4_xs.sh`. The run script runs `serve/server.py --engine strata --config strata-glm-ud-iq3_xxs.json
   --port 8101` with the `.venv` Python; later, `./setup.sh` alone starts it too. The engine's output goes to
   `strata-glm-ud-iq3_xxs.log` in the Strata folder, the server's to the window. The server says
   `ready: http://127.0.0.1:8101/v1 ...` once the engine has read the experts (`warming the experts: ...` lines
   meanwhile). Paste the window's lines from `starting the engine` to `ready`, the output of
   `curl -s http://127.0.0.1:8101/health`, and the `done:` line of a first request:

   ```bash
   curl -s http://127.0.0.1:8101/v1/chat/completions -H "Authorization: Bearer $KEY" \
       -H "Content-Type: application/json" \
       -d '{"messages": [{"role": "user", "content": "Say hello in one sentence."}], "max_tokens": 512}'
   ```

   In Open WebUI add an OpenAI connection with that key and the address its container reaches the host at:
   `http://172.17.0.1:8101/v1` on Docker's default bridge, or `http://100.126.222.90:8101/v1` over Tailscale
   (which the agents and the Discord bot can use too). `--context` sets the longest conversation, up to 1,048,576
   tokens for the agents' 512K-1M conversations; each token of it takes 12.4 KB of VRAM in the MLA cache and the
   indexer (1.7 GB at 131,072, 13.3 GB at 1M, calculated), which the expert cache gives up.
   `./setup.sh --setup --context N` changes it later. The engine options are the run config's `"args"`: setup writes
   `-m`, `--ctx`, `--park-mib` and `--expert-profile` (the use counts, in the data folder `../Strata-GLM-data`), and
   keeps any other option you add there (`--vram-reserve`, `--max-swaps`, ...) when it runs again. To start with
   step 7's counts, copy `~/glm53-ref/profile.bin` over `../Strata-GLM-data/glm-ud-iq3_xxs.profile` before the
   first start.

## Removed on this branch

- `sycl/` (the Intel Arc engine), `docs/INTEL_ARC.md`, `docs/INTEL.md`, `docs/A770_SPEED_SPEC.md`, the Arc
  community benchmark results, `tools/test_setup_sycl.py`, and `setup.py --backend sycl` (M0).
- The Qwen engine, HIP (AMD) and their tests, C++ benches, reference data and community benchmark results
  (29056d3). The GLM engine's own unit tests are in `tests/glm` (ctest).
- In the server, everything that only the Qwen engine had: batch slots, vision, replicas, the VRAM command, slot
  save/restore (the route answers 501), MTP drafts, the speed projection; and `serve/glm_server.py`, the first GLM
  server, which `serve/server.py` replaced. The full list is under Decisions in
  [GLM53_SERVER_PORT.md](GLM53_SERVER_PORT.md).
- The Windows launchers (`START-HERE.bat`, `SETUP.bat`, `UPDATE.bat`, `BENCH.bat`; the GLM engine is POSIX-only)
  and the download of a prebuilt engine: setup compiles the engine on the PC.
- `tools/strata_mcp.py` (an MCP server that ran setup for an assistant) and its page `docs/MCP_SERVER.md`.
- The Qwen engine's and AMD's pages, the paper and the translated READMEs (a88c9f6).
