# Porting Strata to GLM-5.3-Flash

This branch (`glm53-flash`) turns Strata from a Qwen3.8-Flash-Next engine
([Niko1221/Strata](https://github.com/Niko1221/Strata), the upstream project) into a GLM-5.3-Flash engine. It keeps
how Strata works: every routed expert in system RAM, the most-used experts cached in VRAM, the rest computed on
the CPU, one model compiled for speed. GLM-5.3-Flash replaces Qwen3.8-Flash-Next. Nothing of the Qwen model
stays.

Target machine: napster-ai (Xeon E5-2699 v4 with AVX2, 192 GB DDR4, RTX 3090 Ti 24 GB + RTX 3080 Ti 12 GB + RTX
3080 10 GB, all sm_86, Linux). CUDA only: the SYCL (Intel) engine and HIP (AMD) are removed on this branch.

Estimated effort when the port was planned: about 5-6 months of engineering. Code is written off the machine and
tested on napster-ai by hand, so every milestone ends with commands to run and output to paste back.

**Nothing has run on a GPU or on the real model yet.** What has run: the CPU reference against llama.cpp on small
random models, conversation parking and the whole server on the CPU engine, and the Python and C++ tests. Every
memory figure on this page is calculated from the model's shapes, not measured, and there are no speeds yet.

## The model

From the transformers config defaults and llama.cpp's `src/models/glm5-next.cpp`. `tools/glm53_inspect.py` checks
each value against the real GGUF before code depends on it.

| | GLM-5.3-Flash | Qwen3.8-Flash-Next (the upstream engine's model) |
|---|---|---|
| Size | 320B total, 18B active | 125B total, 6B active |
| Layers | 45: KDA linear attention in 34, MLA sparse attention in 11 (every 4th: 3, 7, ..., 43) | 48: GDN linear attention in 36, GQA sparse attention in 12 |
| Hidden size | 4096 | 2560 |
| Linear attention | KDA: 64 heads x 128, decay per channel, separate q/k/v short convs, low-rank gates | GDN: 16 key / 48 value heads, decay per head, one fused conv |
| Sparse attention | MLA without RoPE (latent 512, head dim 256, 64 heads); indexer of 32 heads that pools keys over 16 tokens, top 2048 | GQA with RoPE (24 q / 2 kv heads); indexer of 4 heads that pools over 4 tokens, top 2048 |
| Residual | mHC: 4 streams, Sinkhorn mixing (20 iterations) | 4-stream gated residual |
| Experts | 288 per layer, top 8, sigmoid routing with bias, x2.5 scale, 1 shared expert, layers 0-2 dense, SwiGLU clamp 10 | 512, top 10, softmax routing, gated shared expert, every layer MoE |
| Vocab | 154,880 | 248,320 |
| Context | 1,048,576 | 262,144 |
| Extras | 1 MTP layer (not used); thinking on by default | MTP layer, n-gram table, vision |

## Memory on napster-ai

The routed experts are about 304B parameters (288 experts x 3 matrices x 4096 x 2048 x 42 MoE layers). All figures
below are calculated, not measured.

| Option | Expert quant | Experts in RAM | RAM left beside them on 192 GB (188 GiB reported) |
|---|---|---|---|
| 3-bit | UD-IQ3_XXS (IQ3_XXS: 3.06 bits) | about 117 GB (109 GiB) | about 79 GiB |
| 4-bit | IQ4_XS (4.25 bits); UD-Q4_K_XL when the repository has no IQ4_XS | about 162 GB (151 GiB) | about 37 GiB |

`setup.py` uses the same figures (its `MODELS` table, in GB) and prints what is left beside the experts in GiB,
taken from the RAM Linux reports (`/proc/meminfo`), which is a few GiB under the installed 192 GB: setup's tests
take 188 GiB. With less than 40 GiB left it calls a size tight and gives parked conversations 4096 MiB instead of
8192 (`--park-mib`); below 12 GiB it asks before it installs. On napster-ai it therefore recommends the 3-bit model,
and keeps the 4-bit one when you choose it. For UD-Q4_K_XL setup takes the experts as the same share of its files as
for IQ4_XS.

The MLA cache stores a 512-wide latent per token in each of the 11 MLA layers. The engine keeps it in f16, as
llama.cpp does by default: 11 KB per token, 1.5 GB of VRAM at 131,072 tokens and 11.8 GB at 1M (an 8-bit cache
would halve that). The indexer adds one pooled 128-wide key per 16 tokens per layer, under 0.5 KB per token even in
fp32. The KDA layers hold a fixed-size state (64 heads x 128 x 128 per layer, about 0.15 GB over the 34 layers in
fp32).

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
   logits against llama.cpp on the real GGUF (`tools/glm53_compare.py`). Written, not yet run on a GPU.

| # | Milestone | Status | Test |
|---|---|---|---|
| M0 | Remove SYCL; GGUF inspector; llama.cpp reference harness | done | inspector output, reference logits |
| M1 | CPU reference: config and tensors from the GGUF, mHC, KDA, MLA with the k-pool indexer, MoE with sigmoid routing and SwiGLU clamp | done | `tools/glm53_selftest.sh`: all variants PASS |
| M2 | CUDA: dense projections, mHC, KDA decode, MLA decode with the indexer, routing; layer split over the GPUs | written, not yet run on a GPU | `tools/glm53_gpu_check.sh`; logits vs llama.cpp |
| M3 | Experts: all in RAM (the mapped GGUF), VRAM cache of the most used, CPU for the rest; the warm-up that reads them into the page cache before READY | written, not yet run on a GPU | logits unchanged; tokens/s |
| M4 | Prompt processing: batched projections, chunked KDA, batched MLA (causal below 2,063 tokens of context, each query's own selection past that), experts streamed through VRAM | written, not yet run on a GPU | prompt tokens/s; logits unchanged |
| M5 | Server: Strata's full server (`serve/server.py`: OpenAI, Anthropic and Responses APIs, the web app, MCP, structured output, metrics) on the GLM engine, with GLM's chat template, tool calls (`<arg_key>`/`<arg_value>`) and thinking on or off | ported and reviewed ([GLM53_SERVER_PORT.md](GLM53_SERVER_PORT.md)); tested on the CPU engine (`tools/glm53_server_check.sh`, the Python tests) | Open WebUI and an agent work end to end (step 8) |
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
  used conversation goes first. A parked conversation is the KDA layers' state (about 0.15 GB) plus 11 KB per
  token, so a 100K-token conversation is about 1.3 GB; a checkpoint is about 0.15 GB (calculated). Each copy
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

## What to run on napster-ai

Each step prints a few lines to paste back. Steps 1-4 need no model download. None of these steps has been run
yet. Run them from the Strata folder.

0. **The checkout, the PC check and the tools the steps use.** `./setup.sh --check` checks the GPUs, the driver, the
   RAM, the CPU and the disk, and makes the `.venv` (it asks for sudo when Python's venv is missing); it stops before
   it installs anything else, so the Python packages the checks need (numpy, pyyaml, tqdm, regex, and cmake 3.24+)
   come from `requirements.txt`. Steps 1 and 2 compile CUDA code: `nvcc` (a CUDA toolkit, 12.0 or newer) must be on
   `PATH`; when there is none, [INSTALL.md](INSTALL.md) ("The toolkit") installs one.

   ```bash
   git clone -b glm53-flash https://github.com/Rarebigshot17/Strata ~/Strata && cd ~/Strata
   ./setup.sh --check
   .venv/bin/python -m pip install -r requirements.txt
   source .venv/bin/activate
   nvcc --version || export PATH=/usr/local/cuda/bin:$PATH
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

3. **CPU reference vs llama.cpp** on small random models (paste the four PASS/FAIL lines):

   ```bash
   PYTHONPATH=~/llama.cpp/gguf-py tools/glm53_selftest.sh ~/llama.cpp/build build
   ```

4. **GPU engine vs the CPU reference and llama.cpp** on the same small models, then conversation parking with the
   layers split over all three cards (paste the whole output):

   ```bash
   PYTHONPATH=~/llama.cpp/gguf-py tools/glm53_gpu_check.sh ~/llama.cpp/build build
   ```

5. **The real model.** Pick the 3-bit or the 4-bit GGUF (see "Memory on napster-ai"; Unsloth re-uploaded them on
   2026-10-05 with `glm5-next` metadata). List what the repository has, then download one quant's shards and
   check them against what the engine assumes (paste the whole output):

   ```bash
   pip install -U huggingface_hub
   python -c "from huggingface_hub import list_repo_files as l; print('\n'.join(f for f in l('unsloth/GLM-5.3-Flash-GGUF') if f.endswith('.gguf')))"
   # 3-bit, about 120 GB (calculated):
   hf download unsloth/GLM-5.3-Flash-GGUF --include "*UD-IQ3_XXS*" --local-dir ~/models/glm53
   # or 4-bit, about 170 GB (calculated): the IQ4_XS shards (or the UD-Q4_K_XL ones, whichever the list shows)
   hf download unsloth/GLM-5.3-Flash-GGUF --include "*IQ4_XS*" --local-dir ~/models/glm53-q4
   python tools/glm53_inspect.py ~/models/glm53/<first .gguf shard>
   ```

   The GPU kernels take every type these files use for experts (IQ3_XXS, IQ3_S, IQ4_XS, Q3_K-Q6_K, Q8_0); the
   inspector lists any tensor whose type has none. With the 4-bit file about 37 GiB of RAM is left beside the
   experts (calculated, of the 188 GiB Linux reports), so parked conversations get less of it (`--park-mib 4096`;
   setup does this itself).

6. **Reference logits from llama.cpp** on the real model. `--n-cpu-moe` keeps that many layers' experts on the
   CPU; lower it until VRAM runs out, then raise it by a few:

   ```bash
   ~/llama.cpp/build/bin/llama-strata-ref -m ~/models/glm53/<first shard> -f tools/glm53_ref/prompt.txt \
       -ngl 99 --n-cpu-moe 40 -c 4096 -b 512 -ub 512 \
       --out ~/glm53-ref/ref.bin --ids-out ~/glm53-ref/ids.txt
   ```

7. **Strata's GPU engine on the real model**, compared with step 6 (paste both outputs):

   ```bash
   build/strata-glm -m ~/models/glm53/<first shard> --tokens-file ~/glm53-ref/ids.txt --ctx 4096 \
       --out ~/glm53-ref/strata.bin
   python tools/glm53_compare.py ~/glm53-ref/ref.bin ~/glm53-ref/strata.bin
   ```

   This compares one token at a time. The VRAM expert cache starts empty and fills from what the prompt uses, so
   the first run is the slowest; `--expert-profile ~/glm53-ref/profile.bin` saves the use counts for the next
   start.

8. **The server**, once step 7 agrees with llama.cpp. Setup writes the run config and the start script for the
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

   For the 4-bit model: `--model IQ4_XS --gguf-dir ~/models/glm53-q4`, and the script is `run-glm-iq4_xs.sh`. The
   run script runs `serve/server.py --engine strata --config strata-glm-ud-iq3_xxs.json --port 8101` with the
   `.venv` Python; later, `./setup.sh` alone starts it too. The engine's output goes to `strata-glm-ud-iq3_xxs.log`
   in the Strata folder, the server's to the window. The server says `ready: http://127.0.0.1:8101/v1 ...` once the
   engine has read the experts (`warming the experts: ...` lines meanwhile). Paste the window's lines from
   `starting the engine` to `ready`, the output of `curl -s http://127.0.0.1:8101/health`, and the `done:` line of a
   first request:

   ```bash
   curl -s http://127.0.0.1:8101/v1/chat/completions -H "Authorization: Bearer $KEY" \
       -H "Content-Type: application/json" \
       -d '{"messages": [{"role": "user", "content": "Say hello in one sentence."}], "max_tokens": 512}'
   ```

   In Open WebUI add an OpenAI connection with that key and the address its container reaches the host at:
   `http://172.17.0.1:8101/v1` on Docker's default bridge, or `http://100.126.222.90:8101/v1` over Tailscale
   (which the agents and the Discord bot can use too). `--context` sets the longest conversation, up to 1,048,576
   tokens for the agents' 512K-1M conversations; each token of it takes 11 KB of VRAM in the MLA cache (1.5 GB at
   131,072, 11.8 GB at 1M, calculated), which the expert cache gives up. `./setup.sh --setup --context N` changes it
   later. The engine options are the run config's `"args"`: setup writes `-m`, `--ctx`, `--park-mib` and
   `--expert-profile` (the use counts, in the data folder `../Strata-GLM-data`), and keeps any other option you add
   there (`--vram-reserve`, `--max-swaps`, ...) when it runs again. To start with step 7's counts, copy
   `~/glm53-ref/profile.bin` over `../Strata-GLM-data/glm-ud-iq3_xxs.profile` before the first start.

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
