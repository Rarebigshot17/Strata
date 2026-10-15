# Porting Strata to GLM-5.3-Flash

This branch (`glm53-flash`) turns Strata from a Qwen3.8-Flash-Next engine into a GLM-5.3-Flash engine. It keeps
how Strata works: every routed expert in system RAM, the most-used experts cached in VRAM, the rest computed on
the CPU, one model compiled for speed. GLM-5.3-Flash replaces Qwen3.8-Flash-Next. Nothing of the Qwen model
stays.

Target machine: napster-ai (Xeon E5-2699 v4, 192 GB DDR4, RTX 3090 Ti 24 GB + RTX 3080 Ti 12 GB + RTX 3080
10 GB, Linux). CUDA only: the SYCL (Intel) engine is removed on this branch, and HIP (AMD) follows.

Estimated effort: about 5-6 months of engineering. Code is written off the machine and tested on napster-ai by
hand, so every milestone ends with commands to run and output to paste back.

## The model

From the transformers config defaults and llama.cpp's `src/models/glm5-next.cpp`. `tools/glm53_inspect.py` checks
each value against the real GGUF before code depends on it.

| | GLM-5.3-Flash | Qwen3.8-Flash-Next (what Strata runs today) |
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
| Extras | 1 MTP layer; thinking on by default | MTP layer, n-gram table, vision |

## Memory on napster-ai

The routed experts are about 305B parameters (288 experts x 3 matrices x 4096 x 2048 x 42 MoE layers).

| Option | Expert quant | Experts in RAM | RAM left for KV, Linux and the server |
|---|---|---|---|
| 3-bit | IQ3_XXS (3.06 bits) | about 117 GB | about 65 GB |
| 4-bit | IQ4_XS (4.25 bits) | about 162 GB | about 20 GB |

The MLA cache stores a 512-wide latent per token in each of the 11 MLA layers. The engine keeps it in f16, as
llama.cpp does by default: 11 KB per token, 11.8 GB of VRAM at 1M tokens (an 8-bit cache would halve that). The indexer adds one pooled 128-wide key per 16 tokens per layer,
under 0.5 KB per token even in fp32. The KDA layers hold a fixed-size state (64 heads x 128 x 128 per layer).
These are calculated from the config, not measured.

## How the engine is built

The Qwen engine's code is specialised to that model at every level (sizes compiled into the kernels, two layer
types, captured CUDA graphs built around them). Rewriting it in place would mean changing every one of those
pieces with no way to test them in between. So the GLM engine is new code under `src/glm/` and
`include/strata/glm/`. It keeps Strata's design (every routed expert in RAM, the most used ones cached in VRAM,
the rest computed on the CPU) and reuses Strata's generic parts: the GGUF reader, the quantized GEMV kernels
(`native_mmvq`), the CPU expert dot products, the expert cache and the server's engine protocol. The Qwen engine
is deleted once the GLM engine replaces it.

Every piece is checked against llama.cpp, which has run GLM-5.3-Flash since its `glm5-next` support landed:

1. **CPU reference** (`strata-glm-ref`, `src/glm/cpu_ref.cpp`): the whole model on the CPU, one token at a time,
   with ggml-cpu's own dot products. It matches llama.cpp on small random GLM-5.3-Flash models to float rounding
   (`tools/glm53_selftest.sh`, which runs without a GPU or a download). On the real model it is slow, about one
   token per second, and only used as the reference.
2. **GPU engine**: each CUDA kernel is checked against the CPU reference, then whole-model logits against
   llama.cpp on the real GGUF (`tools/glm53_compare.py`).

| # | Milestone | Status | Test |
|---|---|---|---|
| M0 | Remove SYCL; GGUF inspector; llama.cpp reference harness | done | inspector output, reference logits |
| M1 | CPU reference: config and tensors from the GGUF, mHC, KDA, MLA with the k-pool indexer, MoE with sigmoid routing and SwiGLU clamp | done | `tools/glm53_selftest.sh`: all variants PASS |
| M2 | CUDA: dense projections, mHC, KDA decode, MLA decode with the indexer, routing; layer split over the GPUs | written, not yet run on a GPU | `tools/glm53_gpu_check.sh`; logits vs llama.cpp |
| M3 | Experts: all in RAM (the mapped GGUF), VRAM cache of the most used, CPU for the rest | written, not yet run on a GPU | logits unchanged; tokens/s |
| M4 | Prompt processing: batched projections, chunked KDA, batched MLA (causal below 2,063 tokens of context, each query's own selection past that), experts streamed through VRAM | written, not yet run on a GPU | prompt tokens/s; logits unchanged |
| M5 | Server: the engine protocol, GLM chat template, tool calls (`<arg_key>`/`<arg_value>`), thinking, reasoning effort | written, checked on the CPU engine (`tools/glm53_server_check.sh`) | Open WebUI and an agent work end to end |
| M6 | Several clients: conversation parking and checkpoints; batching; MTP drafting | parking written, checked on the CPU engine (`tools/glm53_park_check.py`); the rest not started | speed for 1, 2 and 4 clients |
| M7 | Remove the Qwen engine and HIP; docs | | |

### Several clients on one engine

The agents, the chat and the bot take turns on one engine: requests queue in the server, one runs at a time. The
KDA layers' state is a running sum that cannot be rewound, so the engine can only continue a state whose tokens
are the start of the new prompt. To keep each client's conversation anyway, the engine (`src/glm/serve_loop.cpp`)
keeps several:

- **Parking.** When a prompt starts another conversation, the current one's whole state is copied to host RAM.
  A later prompt that continues it copies it back and reads only the new tokens.
- **Checkpoints.** The server asks for a checkpoint where the reply begins (`cp=` on the GEN line: the prompt
  without the `<|assistant|><think>` generation prompt). The next turn renders the previous reply again without
  its reasoning, so it parts from the engine's state there; it restarts from the checkpoint and reads the reply
  and the new message, not the whole conversation. A regenerated reply restarts there too.
- **Memory.** `--park-mib` (default 8192) caps what parked conversations and checkpoints use; the least recently
  used conversation goes first. A parked conversation is the KDA layers' state (about 0.15 GB, if they have 64
  heads of 128 like the attention) plus 11 KB per token, so a 100K-token conversation is about 1.3 GB; a
  checkpoint is about 0.15 GB. Copies cross PCIe 3.0 at a few GB/s, a fraction of a second either way.

Nothing about the answers changes: `tools/glm53_park_check.py` checks every resumed reply against a fresh
engine's, token for token.

Small random GLM-5.3-Flash models for tests that need no download: llama.cpp's
`test-llama-archs -a glm5-next -o DIR` writes a 20 MB `glm5-next-moe.gguf` with 2 layers (one MLA, one KDA).
`tools/glm53_tiny_variant.py` switches on the paths it leaves off (SwiGLU clamps, expert weight scaling, an
indexer top-k small enough to drop pools).

Two things found while writing the reference, both checked against llama.cpp's code:

- The KDA decay applies per **key** channel: `S[i][:] *= exp(g[i])`. That is what ggml's fused
  `gated_delta_net` op does, and llama.cpp uses that op by default. Its unfused one-token graph
  (`build_delta_net_autoregressive`) broadcasts the decay over the other axis.
- Below 2,048 tokens of context the indexer selects every pool, so attention is plain causal there. The
  indexer only changes results on longer contexts, which is why the test variants shrink its top-k.

## What to run on napster-ai

Each step prints a few lines to paste back. Steps 1-4 need no model download.

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
   # 3-bit, about 120 GB:
   hf download unsloth/GLM-5.3-Flash-GGUF --include "*UD-IQ3_XXS*" --local-dir ~/models/glm53
   # or 4-bit, about 170 GB: the IQ4_XS shards (or the UD-Q4_K_XL ones, whichever the list shows)
   hf download unsloth/GLM-5.3-Flash-GGUF --include "*IQ4_XS*" --local-dir ~/models/glm53-q4
   python tools/glm53_inspect.py ~/models/glm53/<first .gguf shard>
   ```

   The GPU kernels take every type these files use for experts (IQ3_XXS, IQ3_S, IQ4_XS, Q3_K-Q6_K, Q8_0); the
   inspector lists any tensor whose type has none. With the 4-bit file about 20 GB of RAM is left beside the
   experts, so give parked conversations less of it (`--park-mib 4096`).

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

8. **The server**, once step 7 agrees with llama.cpp. Port 8101 keeps clear of Open WebUI (3000), Portainer
   (8000) and the UPS dashboard (8080). Anything beyond this machine's loopback needs an API key:

   ```bash
   python -m serve.glm_server --model ~/models/glm53/<first shard> --engine build/strata-glm \
       --host 0.0.0.0 --port 8101 --ctx 131072 --api-key "$(openssl rand -hex 24)" --log ~/glm53-engine.log \
       -- --expert-profile ~/glm53-ref/profile.bin
   ```

   In Open WebUI add an OpenAI connection with that key and the address its container reaches the host at:
   `http://172.17.0.1:8101/v1` on Docker's default bridge, or `http://100.126.222.90:8101/v1` over Tailscale
   (which the agents and the Discord bot can use too). `--ctx` sets the longest conversation; each token of it
   takes 11 KB of VRAM in the MLA cache (1.5 GB at 131072, 11.8 GB at 1M), which the expert cache gives up.
   Everything after `--` goes to the engine (`--park-mib`, `--expert-profile`, `--vram-reserve`, ...).

## Removed on this branch

- `sycl/` (the Intel Arc engine), `docs/INTEL_ARC.md`, `docs/INTEL.md`, `docs/A770_SPEED_SPEC.md`, the Arc
  community benchmark results, `tools/test_setup_sycl.py`, and `setup.py --backend sycl`.
- The README and other docs still describe the Qwen engine. They are rewritten at M7, once the GLM engine
  replaces it.
