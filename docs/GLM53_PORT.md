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

The MLA cache stores a 512-wide latent per token in each of the 11 MLA layers: about 5.6 KB per token at 8 bits
(11 KB at 16 bits), so about 6 GB at 1M tokens. The indexer adds one pooled 128-wide key per 16 tokens per layer,
under 0.5 KB per token even in fp32. The KDA layers hold a fixed-size state (64 heads x 128 x 128 per layer).
These are calculated from the config, not measured.

## Milestones

Each milestone ends with a test you run on napster-ai. The main check compares Strata's logits with llama.cpp's
for the same token ids, using the same GGUF (`tools/glm53_compare.py`). llama.cpp has run GLM-5.3-Flash since
the `glm5-next` support landed. For bisecting, Strata's `--dump-layers` is compared with `llama-debug`'s per-layer
tensors.

| # | Milestone | Weeks | Test |
|---|---|---|---|
| M0 | Groundwork: remove SYCL, GGUF inspector, llama.cpp reference harness | done on this branch | inspector output, reference logits |
| M1 | Strip Qwen: n-gram table, vision, Qwen MTP, Qwen packing tools; architecture guard accepts `glm5-next` | 2 | engine builds; refuses a Qwen GGUF |
| M2 | Model sizes read from the GGUF instead of `constexpr` (about 40 places, including the expert kernels and CPU buffers) | 3 | engine builds; unit tests at the GLM sizes |
| M3 | Loader and packer for `glm5-next` tensor names; embeddings, output head, dense layers 0-2 | 1 | pack builds; tensor checks pass |
| M4 | MoE: sigmoid router with bias and x2.5 scale, 288 x top-8, ungated shared expert, SwiGLU clamp, GPU and CPU expert kernels at 4096 / 2048 | 1-2 | routing dump matches llama.cpp |
| M5 | mHC residual (4 streams, Sinkhorn) | 1 | layer dumps match |
| M6 | KDA decode and token-by-token prefill | 2 | KDA layer dumps match |
| M7 | MLA attention, latent KV cache (paged, 8-bit), k-pool indexer | 5-6 | **first full logits match** |
| M8 | Chunked KDA prefill, fast MLA prefill | 3 | prompt speed; logits still match |
| M9 | Server: GLM chat template, tool calls (`<arg_key>`/`<arg_value>`), thinking tags, reasoning effort, tokenizer | 1 | Open WebUI and an agent work end to end |
| M10 | MTP drafter | 2 | speedup measured, output unchanged |
| M11 | Remove HIP; split layers over 3 GPUs; 3-bit and 4-bit packs; batching for several clients; tuning | 2-3 | speed table for 1, 2 and 4 clients |

Small random GLM-5.3-Flash models for tests that need no download: llama.cpp's
`test-llama-archs -a glm5-next -o DIR` writes a 20 MB `glm5-next-moe.gguf` with 2 layers (one MLA, one KDA), and
`llama-strata-ref` runs it on the CPU in a second. The loader, packer and CPU-side tests use it.

## M0: what to run on napster-ai

1. Download the 3-bit GGUF (about 120 GB; Unsloth re-uploaded it on 2026-10-05 with `glm5-next` metadata):

   ```bash
   pip install -U huggingface_hub
   hf download unsloth/GLM-5.3-Flash-GGUF --include "*UD-IQ3_XXS*" --local-dir ~/models/glm53
   ```

2. Check it against what the port assumes, and paste the whole output back:

   ```bash
   python tools/glm53_inspect.py ~/models/glm53/<path to the first .gguf shard>
   ```

3. Build llama.cpp with CUDA plus the reference tool (pinned to the commit the tool was written against):

   ```bash
   tools/glm53_ref/build.sh ~/llama.cpp
   ```

4. Make the reference logits. `--n-cpu-moe` keeps that many layers' experts on the CPU; lower it until VRAM runs
   out, then raise it by a few:

   ```bash
   ~/llama.cpp/build/bin/llama-strata-ref -m ~/models/glm53/<first shard> -f tools/glm53_ref/prompt.txt \
       -ngl 99 --n-cpu-moe 40 -c 4096 -b 512 -ub 512 \
       --out ~/glm53-ref/ref.bin --ids-out ~/glm53-ref/ids.txt
   ```

   Paste back the last 20 lines it prints, and keep `ref.bin` and `ids.txt`: every later milestone compares
   against them.

5. Optional, as a sanity check of the model itself: `~/llama.cpp/build/bin/llama-cli -m <first shard> -ngl 99
   --n-cpu-moe 40 -p "Hello"` and say whether the answer reads sensibly and how many tokens/s it reports.

## Removed on this branch

- `sycl/` (the Intel Arc engine), `docs/INTEL_ARC.md`, `docs/INTEL.md`, `docs/A770_SPEED_SPEC.md`, the Arc
  community benchmark results, `tools/test_setup_sycl.py`, and `setup.py --backend sycl`.
- The README and other docs still describe the Qwen engine. They are rewritten at M9, once the GLM engine runs.
