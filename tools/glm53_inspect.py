#!/usr/bin/env python3
"""Inspect a GLM-5.3-Flash GGUF before porting work depends on it.

The GLM-5.3-Flash port (docs/GLM53_PORT.md) was planned from the transformers config defaults and the llama.cpp
converter, because the model's own config.json could not be read when the plan was written. This tool reads the
real file and reports:

  1. every metadata key (tokenizer arrays shortened), so the port's constants come from the file, not from memory
  2. a check of the values the port assumes, marked OK / DIFFERS / MISSING
  3. the tensor names of one dense layer, one KDA layer, one MLA layer and the MTP layer
  4. the quant types used, split into routed experts and everything else, and whether Strata's expert kernels
     can read each type today
  5. how much RAM the experts and the rest need

Usage:  python tools/glm53_inspect.py /path/to/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00003.gguf
        (pass the first shard; the other shards are found next to it)

Reads headers only. No tensor data is loaded, so it runs in seconds on any machine.
"""
from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402

# Expert quant types Strata's CUDA expert kernels accept today (src/kernels/cuda/iq_kernels.cu:1917-1923,
# native_mmvq.cu:2464-2469). Q6_K only behind STRATA_Q6K_EXPERTS.
STRATA_EXPERT_TYPES = {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q3_K", "Q4_K", "Q5_K", "IQ2_XXS", "IQ2_XS",
                       "IQ3_XXS", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "IQ1_M", "Q2_0"}
STRATA_EXPERT_TYPES_FLAGGED = {"Q6_K"}

# What the port assumes (transformers Glm5NextTextConfig defaults, llama.cpp conversion/glm.py). The suffix after
# the architecture prefix is matched, so this works for both "glm5-next" and "glm5next".
ASSUMED = [
    ("block_count", (45, 46), "45 layers, or 46 if the MTP layer is counted"),
    ("embedding_length", 4096, "hidden size"),
    ("feed_forward_length", 12288, "dense FFN width (layers 0-2)"),
    ("expert_feed_forward_length", 2048, "expert FFN width"),
    ("expert_count", 288, "routed experts"),
    ("expert_used_count", 8, "experts per token"),
    ("expert_shared_count", 1, "shared experts"),
    ("leading_dense_block_count", 3, "dense layers before the MoE layers"),
    ("expert_weights_scale", 2.5, "routed scaling factor"),
    ("expert_weights_norm", True, "renormalise top-k weights"),
    ("expert_gating_func", 2, "2 = sigmoid in llama.cpp"),
    ("context_length", 1048576, "max positions"),
    ("vocab_size", 154880, "vocab"),
    ("attention.head_count", 64, "MLA query heads"),
    ("attention.q_lora_rank", 1536, "MLA q low rank"),
    ("attention.kv_lora_rank", 512, "MLA latent"),
    ("attention.key_length", None, "key length as stored (MLA absorbed form)"),
    ("attention.value_length", None, "value length as stored"),
    ("attention.key_length_mla", 256, "MLA key head dim (qk_nope 256 + rope 0)"),
    ("attention.value_length_mla", 256, "MLA value head dim"),
    ("attention.head_count_kv", None, "per layer: 0 = KDA layer, 1 = MLA layer"),
    ("attention.layer_norm_rms_epsilon", 1e-5, "RMSNorm eps"),
    ("attention.layer_norm_epsilon", None, "LayerNorm eps (indexer key norm)"),
    ("attention.indexer.types", None, "per layer: full or shared indexer"),
    ("swiglu_clamp_exp", None, "SwiGLU clamp for routed experts (10 expected)"),
    ("swiglu_clamp_shexp", None, "SwiGLU clamp for the shared expert"),
    ("kda.gate_lower_bound", None, "KDA safe-gate lower bound (-5 expected, if present)"),
    ("attention.indexer.head_count", 32, "indexer heads"),
    ("attention.indexer.key_length", 128, "indexer head dim"),
    ("attention.indexer.top_k", 2048, "indexer top-k"),
    ("attention.indexer.kpool", 16, "tokens per pooled indexer key"),
    ("attention.indexer.kpool_select_tail", True, "always select the incomplete tail pool"),
    ("kda.head_dim", 128, "KDA head dim"),
    ("ssm.conv_kernel", 4, "KDA short conv"),
    ("hyper_connection.count", 4, "mHC residual streams"),
    ("hyper_connection.sinkhorn_iterations", 20, "mHC Sinkhorn iterations"),
    ("hyper_connection.epsilon", 1e-6, "mHC eps"),
    ("nextn_predict_layers", 1, "MTP layers"),
]


def short(v, n=8):
    if isinstance(v, list):
        if len(v) > n:
            return f"[{', '.join(map(repr, v[:n]))}, ... ({len(v)} items)]"
        return repr(v)
    if isinstance(v, str) and len(v) > 200:
        return repr(v[:200]) + f"... ({len(v)} chars)"
    return repr(v)


def shards_of(first: pathlib.Path) -> list[pathlib.Path]:
    m = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", first.name)
    if not m:
        return [first]
    n = int(m.group(2))
    paths = [first.with_name(f"{m.group(1)}-{i:05d}-of-{m.group(2)}.gguf") for i in range(1, n + 1)]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        sys.exit(f"missing shards: {', '.join(missing)}")
    return paths


def matches(have, want) -> bool:
    if isinstance(want, tuple):
        return any(matches(have, w) for w in want)
    if isinstance(want, float):
        return isinstance(have, (int, float)) and abs(have - want) <= 1e-9 + 1e-6 * abs(want)
    return have == want


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    paths = shards_of(pathlib.Path(sys.argv[1]))
    files = [GGUFFile(p) for p in paths]
    meta = files[0].metadata
    tensors = [t for f in files for t in f.tensors]
    arch = meta.get("general.architecture", "?")

    print(f"== files: {len(paths)} shard(s), {len(tensors)} tensors, architecture '{arch}'")
    if arch not in ("glm5-next", "glm5next"):
        print(f"!! architecture '{arch}' is not glm5-next / glm5next: this does not look like GLM-5.3-Flash")

    print("\n== 1. metadata")
    for k in sorted(meta):
        print(f"  {k} = {short(meta[k])}")

    print("\n== 2. values the port assumes")
    prefix = f"{arch}."
    differs = 0
    for suffix, want, what in ASSUMED:
        key = prefix + suffix
        if key not in meta:
            print(f"  MISSING  {key:58s} ({what})")
            continue
        have = meta[key]
        if want is None:
            print(f"  INFO     {key:58s} = {short(have)}  ({what})")
        elif matches(have, want):
            print(f"  OK       {key:58s} = {short(have)}")
        else:
            differs += 1
            print(f"  DIFFERS  {key:58s} = {short(have)}, assumed {want!r}  ({what})")
    vocab = meta.get("tokenizer.ggml.tokens")
    if isinstance(vocab, list):
        print(f"  INFO     tokenizer.ggml.tokens has {len(vocab)} entries")

    by_layer: dict[int, list] = {}
    for t in tensors:
        m = re.match(r"blk\.(\d+)\.", t.name)
        if m:
            by_layer.setdefault(int(m.group(1)), []).append(t)
    n_blocks = max(by_layer) + 1 if by_layer else 0

    def kind(layer: int) -> str:
        names = {t.name.split(".", 2)[2] for t in by_layer.get(layer, [])}
        attn = "KDA" if any(n.startswith("ssm_") for n in names) else (
            "MLA" if any(n.startswith("attn_kv_a") or n.startswith("attn_q_a") for n in names) else "?")
        ffn = "MoE" if any("_exps" in n for n in names) else "dense"
        mtp = "+MTP" if any(n.startswith("nextn") for n in names) else ""
        return f"{attn}/{ffn}{mtp}"

    print("\n== 3. layer pattern")
    print("  " + " ".join(f"{i}:{kind(i)}" for i in range(n_blocks)))
    shown: set[str] = set()
    for i in range(n_blocks):
        k = kind(i)
        if k in shown or i not in by_layer:
            continue
        shown.add(k)
        print(f"\n  -- blk.{i} ({k})")
        for t in sorted(by_layer[i], key=lambda t: t.name):
            print(f"     {t.name.split('.', 2)[2]:40s} {t.type_name:8s} {t.shape}")
    others = sorted((t for t in tensors if not t.name.startswith("blk.")), key=lambda t: t.name)
    print("\n  -- non-layer tensors")
    for t in others:
        print(f"     {t.name:40s} {t.type_name:8s} {t.shape}")

    print("\n== 4. quant types")
    exp_bytes: dict[str, int] = {}
    rest_bytes: dict[str, int] = {}
    unknown = 0
    for t in tensors:
        b = t.expected_bytes()
        if b is None:
            unknown += 1
            continue
        bucket = exp_bytes if "_exps" in t.name else rest_bytes
        bucket[t.type_name] = bucket.get(t.type_name, 0) + b
    print("  routed experts:")
    unsupported = []
    for ty, b in sorted(exp_bytes.items(), key=lambda kv: -kv[1]):
        if ty in STRATA_EXPERT_TYPES:
            note = "Strata kernels: yes"
        elif ty in STRATA_EXPERT_TYPES_FLAGGED:
            note = "Strata kernels: only with STRATA_Q6K_EXPERTS"
        else:
            note = "Strata kernels: NO, needs a new kernel or a requantize"
            unsupported.append(ty)
        print(f"     {ty:8s} {b / 1e9:8.1f} GB   {note}")
    print("  everything else:")
    for ty, b in sorted(rest_bytes.items(), key=lambda kv: -kv[1]):
        print(f"     {ty:8s} {b / 1e9:8.1f} GB")
    if unknown:
        print(f"  ({unknown} tensors of a type with no known block size were not counted)")

    e = sum(exp_bytes.values()) / 1e9
    r = sum(rest_bytes.values()) / 1e9
    print("\n== 5. memory")
    print(f"  routed experts {e:.1f} GB (Strata keeps all of them in RAM), everything else {r:.1f} GB (GPU)")
    print(f"  RAM left of 192 GB after experts and ~10 GB for Linux and services: {192 - e - 10:.1f} GB")

    print("\n== summary")
    print(f"  {differs} assumed value(s) differ; unsupported expert types: {', '.join(unsupported) or 'none'}")
    print("  Paste this whole output back so the port uses the real values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
