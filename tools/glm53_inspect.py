#!/usr/bin/env python3
"""Inspect a GLM-5.3-Flash GGUF before porting work depends on it.

The GLM-5.3-Flash port (docs/GLM53_PORT.md) was planned from the transformers config defaults and the llama.cpp
converter, because the model's own config.json could not be read when the plan was written. This tool reads the
real file and reports:

  1. every metadata key (tokenizer arrays shortened), so the port's constants come from the file, not from memory
  2. a check of the values the port assumes, marked OK / DIFFERS / MISSING (with what the engine and llama.cpp
     use when a key they have a default for is missing), or INFO for a value the engine takes from the file
  3. the tensor names of one dense layer, one KDA layer, one MLA layer and the MTP layer
  4. the quant types used, split into routed experts and everything else of the layers the engine runs, and
     whether Strata's expert kernels can read each type today; the MTP layer's tensors are counted apart, since
     Strata does not load that layer
  5. how much RAM the experts and the rest need: on the plan's 192 GB, and on this machine (MemTotal); and the
     VRAM per token of context the file's shapes give, beside the figure setup.py uses

Usage:  python tools/glm53_inspect.py /path/to/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00003.gguf
        (pass the first shard; the other shards are found next to it)

Reads headers only. No tensor data is loaded, so it runs in seconds on any machine.
"""
from __future__ import annotations

import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(1, str(pathlib.Path(__file__).resolve().parents[1]))
from gguf_reader import GGUFFile  # noqa: E402
from setup import GIB, RAM_LEFT_GIB, RAM_LEFT_MIN_GIB, VRAM_BYTES_PER_TOKEN, ram_gib  # noqa: E402

# Expert quant types the GLM engine's GPU path takes: a GEMV kernel (gpu_linear.cu path_for: native_mmvq or the
# iq kernels) and an f16 dequantizer for prompt chunks (gpu_gemm.cu weight_f16). The CPU takes any ggml type.
STRATA_EXPERT_TYPES = {"F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q3_K", "Q4_K", "Q5_K", "Q6_K",
                       "IQ2_XXS", "IQ2_XS",
                       "IQ3_XXS", "IQ4_NL", "IQ3_S", "IQ2_S", "IQ4_XS", "IQ1_M", "Q2_0"}

# What the port assumes (transformers Glm5NextTextConfig defaults, llama.cpp conversion/glm.py). The suffix after
# the architecture prefix is matched, so this works for both "glm5-next" and "glm5next". None: no value is assumed,
# the file's is printed. A fourth item says what the engine (src/glm/model.cpp) and llama.cpp
# (src/models/glm5-next.cpp) use when the key is missing.
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
    ("attention.indexer.types", None, "per layer: full or shared indexer",
     "the engine and llama.cpp use 1 everywhere: every layer runs its own (full) indexer"),
    ("swiglu_clamp_exp", None, "SwiGLU clamp for routed experts (10 expected)"),
    ("swiglu_clamp_shexp", None, "SwiGLU clamp for the shared expert"),
    ("kda.gate_lower_bound", None, "KDA safe-gate lower bound (-5 expected, if present)"),
    ("attention.indexer.head_count", 32, "indexer heads"),
    ("attention.indexer.key_length", 128, "indexer head dim"),
    ("attention.indexer.top_k", 2048, "indexer top-k"),
    ("attention.indexer.kpool", None, "tokens per pooled indexer key; the engine uses the file's value",
     "the engine and llama.cpp need it and stop"),
    ("attention.indexer.kpool_select_tail", True, "always select the incomplete tail pool",
     "the engine and llama.cpp use true"),
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


def check_values(meta: dict, prefix: str) -> tuple[list[str], int]:
    """Section 2's lines for the keys under `prefix`, and how many of them differ from what the port assumes."""
    lines, differs = [], 0
    for suffix, want, what, *absent in ASSUMED:
        key = prefix + suffix
        if key not in meta:
            lines.append(f"  MISSING  {key:58s} ({what})" + (f": {absent[0]}" if absent else ""))
            continue
        have = meta[key]
        if want is None:
            lines.append(f"  INFO     {key:58s} = {short(have)}  ({what})")
        elif matches(have, want):
            lines.append(f"  OK       {key:58s} = {short(have)}")
        else:
            differs += 1
            lines.append(f"  DIFFERS  {key:58s} = {short(have)}, assumed {want!r}  ({what})")
    return lines, differs


def layer_index(name: str) -> int | None:
    m = re.match(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def trunk_layers(meta: dict, prefix: str, by_layer: dict[int, list]) -> int:
    """The layers the engine runs: block_count less nextn_predict_layers, as src/glm/model.cpp counts them. The
    layers past them are the MTP layers, which Strata does not load. Without those keys: the layers before the first
    one with nextn tensors."""
    blocks, nextn = meta.get(prefix + "block_count"), meta.get(prefix + "nextn_predict_layers", 0)
    if isinstance(blocks, int) and isinstance(nextn, int) and blocks - nextn > 0:
        return blocks - nextn
    mtp = [i for i, ts in by_layer.items() if any(".nextn." in t.name for t in ts)]
    return min(mtp) if mtp else (max(by_layer) + 1 if by_layer else 0)


def quant_sizes(tensors, n_trunk: int) -> tuple[dict[str, dict[str, int]], int]:
    """Bytes per quant type: "experts" (the routed experts of the layers the engine runs), "rest" (everything else
    it loads) and "mtp" (the MTP layers' tensors, not loaded); and the number of tensors of a type with no known
    block size, which are not counted."""
    sizes: dict[str, dict[str, int]] = {"experts": {}, "rest": {}, "mtp": {}}
    unknown = 0
    for t in tensors:
        b = t.expected_bytes()
        if b is None:
            unknown += 1
            continue
        il = layer_index(t.name)
        bucket = "mtp" if il is not None and il >= n_trunk else "experts" if "_exps" in t.name else "rest"
        sizes[bucket][t.type_name] = sizes[bucket].get(t.type_name, 0) + b
    return sizes, unknown


def unsupported_expert_types(experts: dict[str, int]) -> list[str]:
    """The expert types the layers the engine runs use and Strata's kernels cannot read, largest first."""
    return [ty for ty, _ in sorted(experts.items(), key=lambda kv: -kv[1]) if ty not in STRATA_EXPERT_TYPES]


def per_layer(v, n: int) -> list | None:
    if isinstance(v, list):
        return v[:n] if len(v) >= n else None
    return [v] * n if isinstance(v, (int, float)) else None


def vram_per_token(meta: dict, prefix: str, n_trunk: int) -> tuple[float, str] | None:
    """The VRAM a token of context takes in the engine's buffers (src/glm/engine.cu), from the file's shapes: the
    MLA layers' f16 latent and, in each layer that runs its own indexer, one fp32 pooled key per kpool tokens.
    None when a key it needs is missing."""
    kv_heads = per_layer(meta.get(prefix + "attention.head_count_kv"), n_trunk)
    full = per_layer(meta.get(prefix + "attention.indexer.types", 1), n_trunk)
    latent = meta.get(prefix + "attention.kv_lora_rank")
    key = meta.get(prefix + "attention.indexer.key_length")
    kpool = meta.get(prefix + "attention.indexer.kpool")
    if kv_heads is None or full is None or not all(isinstance(v, int) and v > 0 for v in (latent, key, kpool)):
        return None
    mla = [i for i in range(n_trunk) if kv_heads[i]]          # head_count_kv: 0 for a KDA layer
    n_full = sum(1 for i in mla if full[i])
    per_token = len(mla) * latent * 2 + n_full * key * 4 / kpool
    return per_token, (f"{len(mla)} MLA layers x {latent} f16 (the latent cache) + {n_full} indexers x {key} fp32 "
                       f"/ {kpool} tokens (the pooled keys)")


def memory_lines(experts_b: float, rest_b: float, mtp_b: float, ram: float) -> list[str]:
    """Section 5: the experts beside the plan's 192 GB and beside this machine's RAM (`ram` GiB, MemTotal; 0 when it
    could not be read), judged as setup.py judges a size: RAM_LEFT_GIB or more left fits, RAM_LEFT_MIN_GIB or more
    is tight, less does not fit."""
    e, r = experts_b / 1e9, rest_b / 1e9
    lines = [f"  routed experts {e:.1f} GB (Strata keeps all of them in RAM), everything else {r:.1f} GB (GPU)"
             + (f"; the MTP layer's {mtp_b / 1e9:.1f} GB is not loaded" if mtp_b else ""),
             f"  RAM left of 192 GB after experts and ~10 GB for Linux and services: {192 - e - 10:.1f} GB"]
    if ram <= 0:
        lines.append("  this machine: MemTotal not readable (/proc/meminfo)")
        return lines
    eg = experts_b / GIB
    left = ram - eg
    head = f"  this machine: {ram:.0f} GiB of RAM (MemTotal), the experts {eg:.0f} GiB:"
    if left >= RAM_LEFT_GIB:
        lines.append(f"{head} about {left:.0f} GiB left beside the experts: they fit (setup: {RAM_LEFT_GIB} GiB or "
                     "more left)")
    elif left >= RAM_LEFT_MIN_GIB:
        lines.append(f"{head} about {left:.0f} GiB left beside the experts: tight (setup: under {RAM_LEFT_GIB} GiB "
                     "left; parked conversations get less RAM)")
    else:
        lines.append(f"{head} the experts do not fit (setup wants {RAM_LEFT_MIN_GIB} GiB left beside them at the "
                     f"least: {eg + RAM_LEFT_MIN_GIB:.0f} GiB of RAM; {eg + RAM_LEFT_GIB:.0f} GiB to fit well)")
    return lines


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
    lines, differs = check_values(meta, prefix)
    print("\n".join(lines))
    vocab = meta.get("tokenizer.ggml.tokens")
    if isinstance(vocab, list):
        print(f"  INFO     tokenizer.ggml.tokens has {len(vocab)} entries")

    by_layer: dict[int, list] = {}
    for t in tensors:
        il = layer_index(t.name)
        if il is not None:
            by_layer.setdefault(il, []).append(t)
    n_blocks = max(by_layer) + 1 if by_layer else 0
    n_trunk = trunk_layers(meta, prefix, by_layer)

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
    sizes, unknown = quant_sizes(tensors, n_trunk)
    unsupported = unsupported_expert_types(sizes["experts"])
    print(f"  routed experts (layers 0-{n_trunk - 1}, the ones the engine runs):")
    for ty, b in sorted(sizes["experts"].items(), key=lambda kv: -kv[1]):
        note = "Strata kernels: NO, needs a new kernel or a requantize" if ty in unsupported else "Strata kernels: yes"
        print(f"     {ty:8s} {b / 1e9:8.1f} GB   {note}")
    print("  everything else:")
    for ty, b in sorted(sizes["rest"].items(), key=lambda kv: -kv[1]):
        print(f"     {ty:8s} {b / 1e9:8.1f} GB")
    if sizes["mtp"]:
        mtp = f"blk.{n_trunk}" if n_blocks == n_trunk + 1 else f"blk.{n_trunk}-{n_blocks - 1}"
        print(f"  MTP layer ({mtp}), not loaded: Strata does not run it, so its types need no kernel")
        for ty, b in sorted(sizes["mtp"].items(), key=lambda kv: -kv[1]):
            print(f"     {ty:8s} {b / 1e9:8.1f} GB")
    if unknown:
        print(f"  ({unknown} tensors of a type with no known block size were not counted)")

    print("\n== 5. memory")
    print("\n".join(memory_lines(sum(sizes["experts"].values()), sum(sizes["rest"].values()),
                                 sum(sizes["mtp"].values()), ram_gib())))
    vram = vram_per_token(meta, prefix, n_trunk)
    if vram:
        per_token, how = vram
        same = "the same" if round(per_token) == VRAM_BYTES_PER_TOKEN else f"{VRAM_BYTES_PER_TOKEN}: it differs"
        print(f"  VRAM per token of context: {per_token:.0f} bytes = {how} (calculated); setup.py counts {same}")

    print("\n== summary")
    print(f"  {differs} assumed value(s) differ; unsupported expert types: {', '.join(unsupported) or 'none'}")
    print("  Paste this whole output back so the port uses the real values.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
