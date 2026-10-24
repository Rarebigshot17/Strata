#!/usr/bin/env python3
"""Copy a tiny random GLM-5.3-Flash GGUF with an MTP block added after its trunk, as the real model has one.

llama.cpp's `test-llama-archs -a glm5-next -o DIR` writes a model without the MTP (NextN) block that GLM-5.3-Flash
carries as its last layer (blk.45 of 46). This copy adds one, so the MTP layer of the engine (--mtp) and llama.cpp's
(`--spec-type draft-mtp`) can be compared:

    python tools/glm53_add_mtp.py IN.gguf OUT.gguf [--shared-head] [--embed-tokens] [--seed N]

The block is an MLA layer with its own indexer and a routed-expert FFN, as llama.cpp's glm5-next.cpp loads it: the
attention tensors are shaped like the first MLA layer's of the trunk, the FFN's like the first MoE layer's, every value
new and random (normal, the source tensor's standard deviation, `--seed`). It adds the NextN tensors (`nextn.eh_proj`
[2 * n_embd -> n_embd], `nextn.enorm`, `nextn.hnorm`) and, with `--shared-head`, `nextn.shared_head_norm` and
`nextn.shared_head_head` (else the MTP head uses output_norm and output), and with `--embed-tokens`,
`nextn.embed_tokens` (else token_embd). The trunk's own scale tensors (`.scale`, `.input_scale`) are not copied.

The metadata: block_count one more, nextn_predict_layers 1, and every per-layer array one element longer (the MLA
layer's head_count_kv, 0 in recurrent_layers, else the last element again); attention.indexer.types stays as it is,
since llama.cpp reads it for the trunk only. Needs gguf-py (`pip install gguf`, or PYTHONPATH=<llama.cpp>/gguf-py) and
numpy.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import gguf

MLA_PARTS = ("attn_norm.", "attn_q_a.", "attn_q_a_norm.", "attn_q_b.", "attn_kv_a_mqa.", "attn_kv_a_norm.",
             "attn_k_b.", "attn_v_b.", "attn_output.", "indexer.", "indexer_compressor_gate.",
             "indexer_compressor_ape.")
MOE_PARTS = ("ffn_norm.", "ffn_gate_inp.", "exp_probs_b.", "ffn_gate_exps.", "ffn_up_exps.", "ffn_down_exps.",
             "ffn_gate_shexp.", "ffn_up_shexp.", "ffn_down_shexp.")
# per-layer arrays llama.cpp reads for the trunk only
TRUNK_ONLY = ("attention.indexer.types",)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--shared-head", action="store_true")
    ap.add_argument("--embed-tokens", action="store_true")
    ap.add_argument("--seed", type=int, default=45)
    a = ap.parse_args()

    reader = gguf.GGUFReader(a.src, "r")
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    fields = {f.name: f for f in reader.fields.values()}
    n_layer = int(fields[f"{arch}.block_count"].contents())
    if int(fields[f"{arch}.nextn_predict_layers"].contents()) if f"{arch}.nextn_predict_layers" in fields else 0:
        print(f"{a.src} already has an MTP block", file=sys.stderr)
        return 1
    hkv = fields[f"{arch}.attention.head_count_kv"].contents()
    hkv = hkv if isinstance(hkv, list) else [hkv] * n_layer
    mla = next((i for i, h in enumerate(hkv) if h), None)
    tensors = {t.name: t for t in reader.tensors}
    moe = next((i for i in range(n_layer) if f"blk.{i}.ffn_gate_exps.weight" in tensors), None)
    if mla is None or moe is None:
        print("the trunk needs an MLA layer and a routed-expert layer to shape the block on", file=sys.stderr)
        return 1

    writer = gguf.GGUFWriter(a.dst, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        name = field.name
        if name == gguf.Keys.General.ARCHITECTURE or name.startswith("GGUF."):
            continue
        t = field.types[0]
        sub = field.types[-1] if t == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if t == gguf.GGUFValueType.ARRAY and len(value) == 0:
            continue
        short = name[len(arch) + 1:] if name.startswith(arch + ".") else name
        if short == "block_count":
            value = n_layer + 1
        elif short == "nextn_predict_layers":
            value = 1
        elif t == gguf.GGUFValueType.ARRAY and len(value) == n_layer and short not in TRUNK_ONLY:
            extra = hkv[mla] if short == "attention.head_count_kv" else 0 if short == "attention.recurrent_layers" \
                else value[-1]
            value = list(value) + [extra]
        writer.add_key_value(name, value, t, sub_type=sub)
    if f"{arch}.nextn_predict_layers" not in fields:
        writer.add_uint32(f"{arch}.nextn_predict_layers", 1)

    rng = np.random.default_rng(a.seed)
    nb = f"blk.{n_layer}."

    def fresh(like: np.ndarray, std: float) -> np.ndarray:
        return (rng.standard_normal(like.shape) * std).astype(np.float32)

    added: list[tuple[str, np.ndarray]] = []
    for src_layer, parts in ((mla, MLA_PARTS), (moe, MOE_PARTS)):
        pre = f"blk.{src_layer}."
        for name, t in tensors.items():
            rest = name[len(pre):] if name.startswith(pre) else None
            if rest is None or not rest.startswith(parts) or not rest.endswith((".weight", ".bias")):
                continue
            if t.tensor_type != gguf.GGMLQuantizationType.F32:
                print(f"{name} is {t.tensor_type.name}: the block is made from an F32 model", file=sys.stderr)
                return 1
            d = np.asarray(t.data, dtype=np.float32)
            added.append((nb + rest, fresh(d, float(d.std()) or 0.1)))
    E = int(fields[f"{arch}.embedding_length"].contents())
    V = int(tensors["token_embd.weight"].shape[1])
    added.append((nb + "nextn.eh_proj.weight", (rng.standard_normal((E, 2 * E)) * 0.1).astype(np.float32)))
    added.append((nb + "nextn.enorm.weight", (rng.standard_normal(E) * 0.1).astype(np.float32)))
    added.append((nb + "nextn.hnorm.weight", (rng.standard_normal(E) * 0.1).astype(np.float32)))
    if a.shared_head:
        added.append((nb + "nextn.shared_head_norm.weight", (rng.standard_normal(E) * 0.1).astype(np.float32)))
        added.append((nb + "nextn.shared_head_head.weight", (rng.standard_normal((V, E)) * 0.1).astype(np.float32)))
    if a.embed_tokens:
        added.append((nb + "nextn.embed_tokens.weight", (rng.standard_normal((V, E)) * 0.1).astype(np.float32)))

    for t in reader.tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    for name, arr in added:
        writer.add_tensor_info(name, arr.shape, arr.dtype, arr.nbytes, gguf.GGMLQuantizationType.F32)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    for _, arr in added:
        writer.write_tensor_data(arr)
    writer.close()
    print(f"wrote {a.dst}: blk.{n_layer} as the MTP block ({len(added)} tensors; attention shaped on blk.{mla}, "
          f"FFN on blk.{moe})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
