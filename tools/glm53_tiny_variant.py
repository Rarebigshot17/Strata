#!/usr/bin/env python3
"""Copy a GGUF with some metadata values replaced: test variants of a tiny random GLM-5.3-Flash model.

llama.cpp's `test-llama-archs -a glm5-next -o DIR` writes a 2-layer random model whose settings leave several paths
of the real model off (no SwiGLU clamp, no expert weight scaling, an indexer top-k larger than any test prompt).
This tool switches them on, so the engine and llama.cpp can be compared on every path:

    python tools/glm53_tiny_variant.py IN.gguf OUT.gguf \\
        glm5-next.attention.indexer.top_k=8 glm5-next.swiglu_clamp_exp=0.05 glm5-next.expert_weights_scale=2.5

A value keeps the type the key already has; an array key set to one number sets every element. A minus before a key
and no value (`-glm5-next.attention.indexer.types`) leaves the key out of the copy, for the files that do not carry
it: the reader then takes its default. Needs gguf-py (`pip install gguf`, or PYTHONPATH=<llama.cpp>/gguf-py).
"""
from __future__ import annotations

import sys

import gguf


def parse_value(old_type, sub_type, old_value, text: str):
    T = gguf.GGUFValueType
    def one(t, s):
        if t in (T.FLOAT32, T.FLOAT64):
            return float(s)
        if t == T.BOOL:
            return s.lower() in ("1", "true", "yes")
        if t == T.STRING:
            return s
        return int(s)
    if old_type == T.ARRAY:
        parts = text.split(",")
        if len(parts) == 1:
            return [one(sub_type, parts[0])] * len(old_value)
        return [one(sub_type, p) for p in parts]
    return one(old_type, text)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, dst, edits = sys.argv[1], sys.argv[2], sys.argv[3:]
    changes, removes = {}, set()
    for e in edits:
        if e.startswith("-") and "=" not in e:
            removes.add(e[1:])
            continue
        k, _, v = e.partition("=")
        changes[k] = v

    reader = gguf.GGUFReader(src, "r")
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
    seen = set()
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        t = field.types[0]
        sub = field.types[-1] if t == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if field.name in removes:
            seen.add(field.name)
            print(f"{field.name}: left out")
            continue
        if field.name in changes:
            value = parse_value(t, sub, value, changes[field.name])
            seen.add(field.name)
            print(f"{field.name} = {value}")
        if t == gguf.GGUFValueType.ARRAY and len(value) == 0:
            print(f"{field.name}: empty array, left out (gguf-py cannot write one)")
            continue
        writer.add_key_value(field.name, value, t, sub_type=sub)
    missing = (set(changes) | removes) - seen
    if missing:
        print(f"keys not in the file: {', '.join(sorted(missing))}", file=sys.stderr)
        return 1
    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes,
                               tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)
    writer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
