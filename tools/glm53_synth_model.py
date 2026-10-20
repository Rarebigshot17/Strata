#!/usr/bin/env python3
"""A small random GLM-5.3-Flash with a real GLM-style tokenizer: for testing the server and the tokenizer.

llama.cpp's `test-llama-archs -a glm5-next` writes a random 2-layer model whose tokenizer is a placeholder
(`tok_0` ... `tok_127`). This copies it with:

  * a byte-level BPE vocabulary: the 256 byte symbols, merges learned from a small corpus, and GLM's special
    tokens (`[gMASK]`, `<sop>`, `<|system|>`, `<|user|>`, `<|assistant|>`, `<|observation|>`, `<|endoftext|>` as
    control tokens; `<think>`, `<tool_call>`, `<arg_key>`, ... as user-defined ones), with `pre` = glm5 so
    llama.cpp tokenizes it the way it tokenizes GLM-5
  * the chat template from TEMPLATE (serve/chat_template.jinja is GLM-5.3-Flash's own, the one
    tools/glm53_server_check.sh uses)
  * token_embd and output resized to the new vocabulary, random

    python tools/glm53_synth_model.py IN.gguf TEMPLATE.jinja OUT.gguf [--merges 80]

Needs gguf-py (pip install gguf, or PYTHONPATH=<llama.cpp>/gguf-py) and numpy.
"""
from __future__ import annotations

import argparse
import collections
import pathlib
import sys

import numpy as np

import gguf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from strata_tokenizer import BYTE_TO_UNICODE, GLM4_PATTERN  # noqa: E402

import regex  # noqa: E402

CONTROL = ["<|endoftext|>", "[MASK]", "[gMASK]", "[sMASK]", "<sop>", "<eop>", "<|system|>", "<|user|>",
           "<|assistant|>", "<|observation|>"]
USER_DEFINED = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>",
                "</arg_value>", "<tool_response>", "</tool_response>"]

CORPUS = """The quick brown fox jumps over the lazy dog. Strata keeps every routed expert in RAM and the most used
ones in VRAM. def main(): return 42  # numbers like 12345 and 3.14159 tokenize in runs of three.
Tools: read_file, write_file, search the web. The answer is yes. I think the user wants a summary of the file.
Hello, world! How are you today? The model calls a tool with arguments and reads the tool's response."""


def learn_merges(corpus: str, n: int) -> list[tuple[str, str]]:
    words = collections.Counter()
    for piece in regex.findall(GLM4_PATTERN, corpus):
        words[tuple(BYTE_TO_UNICODE[b] for b in piece.encode("utf-8"))] += 1
    merges = []
    for _ in range(n):
        pairs = collections.Counter()
        for w, c in words.items():
            for a, b in zip(w, w[1:]):
                pairs[(a, b)] += c
        if not pairs:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        merges.append(best)
        new = collections.Counter()
        for w, c in words.items():
            out, i = [], 0
            while i < len(w):
                if i + 1 < len(w) and (w[i], w[i + 1]) == best:
                    out.append(w[i] + w[i + 1])
                    i += 2
                else:
                    out.append(w[i])
                    i += 1
            new[tuple(out)] += c
        words = new
    return merges


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("template")
    ap.add_argument("dst")
    ap.add_argument("--merges", type=int, default=80)
    a = ap.parse_args()

    merges = learn_merges(CORPUS, a.merges)
    tokens = [BYTE_TO_UNICODE[b] for b in range(256)]
    types = [1] * 256
    for l, r in merges:
        if l + r not in tokens:
            tokens.append(l + r)
            types.append(1)
    for t in CONTROL:
        tokens.append(t)
        types.append(3)
    for t in USER_DEFINED:
        tokens.append(t)
        types.append(4)
    V = len(tokens)

    reader = gguf.GGUFReader(a.src, "r")
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(a.dst, arch=arch, endianess=reader.endianess)
    for field in reader.fields.values():
        name = field.name
        if name == gguf.Keys.General.ARCHITECTURE or name.startswith("GGUF.") or name.startswith("tokenizer."):
            continue
        t = field.types[0]
        sub = field.types[-1] if t == gguf.GGUFValueType.ARRAY else None
        value = field.contents()
        if t == gguf.GGUFValueType.ARRAY and len(value) == 0:
            continue
        if name == f"{arch}.vocab_size":
            value = V
        writer.add_key_value(name, value, t, sub_type=sub)
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("glm5")
    writer.add_token_list(tokens)
    writer.add_token_types(types)
    writer.add_token_merges([f"{l} {r}" for l, r in merges])
    writer.add_eos_token_id(tokens.index("<|endoftext|>"))
    writer.add_add_bos_token(False)
    writer.add_chat_template(pathlib.Path(a.template).read_text(encoding="utf-8"))

    rng = np.random.default_rng(7)
    resized = {}
    for tensor in reader.tensors:
        if tensor.name in ("token_embd.weight", "output.weight"):
            n_embd = int(tensor.shape[0])
            resized[tensor.name] = (rng.standard_normal((V, n_embd)) * 0.1).astype(np.float32)
    for tensor in reader.tensors:
        if tensor.name in resized:
            arr = resized[tensor.name]
            writer.add_tensor_info(tensor.name, arr.shape, arr.dtype, arr.nbytes, gguf.GGMLQuantizationType.F32)
        else:
            writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes,
                                   tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(resized.get(tensor.name, tensor.data), tensor_endianess=reader.endianess)
    writer.close()
    print(f"wrote {a.dst}: vocab {V} ({len(merges)} merges, {len(CONTROL) + len(USER_DEFINED)} special tokens)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
