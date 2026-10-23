#!/usr/bin/env python3
"""Where Strata's CPU reference and llama.cpp part, layer by layer, for the GLM-5.3-Flash port.

Both files hold, per position and layer, the points --names lists, by the names llama.cpp's graph gives them, in that
order, float32, no header: `strata-glm-ref --dump-layers` and `llama-strata-ref --dump-layers` (tools/glm53_ref; it
needs -b 1 -ub 1) write them, on the same --tokens-file and with the same --dump-names. The default,
hc_attn_post,l_out, is the residual streams after the attention and after the FFN (hc x n_embd each); hc_attn_pre,
attn_norm, hc_ffn_pre, ffn_norm and ffn_out are n_embd each.

    python tools/glm53_layers_compare.py LLAMA.bin STRATA.bin --layers 45 --embd 4096 [--names a,b] [--limit 1e-3]

For each position: the first point (layer and name) whose relative difference (max |a - b| over max |a|, the
measure of tools/glm53_compare.py) is above --limit. For the first position, every layer's numbers. Exit 0 when no
point is above the limit.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

STREAMS = ("hc_attn_post", "l_out")   # hc x n_embd; every other point is n_embd


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("llama", help="llama-strata-ref --dump-layers output")
    ap.add_argument("strata", help="strata-glm-ref --dump-layers output")
    ap.add_argument("--layers", type=int, required=True,
                    help="layers per position (the engine's: 45 for the real file)")
    ap.add_argument("--embd", type=int, required=True, help="n_embd (4096 for the real file)")
    ap.add_argument("--hc", type=int, default=4, help="hyper-connection streams (4)")
    ap.add_argument("--names", default="hc_attn_post,l_out", help="the points each layer holds, in order")
    ap.add_argument("--limit", type=float, default=1e-3, help="relative difference that counts as parted")
    ap.add_argument("--positions", type=int, default=0, help="compare at most this many positions (0: all)")
    a = ap.parse_args()

    names = [n for n in a.names.split(",") if n]
    sizes = [a.hc * a.embd if n in STREAMS else a.embd for n in names]
    per_layer = sum(sizes)
    per_pos = a.layers * per_layer
    counts = [os.path.getsize(p) // 4 for p in (a.llama, a.strata)]
    for p, n in zip((a.llama, a.strata), counts):
        if n % per_pos:
            sys.exit(f"{p}: {n} floats is not a whole number of positions of {per_pos} ({a.layers} layers of "
                     f"{', '.join(names)}): check --layers, --embd and --names")
    n_pos = min(counts) // per_pos
    if counts[0] != counts[1]:
        print(f"note: the files hold {counts[0] // per_pos} and {counts[1] // per_pos} positions; comparing {n_pos}")
    if a.positions:
        n_pos = min(n_pos, a.positions)
    ref = np.fromfile(a.llama, dtype="<f4", count=n_pos * per_pos).reshape(n_pos, a.layers, per_layer)
    got = np.fromfile(a.strata, dtype="<f4", count=n_pos * per_pos).reshape(n_pos, a.layers, per_layer)
    rel = np.empty((n_pos, a.layers, len(names)))
    at = 0
    for k, size in enumerate(sizes):
        r = ref[..., at:at + size].astype(np.float64)
        g = got[..., at:at + size].astype(np.float64)
        rel[..., k] = np.abs(r - g).max(axis=-1) / np.maximum(np.abs(r).max(axis=-1), 1e-30)
        at += size

    flat = rel.reshape(n_pos, -1)
    first0 = int(np.argmax(flat[0] > a.limit)) if (flat[0] > a.limit).any() else -1
    print(f"position 0, relative difference per layer ({' / '.join(names)}; limit {a.limit:g}):")
    for il in range(a.layers):
        row = "   ".join(f"{rel[0, il, k]:.3e}" for k in range(len(names)))
        flag = "  <- first above the limit" if first0 >= 0 and first0 // len(names) == il else ""
        print(f"  layer {il:3d}   {row}{flag}")
    bad = 0
    print("first point above the limit, per position:")
    for p in range(n_pos):
        over = np.flatnonzero(flat[p] > a.limit)
        if over.size == 0:
            print(f"  position {p}: none (worst {flat[p].max():.3e})")
            continue
        bad += 1
        il, k = divmod(int(over[0]), len(names))
        print(f"  position {p}: layer {il} {names[k]}, {rel[p, il, k]:.3e} (worst {flat[p].max():.3e})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
