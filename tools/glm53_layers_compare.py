#!/usr/bin/env python3
"""Where Strata's CPU reference and llama.cpp part, layer by layer, for the GLM-5.3-Flash port.

Both files hold, per position and layer, the residual streams after the layer's attention and after its FFN
(2 x hc x n_embd float32 per layer, no header): `strata-glm-ref --dump-layers` and `llama-strata-ref --dump-layers`
(tools/glm53_ref; it needs -b 1 -ub 1) write them, on the same --tokens-file.

    python tools/glm53_layers_compare.py LLAMA.bin STRATA.bin --layers 45 --embd 4096 [--hc 4] [--limit 1e-3]

For each position: the first point (layer and attention or FFN) whose relative difference (max |a - b| over
max |a|, the measure of tools/glm53_compare.py) is above --limit. For the first position, every layer's two
numbers. Exit 0 when no point is above the limit.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("llama", help="llama-strata-ref --dump-layers output")
    ap.add_argument("strata", help="strata-glm-ref --dump-layers output")
    ap.add_argument("--layers", type=int, required=True,
                    help="layers per position (the engine's: 45 for the real file)")
    ap.add_argument("--embd", type=int, required=True, help="n_embd (4096 for the real file)")
    ap.add_argument("--hc", type=int, default=4, help="hyper-connection streams (4)")
    ap.add_argument("--limit", type=float, default=1e-3, help="relative difference that counts as parted")
    ap.add_argument("--positions", type=int, default=0, help="compare at most this many positions (0: all)")
    a = ap.parse_args()

    per_pos = a.layers * 2 * a.hc * a.embd
    sizes = [os.path.getsize(p) // 4 for p in (a.llama, a.strata)]
    for p, n in zip((a.llama, a.strata), sizes):
        if n % per_pos:
            sys.exit(f"{p}: {n} floats is not a whole number of positions of {per_pos} "
                     f"({a.layers} layers x 2 x {a.hc} x {a.embd}): check --layers and --embd")
    n_pos = min(sizes) // per_pos
    if sizes[0] != sizes[1]:
        print(f"note: the files hold {sizes[0] // per_pos} and {sizes[1] // per_pos} positions; comparing {n_pos}")
    if a.positions:
        n_pos = min(n_pos, a.positions)
    shape = (n_pos, a.layers, 2, a.hc * a.embd)
    ref = np.fromfile(a.llama, dtype="<f4", count=n_pos * per_pos).reshape(shape).astype(np.float64)
    got = np.fromfile(a.strata, dtype="<f4", count=n_pos * per_pos).reshape(shape).astype(np.float64)
    rel = np.abs(ref - got).max(axis=-1) / np.maximum(np.abs(ref).max(axis=-1), 1e-30)   # [pos, layer, 2]

    point = ("attention", "FFN")
    print(f"position 0, relative difference per layer (after the attention / after the FFN; limit {a.limit:g}):")
    for il in range(a.layers):
        flag = "  <- first above the limit" if rel[0, il].max() > a.limit and (rel[0, :il] <= a.limit).all() else ""
        print(f"  layer {il:3d}   {rel[0, il, 0]:.3e}   {rel[0, il, 1]:.3e}{flag}")
    bad = 0
    print("first point above the limit, per position:")
    for p in range(n_pos):
        over = np.argwhere(rel[p].reshape(-1) > a.limit)
        if over.size == 0:
            print(f"  position {p}: none (worst {rel[p].max():.3e})")
            continue
        bad += 1
        k = int(over[0, 0])
        il, which = divmod(k, 2)
        print(f"  position {p}: layer {il} after the {point[which]}, {rel[p, il, which]:.3e} "
              f"(worst {rel[p].max():.3e})")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
