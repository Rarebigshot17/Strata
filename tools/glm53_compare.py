#!/usr/bin/env python3
"""Compare Strata's logits with llama.cpp's for the GLM-5.3-Flash port.

Both files use the layout `strata generate --dump-logits` writes and `llama-strata-ref --out` copies:
int32 n_vocab, int32 n_rows, then n_rows x n_vocab float32.

Reports, per position and in total:
  KL(ref || strata)   from the softmax of each row; the main number
  top-1 agreement     whether both pick the same most likely token
  max |logit diff|

Usage:  python tools/glm53_compare.py REF.bin STRATA.bin [--kl-max 0.01] [--top1-min 0.98] [--rows N]

Exit code 0 when mean KL <= --kl-max and top-1 agreement >= --top1-min, 1 otherwise. The defaults are a first
guess for two engines running the same quantized file; tighten them once the first layers match.
"""
from __future__ import annotations

import argparse
import struct
import sys

import numpy as np


def load(path: str) -> np.ndarray:
    with open(path, "rb") as fh:
        n_vocab, n_rows = struct.unpack("<ii", fh.read(8))
        data = np.fromfile(fh, dtype="<f4")
    if data.size != n_vocab * n_rows:
        sys.exit(f"{path}: header says {n_rows} x {n_vocab}, file holds {data.size} floats")
    return data.reshape(n_rows, n_vocab)


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("strata")
    ap.add_argument("--kl-max", type=float, default=0.01)
    ap.add_argument("--top1-min", type=float, default=0.98)
    ap.add_argument("--rows", type=int, default=0, help="compare only the first N rows")
    a = ap.parse_args()

    ref, got = load(a.ref), load(a.strata)
    if ref.shape[1] != got.shape[1]:
        print(f"vocab differs: ref {ref.shape[1]}, strata {got.shape[1]}; comparing the shared first columns")
        v = min(ref.shape[1], got.shape[1])
        ref, got = ref[:, :v], got[:, :v]
    n = min(ref.shape[0], got.shape[0])
    if ref.shape[0] != got.shape[0]:
        print(f"row counts differ: ref {ref.shape[0]}, strata {got.shape[0]}; comparing the first {n}")
    if a.rows:
        n = min(n, a.rows)
    ref, got = ref[:n], got[:n]

    bad = ~np.isfinite(got).all(axis=1)
    if bad.any():
        print(f"strata has NaN/inf in {int(bad.sum())} row(s), first at position {int(np.argmax(bad))}")

    lr, lg = log_softmax(ref), log_softmax(got)
    kl = (np.exp(lr) * (lr - lg)).sum(axis=1)
    top_r, top_g = ref.argmax(axis=1), got.argmax(axis=1)
    agree = top_r == top_g
    diff = np.abs(ref.astype(np.float64) - got).max(axis=1)

    print(f"{'pos':>5} {'KL':>10} {'top1 ref':>9} {'top1 strata':>11} {'max|d|':>9}")
    for i in range(n):
        if n <= 64 or i < 16 or i >= n - 16 or not agree[i] or kl[i] > a.kl_max:
            mark = "" if agree[i] else "  <- top-1 differs"
            print(f"{i:5d} {kl[i]:10.5f} {top_r[i]:9d} {top_g[i]:11d} {diff[i]:9.4f}{mark}")
    mean_kl = float(np.nanmean(kl))
    top1 = float(agree.mean())
    print(f"\nrows {n}   mean KL {mean_kl:.5f}   max KL {float(np.nanmax(kl)):.5f}   "
          f"top-1 agreement {top1 * 100:.2f}%   max |logit diff| {float(diff.max()):.4f}")
    ok = mean_kl <= a.kl_max and top1 >= a.top1_min and not bad.any()
    print("PASS" if ok else f"FAIL (needs mean KL <= {a.kl_max} and top-1 >= {a.top1_min * 100:.0f}%)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
