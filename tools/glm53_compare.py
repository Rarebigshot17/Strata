#!/usr/bin/env python3
"""Compare Strata's logits with llama.cpp's for the GLM-5.3-Flash port.

Both files use the layout `strata generate --dump-logits` writes and `llama-strata-ref --out` copies:
int32 n_vocab, int32 n_rows, then n_rows x n_vocab float32.

Reports, per position and in total:
  KL(ref || strata)   from the softmax of each row; the main number
  top-1 agreement     whether both pick the same most likely token
  max |logit diff|

Usage:  python tools/glm53_compare.py REF.bin STRATA.bin [--kl-max 0.01] [--top1-min 0.98] [--tie-margin 0.01]
                                      [--diff-max D] [--rows N]

Exit code 0 when mean KL <= --kl-max, top-1 agreement >= --top1-min and (with --diff-max) max |logit diff| <= D,
1 otherwise. The defaults are a first guess for two engines running the same quantized file; tighten them once the
first layers match.

A row where the two pick different tokens is a tie, and counts as agreeing, when each engine has the other's token
within --tie-margin of its own: then the two tokens are about as likely as each other (a logit gap of 0.01 is a 1%
difference in probability) and rounding decides which comes first. Measured on seeds 1-100 of glm53_selftest.sh's
models (strata-glm-ref against llama.cpp's CPU build with an f32 cache, 2026-10-20): every top-1 disagreement had
both gaps at most 0.0017. With llama.cpp's default f16 cache, which made the indexer select other pools in 5 of
those seeds, the rows that changed had one gap of 0.08 or more. A tie does not excuse the other limits: those 5
runs fail the selftest's mean KL limit (by 43 times or more) and its --diff-max 0.01 (their largest logit
difference is 0.15 or more) with or without ties.
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
    ap.add_argument("--tie-margin", type=float, default=0.01,
                    help="a top-1 disagreement is a tie when each side has the other's token within this many logits "
                         "of its own (0: no ties)")
    ap.add_argument("--diff-max", type=float, default=None, help="also fail when max |logit diff| is above this")
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
    # how far below its own top-1 each side puts the other's (0 where they agree; NaN never makes a tie)
    rows = np.arange(n)
    gap_r = ref[rows, top_r].astype(np.float64) - ref[rows, top_g]
    gap_g = got[rows, top_g].astype(np.float64) - got[rows, top_r]
    tie = ~agree & (gap_r <= a.tie_margin) & (gap_g <= a.tie_margin)

    print(f"{'pos':>5} {'KL':>10} {'top1 ref':>9} {'top1 strata':>11} {'max|d|':>9}")
    for i in range(n):
        if n <= 64 or i < 16 or i >= n - 16 or not agree[i] or kl[i] > a.kl_max:
            mark = "" if agree[i] else f"  <- top-1 differs (gaps {gap_r[i]:.4f} ref, {gap_g[i]:.4f} strata)"
            if tie[i]:
                mark += ": a tie"
            print(f"{i:5d} {kl[i]:10.5f} {top_r[i]:9d} {top_g[i]:11d} {diff[i]:9.4f}{mark}")
    mean_kl = float(np.nanmean(kl))
    n_tie = int(tie.sum())
    top1 = float((agree | tie).mean())
    ties = f" ({top1 * 100:.2f}% with {n_tie} tie{'s' if n_tie > 1 else ''})" if n_tie else ""
    print(f"\nrows {n}   mean KL {mean_kl:.5f}   max KL {float(np.nanmax(kl)):.5f}   "
          f"top-1 agreement {float(agree.mean()) * 100:.2f}%{ties}   max |logit diff| {float(diff.max()):.4f}")
    ok = mean_kl <= a.kl_max and top1 >= a.top1_min and not bad.any()
    needs = f"mean KL <= {a.kl_max} and top-1 >= {a.top1_min * 100:.0f}%"
    if a.diff_max is not None:
        ok = ok and float(diff.max()) <= a.diff_max
        needs += f" and max |logit diff| <= {a.diff_max}"
    print("PASS" if ok else f"FAIL (needs {needs})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
