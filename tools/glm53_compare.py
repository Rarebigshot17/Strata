#!/usr/bin/env python3
"""Compare Strata's logits with llama.cpp's for the GLM-5.3-Flash port.

Both files use the layout `strata generate --dump-logits` writes and `llama-strata-ref --out` copies:
int32 n_vocab, int32 n_rows, then n_rows x n_vocab float32.

Reports, per position and in total:
  KL(ref || strata)   from the softmax of each row; the main number
  top-1 agreement     whether both pick the same most likely token
  max |logit diff|

Usage:  python tools/glm53_compare.py REF.bin STRATA.bin [--kl-max 0.01] [--top1-min 0.98] [--tie-margin 0.01]
                                      [--diff-max D] [--rel-diff-max R] [--rows N] [--tie-rows LIST]

Exit code 0 when mean KL <= --kl-max, top-1 agreement >= --top1-min, (with --diff-max) max |logit diff| <= D and
(with --rel-diff-max) the worst row's max |logit diff| over its largest |ref logit| <= R, 1 otherwise. The defaults
are a first guess for two engines running the same quantized file; tighten them once the first layers match. The
relative diff is the measure of strata-glm's GPU check (include/strata/glm/check.hpp), and tools/glm53_gpu_check.sh
holds this comparison to that rule.

A row where the two pick different tokens is a tie, and counts as agreeing, when each engine has the other's token
within --tie-margin of its own: then the two tokens are about as likely as each other (a logit gap of 0.01 is a 1%
difference in probability) and rounding decides which comes first. Measured on seeds 1-100 of glm53_selftest.sh's
models (strata-glm-ref against llama.cpp's CPU build with an f32 cache, 2026-10-20): every top-1 disagreement had
both gaps at most 0.0017. With llama.cpp's default f16 cache, which made the indexer select other pools in 5 of
those seeds, the rows that changed had one gap of 0.08 or more, all but one: seed 26's v2 also swapped a row whose
gaps were 0.0017 and 0.0007, a tie. A tie does not excuse the other limits: those 5 runs fail the selftest's mean
KL limit (by 43 times or more) and its --diff-max 0.01 (their largest logit difference is 0.15 or more) with or
without ties.

--tie-rows names positions where the reference may have taken another branch than strata at a near tie that moves
the logits a lot (glm53_gpu_check.sh: where strata-glm's CPU reference had the MTP block's indexer pool scores within
10% at the cut, so llama.cpp's own arithmetic may keep the other pool). A listed row that is off by itself, its KL
above --kl-max or its difference above --diff-max or --rel-diff-max, is left out of every figure and named; the others
count as usual. More than a quarter of the rows left out fails.
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


def row_stats(ref: np.ndarray, got: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each row's KL(ref || got), max |logit diff|, and that diff over the row's largest |ref logit| (NaN for a row
    with a NaN)."""
    lr, lg = log_softmax(ref), log_softmax(got)
    kl = (np.exp(lr) * (lr - lg)).sum(axis=1)
    diff = np.abs(ref.astype(np.float64) - got).max(axis=1)
    return kl, diff, diff / np.maximum(np.abs(ref.astype(np.float64)).max(axis=1), 1e-30)


def positions(ps: list[int]) -> str:
    return f"position{'s' if len(ps) != 1 else ''} {', '.join(map(str, ps))}" if ps else "none"


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
    ap.add_argument("--rel-diff-max", type=float, default=None,
                    help="also fail when a row's max |logit diff| is above this times its largest |ref logit|")
    ap.add_argument("--rows", type=int, default=0, help="compare only the first N rows")
    ap.add_argument("--tie-rows", default="",
                    help="positions (from 0, comma separated) of near ties: each one off by itself is left out of "
                         "every figure and named in the output")
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
    pos = np.arange(n)                       # each row's position, as the tables print it
    n_all, out = n, []
    if a.tie_rows:
        listed = sorted({int(x) for x in a.tie_rows.split(",") if x.strip()} & set(range(n)))
        kl, diff, rel_row = row_stats(ref, got)
        off = kl > a.kl_max                  # NaN is never off by itself, so never left out
        if a.diff_max is not None:
            off |= diff > a.diff_max
        if a.rel_diff_max is not None:
            off |= rel_row > a.rel_diff_max
        drop = off & np.isin(pos, listed)
        out = pos[drop].tolist()
        ref, got, pos = ref[~drop], got[~drop], pos[~drop]
        n = len(pos)
        print(f"near ties (--tie-rows): {positions(listed)}; "
              + (f"left out, off by itself: {positions(out)}" if out else "none off by itself, none left out"))
    left = f" ({len(out)} left out)" if out else ""

    bad = ~np.isfinite(got).all(axis=1)
    if bad.any():
        print(f"strata has NaN/inf in {int(bad.sum())} row(s), first at position {int(pos[np.argmax(bad)])}")

    kl, diff, rel_row = row_stats(ref, got)
    top_r, top_g = ref.argmax(axis=1), got.argmax(axis=1)
    agree = top_r == top_g
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
            print(f"{pos[i]:5d} {kl[i]:10.5f} {top_r[i]:9d} {top_g[i]:11d} {diff[i]:9.4f}{mark}")
    mean_kl = float(np.nanmean(kl))
    n_tie = int(tie.sum())
    top1 = float((agree | tie).mean())
    ties = f" ({top1 * 100:.2f}% with {n_tie} tie{'s' if n_tie > 1 else ''})" if n_tie else ""
    worst_rel = float(rel_row.max())
    rel = f"   worst relative diff {worst_rel:.3g}" if a.rel_diff_max is not None else ""
    print(f"\nrows {n}{left}   mean KL {mean_kl:.5f}   max KL {float(np.nanmax(kl)):.5f}   "
          f"top-1 agreement {float(agree.mean()) * 100:.2f}%{ties}   max |logit diff| {float(diff.max()):.4f}{rel}")
    ok = mean_kl <= a.kl_max and top1 >= a.top1_min and not bad.any()
    # --top1-min 0 leaves top-1 agreement out of the rule (glm53_gpu_check.sh), and out of what a FAIL asks for
    needs = f"mean KL <= {a.kl_max}" + (f" and top-1 >= {a.top1_min * 100:.0f}%" if a.top1_min > 0 else "")
    if a.diff_max is not None:
        ok = ok and float(diff.max()) <= a.diff_max
        needs += f" and max |logit diff| <= {a.diff_max}"
    if a.rel_diff_max is not None:
        ok = ok and worst_rel <= a.rel_diff_max
        needs += f" and worst relative diff <= {a.rel_diff_max}"
    if out:
        ok = ok and len(out) * 4 <= n_all
        needs += " and at most a quarter of the rows left out"
    print("PASS" if ok else f"FAIL (needs {needs})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
