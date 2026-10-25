#!/usr/bin/env python3
"""How well two engines predict the same text, for the GLM-5.3-Flash port: the mean negative log-likelihood of each
next token (teacher-forced), and the paired difference with a bootstrap 95% interval.

    python tools/glm53_nll.py IDS.txt REF.bin STRATA.bin [--max-diff 0.02] [--from N] [--strata-first K]

REF.bin and STRATA.bin are logits in the layout tools/glm53_compare.py reads (llama-strata-ref --out,
strata-glm --out, strata-glm-ref --out), one row per position of IDS.txt (llama-strata-ref --ids-out). --from N
scores the positions from N on only: past the indexer's dense range (2,051 tokens of context in the real file, so
from position 2051, counted from 0), the positions where it selects which pools to attend to. --strata-first K reads STRATA.bin's first row as position K:
`strata-glm --prefill --prompt-tokens P` writes the prompt path's last row (position P - 1) and then the rows of the
tokens it decodes one at a time. The files are read as they are needed (memory-mapped): 16,384 rows of the real
file's vocabulary are 10 GB.

Why not the logits themselves: the real file's weights are 2- to 3-bit (IQ2_S, IQ3_S) and every product rounds its
activations to 8 bits (Q8_K, Q8_0 on the CPU, q8_1 on the GPU). A difference of 1e-7 between two engines flips the
rounding of a few activations, and one flip moves a layer's output by 1% (napster-ai, 2026-10-23: layer 0's dense
FFN, one flipped Q8_K value of 4,096 gave the whole 1.4e-2 between Strata's CPU reference and llama.cpp's CPU path,
each engine exact against a float64 recomputation from its own input). Through 45 layers that gives a mean KL of
0.03 to 0.05 between any two paths, llama.cpp's own CPU and GPU paths included (0.0375), so glm53_compare.py's
logit limits cannot pass on the real file. How well each predicts the text is what matters: on 1,953 tokens Strata's
CPU reference and llama.cpp's CPU path gave a mean NLL of 1.7211 and 1.7238, a paired difference of -0.0028 (95%
interval -0.0137 to +0.0091).

Exit 0 when the interval's upper end is below --max-diff: STRATA.bin predicts the text no worse than REF.bin by more
than that (nats per token). The default, 0.02, is about 1% of the mean NLL on the port's texts (a perplexity 2%
higher at most); it needs a text of 1,000 tokens or more for an interval that narrow.
"""
from __future__ import annotations

import argparse
import struct
import sys

import numpy as np


def load(path: str) -> np.ndarray:
    with open(path, "rb") as fh:
        n_vocab, n_rows = struct.unpack("<ii", fh.read(8))
    data = np.memmap(path, dtype="<f4", mode="r", offset=8)
    if data.size != n_vocab * n_rows:
        sys.exit(f"{path}: header says {n_rows} x {n_vocab}, file holds {data.size} floats")
    return data.reshape(n_rows, n_vocab)


def next_token_nll(logits: np.ndarray, ids: list[int], first: int = 0) -> np.ndarray:
    """-log p(ids[first + j + 1] | logits row j), in float64: row j is position first + j, one value per row whose
    position has a next token."""
    n = max(0, min(len(logits), len(ids) - 1 - first))     # row j needs ids[first + j + 1]
    out = np.empty(n)
    for start in range(0, n, 256):                         # in pieces: the rows are 154,880 wide
        x = np.asarray(logits[start:min(n, start + 256)], dtype=np.float64)
        m = x.max(axis=1, keepdims=True)
        lse = (m[:, 0] + np.log(np.exp(x - m).sum(axis=1)))
        nxt = np.asarray(ids[first + start + 1:first + start + 1 + len(x)])
        out[start:start + len(x)] = lse - x[np.arange(len(x)), nxt]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ids", help="the token ids, comma or space separated")
    ap.add_argument("ref", help="the reference engine's logits (llama.cpp)")
    ap.add_argument("strata", help="Strata's logits")
    ap.add_argument("--max-diff", type=float, default=0.02, help="largest NLL excess allowed (nats per token)")
    ap.add_argument("--seed", type=int, default=0, help="the bootstrap's seed")
    ap.add_argument("--from", dest="start", type=int, default=0, metavar="N",
                    help="score the positions from N on (past the indexer's dense range: 2051, the first position "
                         "with more than 2,051 tokens of context)")
    ap.add_argument("--strata-first", type=int, default=0, metavar="K",
                    help="STRATA.bin's first row is position K (strata-glm --prefill --prompt-tokens P: P - 1)")
    a = ap.parse_args()

    with open(a.ids) as fh:
        ids = [int(t) for t in fh.read().replace(",", " ").split()]
    ref_l, got_l = load(a.ref), load(a.strata)
    start = max(a.start, a.strata_first)                   # the first position both files hold, from --from on
    n = min(len(ref_l) - start, len(got_l) - (start - a.strata_first), len(ids) - 1 - start)
    if n <= 0:
        sys.exit(f"no position to score from {start} on (ref {len(ref_l)} rows, strata {len(got_l)} rows from "
                 f"position {a.strata_first}, {len(ids)} ids)")
    ref = next_token_nll(ref_l[start:start + n], ids, start)
    got = next_token_nll(got_l[start - a.strata_first:start - a.strata_first + n], ids, start)
    d = got - ref
    rng = np.random.default_rng(a.seed)
    boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(2000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    span = f" (positions {start} to {start + n - 1})" if start else ""
    print(f"tokens {n}{span}   mean NLL {ref.mean():.4f} (ref) {got.mean():.4f} (strata)   "
          f"perplexity {np.exp(ref.mean()):.3f} {np.exp(got.mean()):.3f}")
    print(f"strata - ref: {d.mean():+.4f} nats per token, 95% interval {lo:+.4f} .. {hi:+.4f}")
    if hi < a.max_diff:
        print(f"PASS (the interval's upper end is below {a.max_diff:g})")
        return 0
    print(f"FAIL (needs the interval's upper end below {a.max_diff:g}"
          f"{'; the text is short, the interval wide' if hi - lo > a.max_diff else ''})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
