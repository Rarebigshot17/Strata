"""tools/glm53_compare.py on small hand-made logit files: when a top-1 disagreement is a tie, and that the KL and
logit-difference limits still fail a real mismatch.

    python3 tools/test_glm53_compare.py
"""
from __future__ import annotations

import math
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPARE = ROOT / "tools" / "glm53_compare.py"
try:
    import numpy  # noqa: F401  (glm53_compare.py needs it)
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False


def rows40():
    """40 rows of 8 logits, each row's values 0.05 apart, so every row has one clear top-1."""
    return [[0.05 * ((r * 7 + i * 3) % 11) for i in range(8)] for r in range(40)]


def top2(row):
    order = sorted(range(len(row)), key=lambda i: -row[i])
    return order[0], order[1]


def write(path, rows):
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", len(rows[0]), len(rows)))
        for r in rows:
            f.write(struct.pack(f"<{len(r)}f", *r))


@unittest.skipUnless(HAVE_NUMPY, "glm53_compare.py needs numpy")
class Compare(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def run_compare(self, ref, got, *args):
        write(self.tmp / "ref.bin", ref)
        write(self.tmp / "got.bin", got)
        r = subprocess.run([sys.executable, str(COMPARE), str(self.tmp / "ref.bin"), str(self.tmp / "got.bin"), *args],
                           capture_output=True, text=True, timeout=60)
        return r.returncode, r.stdout + r.stderr

    def near_tie(self, row, ref_gap, got_gap):
        """Logits for one row whose two top tokens are ref_gap apart in ref and swapped, got_gap apart, in got."""
        ref, got = rows40(), rows40()
        a, b = top2(ref[row])
        hi = ref[row][a]
        ref[row][b] = hi - ref_gap
        got[row][b] = hi
        got[row][a] = hi - got_gap
        return ref, got

    # the selftest's limits
    TIGHT = ("--kl-max", "1e-6", "--top1-min", "1.0")

    def test_the_same_logits_pass_without_ties(self):
        rc, out = self.run_compare(rows40(), rows40(), *self.TIGHT)
        self.assertEqual(rc, 0, out)
        self.assertIn("top-1 agreement 100.00%", out)
        self.assertNotIn("tie", out)

    def test_two_tokens_within_the_margin_on_both_sides_are_a_tie(self):
        # gaps of the size the selftest's noise makes (llama.cpp's f32 cache against Strata's f16 latent, q4_0)
        ref, got = self.near_tie(7, 0.0011, 0.0006)
        rc, out = self.run_compare(ref, got, *self.TIGHT)
        self.assertEqual(rc, 0, out)
        self.assertIn("top-1 agreement 97.50% (100.00% with 1 tie)", out)
        self.assertIn("a tie", next(ln for ln in out.splitlines() if ln.split()[:1] == ["7"]))
        # without the rule the same files fail
        rc, out = self.run_compare(ref, got, *self.TIGHT, "--tie-margin", "0")
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL", out)

    def test_a_clear_gap_on_one_side_is_not_a_tie(self):
        # llama.cpp sure of its token, Strata nearly tied: what a different indexer selection made (seed 26, row 27)
        ref, got = self.near_tie(7, 0.06, 0.003)
        rc, out = self.run_compare(ref, got, "--kl-max", "1", "--top1-min", "1.0")
        self.assertEqual(rc, 1, out)
        self.assertIn("top-1 agreement 97.50%", out)
        self.assertNotIn("tie", out)
        ref, got = self.near_tie(7, 0.003, 0.06)
        rc, out = self.run_compare(ref, got, "--kl-max", "1", "--top1-min", "1.0")
        self.assertEqual(rc, 1, out)

    def test_a_tie_is_between_the_same_two_tokens(self):
        # each side has a close second, but Strata's top-1 is far down in llama.cpp's row
        ref, got = rows40(), rows40()
        a, b = top2(ref[3])
        c = min(range(8), key=lambda i: ref[3][i])
        hi = ref[3][a]
        ref[3][b] = hi - 0.001
        got[3][c] = hi + 0.5
        got[3][a] = hi + 0.4995
        rc, out = self.run_compare(ref, got, "--kl-max", "1", "--top1-min", "1.0")
        self.assertEqual(rc, 1, out)
        self.assertNotIn("tie", out)

    def test_a_tie_does_not_excuse_the_kl_limit(self):
        ref, got = self.near_tie(7, 0.001, 0.001)
        got[20] = [v * 3 for v in got[20]]           # same top-1, very different probabilities
        rc, out = self.run_compare(ref, got, *self.TIGHT)
        self.assertEqual(rc, 1, out)
        self.assertIn("1 tie", out)
        self.assertIn("FAIL", out)

    def test_the_logit_difference_limit(self):
        ref, got = rows40(), rows40()
        a, b = top2(got[12])
        low = min(range(8), key=lambda i: got[12][i])
        got[12][low] += 0.02                         # far below the top: top-1 and KL hardly change
        rc, out = self.run_compare(ref, got, *self.TIGHT)
        self.assertEqual(rc, 0, out)
        rc, out = self.run_compare(ref, got, *self.TIGHT, "--diff-max", "0.01")
        self.assertEqual(rc, 1, out)
        self.assertIn("max |logit diff| <= 0.01", out)
        rc, out = self.run_compare(ref, got, *self.TIGHT, "--diff-max", "0.05")
        self.assertEqual(rc, 0, out)

    def test_the_relative_difference_limit(self):
        # each row's largest |ref logit| is 0.5 here, so 0.02 is 4% of row 12's
        ref, got = rows40(), rows40()
        low = min(range(8), key=lambda i: got[12][i])
        got[12][low] += 0.02
        rc, out = self.run_compare(ref, got, *self.TIGHT, "--rel-diff-max", "0.05")
        self.assertEqual(rc, 0, out)
        self.assertIn("worst relative diff 0.04", out)
        rc, out = self.run_compare(ref, got, *self.TIGHT, "--rel-diff-max", "0.03")
        self.assertEqual(rc, 1, out)
        self.assertIn("worst relative diff <= 0.03", out)
        # without the option the summary is as before
        rc, out = self.run_compare(ref, got, *self.TIGHT)
        self.assertNotIn("relative", out)

    def test_a_nan_fails_the_relative_difference_limit(self):
        ref, got = rows40(), rows40()
        got[9][2] = math.nan
        rc, out = self.run_compare(ref, got, "--kl-max", "1", "--top1-min", "0", "--rel-diff-max", "100")
        self.assertEqual(rc, 1, out)
        self.assertIn("NaN", out)

    def test_a_row_of_nan_is_neither_a_tie_nor_a_pass(self):
        ref, got = rows40(), rows40()
        got[9] = [math.nan] * 8
        rc, out = self.run_compare(ref, got, "--kl-max", "1", "--top1-min", "0.9", "--tie-margin", "100")
        self.assertEqual(rc, 1, out)
        self.assertIn("NaN", out)
        self.assertNotIn("tie", out)


if __name__ == "__main__":
    unittest.main()
