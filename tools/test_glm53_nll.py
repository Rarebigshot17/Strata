#!/usr/bin/env python3
"""Tests of tools/glm53_nll.py on small hand-made logit files: no model, no GPU."""
from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import glm53_nll  # noqa: E402

VOCAB, ROWS = 32, 1200


def write(path: Path, logits: np.ndarray) -> None:
    with open(path, "wb") as fh:
        fh.write(struct.pack("<ii", logits.shape[1], logits.shape[0]))
        fh.write(logits.astype("<f4").tobytes())


class Nll(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        rng = np.random.default_rng(1)
        self.ids = [int(i) for i in rng.integers(0, VOCAB, ROWS)]
        (self.tmp / "ids.txt").write_text(",".join(map(str, self.ids)))
        self.ref = rng.normal(0, 1, (ROWS, VOCAB)).astype(np.float32)
        for i in range(ROWS - 1):                   # a model that mostly predicts the next token
            self.ref[i, self.ids[i + 1]] += 3.0
        write(self.tmp / "ref.bin", self.ref)

    def run_tool(self, strata: np.ndarray, *extra: str):
        write(self.tmp / "strata.bin", strata)
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "glm53_nll.py"), str(self.tmp / "ids.txt"),
                            str(self.tmp / "ref.bin"), str(self.tmp / "strata.bin"), *extra],
                           capture_output=True, text=True, timeout=120)
        return r.returncode, r.stdout + r.stderr

    def test_the_nll_is_the_next_tokens(self):
        x = self.ref[:3].astype(np.float64)
        want = [np.log(np.exp(x[i]).sum()) - x[i, self.ids[i + 1]] for i in range(3)]
        got = glm53_nll.next_token_nll(self.ref[:3], self.ids[:4])
        np.testing.assert_allclose(got, want, rtol=1e-9)
        self.assertEqual(len(glm53_nll.next_token_nll(self.ref, self.ids)), ROWS - 1)

    def test_the_same_logits_pass(self):
        rc, out = self.run_tool(self.ref)
        self.assertEqual(rc, 0, out)
        self.assertIn("strata - ref: +0.0000", out)
        self.assertIn("PASS", out)

    def test_rounding_noise_passes(self):
        noise = np.random.default_rng(2).normal(0, 0.05, self.ref.shape).astype(np.float32)
        rc, out = self.run_tool(self.ref + noise)
        self.assertEqual(rc, 0, out)

    def test_a_worse_engine_fails(self):
        worse = self.ref.copy()
        for i in range(ROWS - 1):                   # every next token half a nat less likely, about
            worse[i, self.ids[i + 1]] -= 0.5
        rc, out = self.run_tool(worse)
        self.assertEqual(rc, 1, out)
        self.assertIn("FAIL", out)

    def test_from_scores_only_the_positions_past_it(self):
        # worse only before position 600: it fails over the whole text and passes from 600 on
        worse = self.ref.copy()
        for i in range(600):
            worse[i, self.ids[i + 1]] -= 2.0
        rc, out = self.run_tool(worse)
        self.assertEqual(rc, 1, out)
        rc, out = self.run_tool(worse, "--from", "600")
        self.assertEqual(rc, 0, out)
        self.assertIn(f"tokens {ROWS - 1 - 600} (positions 600 to {ROWS - 2})", out)
        self.assertIn("strata - ref: +0.0000", out)

    def test_strata_first_reads_a_file_that_starts_later(self):
        # strata-glm --prefill --prompt-tokens 601: its rows are positions 600 on
        rc, out = self.run_tool(self.ref[600:], "--strata-first", "600")
        self.assertEqual(rc, 0, out)
        self.assertIn(f"tokens {ROWS - 1 - 600} (positions 600 to {ROWS - 2})", out)
        self.assertIn("strata - ref: +0.0000", out)
        # read as position 0 instead, every row is matched with the wrong next token
        rc, out = self.run_tool(self.ref[600:])
        self.assertEqual(rc, 1, out)
        # --from past the first row: both files from there
        rc, out = self.run_tool(self.ref[600:], "--strata-first", "600", "--from", "1000")
        self.assertEqual(rc, 0, out)
        self.assertIn(f"tokens {ROWS - 1 - 1000} (positions 1000 to {ROWS - 2})", out)
        self.assertIn("strata - ref: +0.0000", out)

    def test_a_short_text_says_its_interval_is_wide(self):
        noise = np.random.default_rng(3).normal(0, 1.0, self.ref.shape).astype(np.float32)
        write(self.tmp / "ref.bin", self.ref[:40])
        (self.tmp / "ids.txt").write_text(",".join(map(str, self.ids[:40])))
        rc, out = self.run_tool(self.ref[:40] + noise[:40])
        self.assertEqual(rc, 1, out)
        self.assertIn("the interval wide", out)


if __name__ == "__main__":
    unittest.main()
