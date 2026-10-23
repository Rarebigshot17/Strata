#!/usr/bin/env python3
"""Tests for tools/nsys_busy.py on a small hand-made profile in nsys's SQLite layout."""
from __future__ import annotations

import io
import sqlite3
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nsys_busy  # noqa: E402

S = 1_000_000_000   # ns per second


def make_profile(path: str) -> None:
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    db.executemany("INSERT INTO StringIds VALUES (?, ?)", [(1, "gemm_a"), (2, "softmax"), (3, "chunk 0+1024")])
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, end INT, deviceId INT, shortName INT)")
    # card 0: kernels 0-2 s and 1-3 s (overlapping: 3 s busy), card 1: 4-5 s
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
                   [(0, 2 * S, 0, 1), (1 * S, 3 * S, 0, 2), (4 * S, 5 * S, 1, 1)])
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INT, end INT, deviceId INT, copyKind INT, bytes INT)")
    # card 0 copies 10 GB from RAM over 2-6 s (1 s of it under its kernels); card 1 sends 1 GB to RAM over 9-10 s
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, ?, ?)",
                   [(2 * S, 6 * S, 0, 1, 10 * 10**9), (9 * S, 10 * S, 1, 2, 10**9)])
    db.execute("CREATE TABLE NVTX_EVENTS (start INT, end INT, text TEXT, textId INT)")
    db.executemany("INSERT INTO NVTX_EVENTS VALUES (?, ?, ?, ?)",
                   [(0, 10 * S, None, 3), (5 * S, 6 * S, "routing: wait for the GPU", None)])
    db.commit()
    db.close()


class NsysBusyTest(unittest.TestCase):
    def test_merge_and_intersect(self):
        self.assertEqual(nsys_busy.merge([(5, 6), (0, 2), (1, 3)]), [(0, 3), (5, 6)])
        self.assertEqual(nsys_busy.total([(0, 3), (5, 6)]), 4)
        self.assertEqual(nsys_busy.intersect([(0, 3), (5, 6)], [(2, 5)]), 1)
        self.assertEqual(nsys_busy.intersect([(0, 1)], [(1, 2)]), 0)

    def test_report(self):
        with tempfile.TemporaryDirectory() as d:
            p = str(Path(d) / "p.sqlite")
            make_profile(p)
            sys.argv = ["nsys_busy.py", p]
            out = io.StringIO()
            with redirect_stdout(out):
                rc = nsys_busy.main()
        text = out.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("profile window: 10.000 s", text)
        self.assertIn("kernels              3.000 s   30.0% of the window", text)          # card 0
        self.assertIn("RAM->card            4.000 s   40.0%     10.00 GB    2.50 GB/s", text)
        self.assertIn("RAM->card while kernels ran: 1.000 s (25% of that copying)", text)
        self.assertIn("idle (no kernel, no copy)    4.000 s   40.0%", text)                 # card 0: 6-10 s
        self.assertIn("card->RAM            1.000 s   10.0%      1.00 GB    1.00 GB/s", text)  # card 1
        # computing at once: 0-3 s one card, 3-4 s none, 4-5 s one card, 5-10 s none
        self.assertIn("  0:    6.000 s   60.0%", text)
        self.assertIn("  1:    4.000 s   40.0%", text)
        self.assertIn("chunk 0+1024", text)
        self.assertIn("102.4 tokens/s", text)
        self.assertIn("routing: wait for the GPU", text)
        self.assertIn("gemm_a", text)


if __name__ == "__main__":
    unittest.main()
