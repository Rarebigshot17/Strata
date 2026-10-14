"""Tests for serve/server.py's effort_end_args (#458): "effort_position": "end" is used when the engine binary knows
--tail-role-token.  Fake files in a temp dir - nothing is built or run.

    python -m unittest tools.test_effort_end_args
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import server as S  # noqa: E402


class Tok:
    def encode(self, text, parse_special=False):
        return [42] if text == "system" else [1, 2]


ENGINE_WITH = b"\x7fELF ... --tail-role-token ... --serve ..."
ENGINE_WITHOUT = b"\x7fELF ... --serve ..."


class EffortEndArgs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.exe = Path(self.tmp.name) / "strata"

    def tearDown(self):
        self.tmp.cleanup()

    def call(self, cfg, exe):
        with contextlib.redirect_stdout(io.StringIO()):
            return S.effort_end_args(cfg, str(exe), Tok())

    def test_engine_with_the_option(self):
        self.exe.write_bytes(ENGINE_WITH)
        self.assertEqual(self.call({"effort_position": "end"}, self.exe), ["--tail-role-token", "42"])

    def test_engine_without_the_option(self):
        self.exe.write_bytes(ENGINE_WITHOUT)
        self.assertIsNone(self.call({"effort_position": "end"}, self.exe))

    def test_start_is_the_default(self):
        self.exe.write_bytes(ENGINE_WITH)
        self.assertIsNone(self.call({}, self.exe))
        self.assertIsNone(self.call({"effort_position": "start"}, self.exe))


if __name__ == "__main__":
    unittest.main()
