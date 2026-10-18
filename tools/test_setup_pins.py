"""Tests for setup.py's reproducible installs (#214): the pinned requirements file (an existing install left as it
is), and the resumable download of the model's files.  Mocked network - nothing is downloaded.

    python -m unittest tools.test_setup_pins
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402


class Response(io.BytesIO):
    def __init__(self, body=b"", status=200):
        super().__init__(body)
        self.status = status
        self.headers = {"Content-Length": str(len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def quiet(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*args, **kw), out.getvalue()


class Download(unittest.TestCase):
    def test_a_complete_part_is_finished_without_a_request(self):
        """A .part with every byte (setup stopped between the last byte and the rename): renamed, not resumed with a
        range past its end - the server answers that with 416, which download() retried 30 times, 10 s apart."""
        seen = []

        def urlopen(req, timeout=None):
            seen.append((req.get_method(), req.headers.get("Range")))
            if req.get_method() == "HEAD":
                return Response(b"model bytes")
            raise urllib.error.HTTPError(req.full_url, 416, "Range Not Satisfiable", {}, None)

        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup.urllib.request, "urlopen", urlopen), \
                mock.patch.object(setup.time, "sleep", lambda s: None):
            dst = Path(d) / "m.gguf"
            dst.with_name("m.gguf.part").write_bytes(b"model bytes")
            quiet(setup.download, "https://example.com/m.gguf", dst)
            self.assertEqual(dst.read_bytes(), b"model bytes")
            self.assertTrue(setup.done(dst))
        self.assertEqual(seen, [("HEAD", None)])

    def test_a_part_is_resumed_with_a_range(self):
        """A .part from a run that stopped half-way: the rest is asked for from its length on, and appended."""
        seen = []

        def urlopen(req, timeout=None):
            seen.append((req.get_method(), req.headers.get("Range")))
            if req.get_method() == "HEAD":
                return Response(b"model bytes")
            return Response(b"model bytes"[6:], status=206)

        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup.urllib.request, "urlopen", urlopen):
            dst = Path(d) / "m.gguf"
            dst.with_name("m.gguf.part").write_bytes(b"model ")
            quiet(setup.download, "https://example.com/m.gguf", dst)
            self.assertEqual(dst.read_bytes(), b"model bytes")
            self.assertFalse(dst.with_name("m.gguf.part").exists())
            self.assertTrue(setup.done(dst))
            seen.clear()
            _, out = quiet(setup.download, "https://example.com/m.gguf", dst)  # finished: no request at all
        self.assertEqual(seen, [])
        self.assertIn("already downloaded", out)

    def test_a_wrong_size_is_deleted(self):
        with tempfile.TemporaryDirectory() as d:
            s = Path(d) / "m.gguf"
            s.write_bytes(b"short")
            setup.mark(s)
            with self.assertRaises(SystemExit):
                quiet(setup.verify_file, s, 11, None)
            self.assertFalse(s.exists())
            self.assertFalse(setup.done(s))                                  # the next run downloads it again

    def test_the_sha256_is_checked_once(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            s = Path(d) / "m.gguf"
            s.write_bytes(b"model bytes")
            sha = hashlib.sha256(b"model bytes").hexdigest()
            setup.mark(s)
            _, out = quiet(setup.verify_file, s, 11, sha)
            self.assertIn("checking m.gguf (SHA-256", out)
            _, out = quiet(setup.verify_file, s, 11, sha)
            self.assertEqual(out, "")                                        # kept in the finish mark
            with self.assertRaises(SystemExit):
                quiet(setup.verify_file, s, 11, "0" * 64)
            self.assertFalse(s.exists())


class Requirements(unittest.TestCase):
    def test_every_package_is_pinned(self):
        lines = setup.requirement_lines()
        names = {setup.req_name(x) for x in lines}
        for p in setup.PY_PACKAGES:
            self.assertIn(p, names)
        for x in lines:
            self.assertIn("==", x, x)
        self.assertEqual(setup.req_name('numpy==2.5.3; python_version >= "3.12"'), "numpy")
        self.assertEqual(setup.req_name("charset_normalizer==3"), "charset-normalizer")

    def pip(self, stamp, packages, installed=()):
        ran = []
        with tempfile.TemporaryDirectory() as d:
            if stamp is not None:
                (Path(d) / ".strata-pip.json").write_text(json.dumps(stamp))
            with mock.patch.object(setup.sys, "prefix", d), \
                    mock.patch.object(setup, "run", lambda cmd, **kw: ran.append(cmd)), \
                    mock.patch.object(setup, "_installed", lambda name: name in installed):
                quiet(setup.pip_install, packages, "the packages")
            after = json.loads((Path(d) / ".strata-pip.json").read_text()) if ran else stamp
        return [c for cmd in ran for c in cmd if "==" in c or c in setup.PY_PACKAGES], after

    def test_fresh_install_gets_every_pin(self):
        lines = setup.requirement_lines()
        ran, stamp = self.pip(None, lines)
        self.assertEqual(ran, lines)
        self.assertEqual(sorted(stamp), sorted(lines))
        self.assertEqual(self.pip(stamp, lines)[0], [])                       # the second run: nothing

    def test_an_install_from_before_the_pins_is_left_alone(self):
        lines = setup.requirement_lines()
        legacy = sorted(setup.PY_PACKAGES) + ["nvidia-cublas==13.0.2.14"]
        deps = {"markupsafe", "certifi", "charset-normalizer", "idna", "urllib3", "colorama"}
        self.assertEqual(self.pip(legacy, lines, installed=deps)[0], [])
        missing = [p for p in legacy if p != "psutil"]                         # an older list without psutil
        ran, _ = self.pip(missing, lines, installed=deps)
        self.assertEqual(ran, ["psutil==7.2.2"])

    def test_a_changed_pin_is_installed(self):
        lines = setup.requirement_lines()
        old = [x.replace("tqdm==4.70.1", "tqdm==4.60.0") for x in lines]
        ran, _ = self.pip(old, lines)
        self.assertEqual(ran, ["tqdm==4.70.1"])


if __name__ == "__main__":
    unittest.main()
