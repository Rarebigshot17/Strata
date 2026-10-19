"""Tests for `setup.py --update` (#475, update.sh): it refreshes what a start would - the Python packages and the
engine - and never starts the model.  Every outside effect is mocked: no GPU, no downloads, nothing written outside
a temp folder.

    python -m unittest tools.test_setup_update
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class Update(unittest.TestCase):
    def config(self, d: Path, **extra) -> Path:
        p = d / "strata-glm-ud-iq3_xxs.json"
        p.write_text(json.dumps({"model_name": "glm-5.3-flash", "exe": str(d / "build" / setup.EXE),
                                 "args": ["-m", str(d / "m-00001-of-00003.gguf"), "--ctx", "131072"], **extra}),
                     encoding="utf-8")
        return p

    def run_update(self, have, build=False, yes=False):
        a = mock.Mock(build=build, yes=yes)
        out = io.StringIO()
        with mock.patch.object(setup, "python_packages") as pip, \
                mock.patch.object(setup, "update_installed_engine") as eng, \
                mock.patch.object(setup, "start") as start, \
                mock.patch("subprocess.call") as call, \
                contextlib.redirect_stdout(out):
            rc = setup.update_install(have, a)
        return rc, pip, eng, start, call, out.getvalue()

    def test_refreshes_without_starting(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d))
            rc, pip, eng, start, call, out = self.run_update([p])
        self.assertEqual(rc, 0)
        pip.assert_called_once()
        eng.assert_called_once_with(force=False, yes=False)
        start.assert_not_called()
        call.assert_not_called()
        self.assertIn("strata-glm-ud-iq3_xxs.json: up to date", out)
        self.assertIn("Strata is updated", out)

    def test_build_compiles_the_engine_again(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d))
            rc, pip, eng, *_ = self.run_update([p], build=True)
        self.assertEqual(rc, 0)
        eng.assert_called_once_with(force=True, yes=False)

    def test_yes_never_asks_for_the_build_tools(self):
        """./update.sh --yes where the build tools went missing: --yes takes the recommended answer, it never waits
        for one (an agent's shell has nobody to type it).  The installed engine stays when the compile fails."""
        with tempfile.TemporaryDirectory() as d:
            build = Path(d) / "build"
            build.mkdir()
            (build / setup.EXE).write_bytes(b"")
            (build / "BUILD.json").write_text(json.dumps({"src": "older", "archs": [86]}), encoding="utf-8")
            p = self.config(Path(d))
            card = {"index": 0, "name": "RTX 3090 Ti", "vram_gb": 24.0, "arch": "86", "driver": "580.95.05"}
            asked = mock.Mock(return_value="n")
            out = io.StringIO()
            with mock.patch.object(setup, "BUILD_DIR", build), mock.patch.object(setup, "python_packages"), \
                    mock.patch.object(setup, "gpus", return_value=[card]), \
                    mock.patch.object(setup, "get_llama_cpp", return_value=Path(d) / "llama.cpp"), \
                    mock.patch.object(setup, "find_nvcc", return_value=(None, None)), \
                    mock.patch.object(setup.shutil, "which", return_value=None), \
                    mock.patch("builtins.input", asked), contextlib.redirect_stdout(out):
                self.assertEqual(setup.update_install([p], mock.Mock(build=False, yes=True)), 0)
        asked.assert_not_called()
        self.assertIn("could not compile the updated engine: starting the installed one", out.getvalue())

    def test_a_rebuild_takes_a_toolkit_the_driver_runs(self):
        """Driver 570 runs an engine built with CUDA 12 (525 or newer), not one built with CUDA 13 (580 or newer).
        The engine compiled again after a git pull takes the CUDA 12 toolkit beside a CUDA 13 one; with CUDA 13 alone
        it is not compiled (nor CUDA 13.0 installed) and the installed engine starts: a CUDA 13 build would not."""
        cuda13, cuda12 = ("/usr/local/cuda-13.0/bin/nvcc", (13, 0)), ("/usr/local/cuda-12.8/bin/nvcc", (12, 8))
        for toolkits, used in (((cuda13, cuda12), cuda12[0]), ((cuda13,), None)):
            def find_nvcc(below=None):                 # the newest of `toolkits` older than `below`
                return max((t for t in toolkits if below is None or t[1] < below), key=lambda t: t[1],
                           default=(None, None))
            with self.subTest(toolkits=toolkits), tempfile.TemporaryDirectory() as d:
                build = Path(d) / "build"
                build.mkdir()
                (build / setup.EXE).write_bytes(b"")
                (build / "BUILD.json").write_text(json.dumps({"src": "older", "archs": [86]}), encoding="utf-8")
                p = self.config(Path(d))
                card = {"index": 0, "name": "RTX 3090 Ti", "vram_gb": 24.0, "arch": "86", "driver": "570.172.08"}
                out = io.StringIO()
                with mock.patch.object(setup, "BUILD_DIR", build), mock.patch.object(setup, "python_packages"), \
                        mock.patch.object(setup, "gpus", return_value=[card]), \
                        mock.patch.object(setup, "get_llama_cpp", return_value=Path(d) / "llama.cpp"), \
                        mock.patch.object(setup, "find_nvcc", find_nvcc), \
                        mock.patch.object(setup.shutil, "which", lambda name: f"/usr/bin/{name}"), \
                        mock.patch.object(setup, "run", side_effect=AssertionError("installer started")), \
                        mock.patch.object(setup, "download", side_effect=AssertionError("download started")), \
                        mock.patch.object(setup, "cmake_build") as cmake, \
                        mock.patch.object(setup, "write_build_stamp"), contextlib.redirect_stdout(out):
                    self.assertEqual(setup.update_install([p], mock.Mock(build=False, yes=True)), 0)
                if used:
                    cmake.assert_called_once()
                    self.assertIn(f"-DCMAKE_CUDA_COMPILER={used}", cmake.call_args[0][3])
                else:
                    cmake.assert_not_called()
                    self.assertIn("the NVIDIA driver (570) is too old for CUDA 13", out.getvalue())
                    self.assertIn("starting the installed one", out.getvalue())

    def test_nothing_installed(self):
        rc, pip, eng, start, call, out = self.run_update([])
        self.assertEqual(rc, 0)
        for m in (pip, eng, start, call):
            m.assert_not_called()
        self.assertIn("run ./setup.sh to set it up", out)

    def test_a_json_that_is_no_model_config_is_skipped(self):
        """#549: a strata-glm-*.json without "args" (not written by setup) must not stop update.sh."""
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d))
            other = Path(d) / "strata-glm-notes.json"
            other.write_text(json.dumps({"note": "mine"}), encoding="utf-8")
            broken = Path(d) / "strata-glm-cut.json"
            broken.write_text("{\"args\": [", encoding="utf-8")
            rc, pip, eng, start, call, out = self.run_update([other, broken, p])
        self.assertEqual(rc, 0, out)
        self.assertIn('skipped strata-glm-notes.json (no "exe" or "args"): it is not a Strata model config', out)
        self.assertIn("skipped strata-glm-cut.json (not valid JSON)", out)
        eng.assert_called_once()                                    # the real model's engine is still refreshed
        self.assertIn("strata-glm-ud-iq3_xxs.json: up to date", out)

    def test_installed_configs_lists_only_model_configs(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d))
            (Path(d) / "strata-glm-notes.json").write_text("{}", encoding="utf-8")
            (Path(d) / "strata-iq3_s.json").write_text(json.dumps({"exe": "x", "args": []}), encoding="utf-8")
            with mock.patch.object(setup, "ROOT", Path(d)), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.installed_configs(), [p])    # a Qwen config is no GLM model
                self.assertEqual(setup.older_configs(), [Path(d) / "strata-iq3_s.json"])

    def test_a_start_with_yes_passes_it_on(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d))
            with mock.patch.object(sys, "argv", ["setup.py", "--yes"]), \
                    mock.patch.object(setup, "data_folder", return_value=(Path(d), [])), \
                    mock.patch.object(setup, "installed_configs", return_value=[p]), \
                    mock.patch.object(setup, "update_installed_engine") as eng, \
                    mock.patch.object(setup, "start", return_value=0), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.main(), 0)
        eng.assert_called_once_with(force=False, yes=True)

    def test_main_update_never_starts(self):
        with tempfile.TemporaryDirectory() as d:
            p = self.config(Path(d))
            with mock.patch.object(sys, "argv", ["setup.py", "--update"]), \
                    mock.patch.object(setup, "data_folder", return_value=(Path(d), [])), \
                    mock.patch.object(setup, "installed_configs", return_value=[p]), \
                    mock.patch.object(setup, "update_install", return_value=0) as up, \
                    mock.patch.object(setup, "start") as start, \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.main(), 0)
        up.assert_called_once()
        start.assert_not_called()


class SettingsLine(unittest.TestCase):
    """#564: a start prints the settings it uses (the engine options without the model's paths, and the server's
    fields), so a change made by hand to strata-glm-<size>.json shows without reading the log."""

    CFG = {"exe": "x", "host": "0.0.0.0", "port": 8102, "api_key": "secret", "gpu": [0, 1], "fit_max_tokens": True,
           "args": ["-m", "/data/models/glm-ud-iq3_xxs/m-00001-of-00003.gguf", "--ctx", "131072", "--park-mib",
                    "8192", "--expert-profile", "/data/glm-ud-iq3_xxs.profile", "--vram-reserve", "1.5",
                    "--no-warm"]}

    def test_summary(self):
        s = setup.settings_summary(self.CFG)
        self.assertEqual(s, "--ctx 131072 --park-mib 8192 --vram-reserve 1.5 --no-warm; server 0.0.0.0:8102, api key "
                            "set, gpu 0,1, fit_max_tokens true")
        self.assertNotIn("secret", s)
        self.assertIn("127.0.0.1:9000", setup.settings_summary({"args": []}, 9000))
        self.assertIn("127.0.0.1:8101", setup.settings_summary({"args": []}))

    def test_a_start_prints_it(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "strata-glm"
            exe.write_bytes(b"")
            p = Path(d) / "strata-glm-ud-iq3_xxs.json"
            args = self.CFG["args"][2:]                                      # no model file here
            p.write_text(json.dumps({**self.CFG, "exe": str(exe), "gpu": 0, "args": args}), encoding="utf-8")
            out = io.StringIO()
            with mock.patch.object(setup, "gpus", lambda: []), \
                    mock.patch.object(setup.subprocess, "call", return_value=0), \
                    mock.patch.dict(setup.os.environ, {}, clear=False), \
                    contextlib.redirect_stdout(out):
                setup.os.environ.pop("STRATA_EXECV", None)
                setup.start(p, None)
        text = " ".join(out.getvalue().split())
        self.assertIn("Settings (strata-glm-ud-iq3_xxs.json): --ctx 131072 --park-mib 8192", text)
        self.assertIn("--vram-reserve 1.5", text)


if __name__ == "__main__":
    unittest.main()
