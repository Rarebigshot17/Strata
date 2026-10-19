"""Setup prompts require input or --yes; EOF must not authorize system installs (#393).
No GPU, downloads or installers are used.

    python -m unittest tools.test_setup_prompts
"""
from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402


class Prompts(unittest.TestCase):
    def test_eof_stops_with_guidance(self):
        out = io.StringIO()
        with mock.patch("sys.stdin", io.StringIO()), contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as cm:
                setup.ask("Install them now?", ["y", "n"], "y", False)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("Install them now?", out.getvalue())
        self.assertIn("--yes", out.getvalue())
        self.assertIn("Setup stopped", out.getvalue())

    def test_yes_accepts_the_default_without_reading_input(self):
        with mock.patch("builtins.input", side_effect=AssertionError("unexpected prompt")):
            for default in ("y", "n"):
                with self.subTest(default=default):
                    self.assertEqual(setup.ask("Install?", ["y", "n"], default, True), default)

    def test_enter_and_explicit_answers(self):
        for answer, expected in (("\n", "y"), ("  \n", "y"), ("y\n", "y"), (" N \n", "n")):
            with self.subTest(answer=answer), mock.patch("sys.stdin", io.StringIO(answer)), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(setup.ask("Install?", ["y", "n"], "y", False), expected)

    def test_invalid_answer_reprompts_and_preserves_choice_case(self):
        out = io.StringIO()
        with mock.patch("sys.stdin", io.StringIO("invalid\nyArN\n")), contextlib.redirect_stdout(out):
            self.assertEqual(setup.ask("Method?", ["Yarn", "Linear"], "Linear", False), "Yarn")
        self.assertIn("please answer one of: Yarn, Linear", out.getvalue())

    def test_eof_after_invalid_answer_stops(self):
        with mock.patch("sys.stdin", io.StringIO("invalid\n")), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                setup.ask("Install?", ["y", "n"], "y", False)
        self.assertEqual(cm.exception.code, 1)


class BuildTools(unittest.TestCase):
    """The compiler and the CUDA toolkit are installed only on a "y" or with --yes, through apt on Ubuntu."""

    def which(self, have=()):
        """shutil.which with only these commands on PATH (and g++ once build-essential is in)."""
        on_path = set(have)

        def which(name):
            return f"/usr/bin/{name}" if name in on_path else None
        return which, on_path

    def test_eof_never_starts_an_installer(self):
        which, _ = self.which(["apt-get"])
        with mock.patch.object(setup, "find_nvcc", return_value=(None, None)), \
                mock.patch.object(setup.shutil, "which", which), \
                mock.patch.object(setup, "run", side_effect=AssertionError("installer started")) as run, \
                mock.patch.object(setup, "download", side_effect=AssertionError("download started")) as download, \
                mock.patch("sys.stdin", io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as cm:
                setup.install_build_tools([86], False)
        self.assertEqual(cm.exception.code, 1)
        run.assert_not_called()
        download.assert_not_called()

    def test_yes_installs_the_compiler_and_cuda_13_on_ubuntu(self):
        which, on_path = self.which(["apt-get"])
        nvcc = "/usr/local/cuda-13.0/bin/nvcc"
        commands = []

        def run(cmd, **kw):
            commands.append([str(c) for c in cmd])
            if "build-essential" in cmd:
                on_path.add("g++")
            return mock.Mock(returncode=0)
        with mock.patch.object(setup, "find_nvcc", side_effect=[(None, None), (nvcc, (13, 0))]), \
                mock.patch.object(setup.shutil, "which", which), \
                mock.patch.object(setup, "os_release", return_value={"ID": "ubuntu", "VERSION_ID": "24.04"}), \
                mock.patch.object(setup, "run", run), \
                mock.patch.object(setup, "download") as download, \
                mock.patch("builtins.input", side_effect=AssertionError("unexpected prompt")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(setup.install_build_tools([86], True), nvcc)
        self.assertIn("/ubuntu2404/x86_64/cuda-keyring", download.call_args[0][0])
        self.assertEqual(commands, [["sudo", "apt-get", "install", "-y", "build-essential"],
                                    ["sudo", "dpkg", "-i", str(download.call_args[0][1])],
                                    ["sudo", "apt-get", "update"],
                                    ["sudo", "apt-get", "install", "-y", "cuda-toolkit-13-0"]])

    def test_another_distribution_is_told_what_to_install(self):
        which, _ = self.which(["apt-get", "g++"])
        out = io.StringIO()
        with mock.patch.object(setup, "find_nvcc", return_value=(None, None)), \
                mock.patch.object(setup.shutil, "which", which), \
                mock.patch.object(setup, "os_release", return_value={"ID": "debian", "VERSION_ID": "12"}), \
                mock.patch.object(setup, "run", side_effect=AssertionError("installer started")), \
                contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit):
                setup.install_build_tools([86], True)
        self.assertIn("Ubuntu 22.04 / 24.04 only", out.getvalue())

    def test_an_installed_toolkit_is_used(self):
        which, _ = self.which(["g++"])
        with mock.patch.object(setup, "find_nvcc", return_value=("/usr/local/cuda/bin/nvcc", (12, 8))), \
                mock.patch.object(setup.shutil, "which", which), \
                mock.patch.object(setup, "run", side_effect=AssertionError("installer started")), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(setup.install_build_tools([86], False), "/usr/local/cuda/bin/nvcc")


if __name__ == "__main__":
    unittest.main()
