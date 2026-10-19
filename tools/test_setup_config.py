"""Tests for what setup keeps in a model's run config when it runs again (#629), and the browser choice (#609).  Every
outside effect is mocked (tools/test_setup_glm.py's install()).

    python -m unittest tools.test_setup_config
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402
from test_setup_glm import install, owner_gpus  # noqa: E402

CONFIG = "strata-glm-ud-iq3_xxs.json"


def quiet(fn, *args, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        return fn(*args, **kw), out.getvalue()


USER = {"sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 40, "min_p": 0.0},
        "mcp_servers": {"files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]}},
        "mcp": {"timeout_s": 60, "max_result_chars": 16000, "max_rounds": 8},
        "cors_origins": ["http://localhost:3000"], "reasoning_effort": "low"}


def run_setup(*extra, configs=()):
    """setup --yes --no-start for the 3-bit model: (exit code, output, its config)."""
    code, out, cfgs, _, _ = install(["--yes", "--no-start", *extra], configs=configs)
    return code, out, cfgs.get(CONFIG)


class CarryOver(unittest.TestCase):
    """#629: setup run again (a different --context) keeps the keys and engine options it does not write, and keeps
    the old file as .bak."""

    def test_user_keys_and_options_survive_a_rerun(self):
        code, out, first = run_setup("--context", "32768")
        self.assertEqual(code, 0, out)
        old = {**first, **USER, "args": first["args"] + ["--vram-reserve", "2", "--no-warm"]}
        code, out, cfg = run_setup("--context", "131072", configs=[(CONFIG, old)])
        self.assertEqual(code, 0, out)
        for k, v in USER.items():
            self.assertEqual(cfg[k], v, k)
        a = cfg["args"]
        self.assertEqual(a[a.index("--ctx") + 1], "131072")                # the new choice is used
        self.assertEqual(a[a.index("--vram-reserve") + 1], "2")            # an option added by hand (or by the
        self.assertIn("--no-warm", a)                                      # Settings page) is kept
        self.assertEqual(a.count("--ctx"), 1)
        self.assertIn(f"kept from your earlier {CONFIG}: sampling, mcp_servers, mcp, cors_origins, reasoning_effort, "
                      "args --vram-reserve, args --no-warm", out)
        self.assertIn(f"kept as {CONFIG}.bak", out)

    def test_a_first_install_is_unchanged(self):
        code, out, cfg = run_setup()
        self.assertEqual(code, 0, out)
        self.assertNotIn("kept from", out)
        self.assertNotIn(".bak", out)
        self.assertFalse(set(cfg) - setup.SETUP_KEYS)                     # setup writes only its own keys

    def test_setup_keys_are_rewritten(self):
        code, out, first = run_setup()
        old = {**first, "model_name": "renamed", "port": 9999, "host": "0.0.0.0"}
        code, out, cfg = run_setup(configs=[(CONFIG, old)])
        self.assertEqual(code, 0, out)
        self.assertEqual(cfg["model_name"], "glm-5.3-flash")
        self.assertEqual(cfg["port"], 8101)
        self.assertNotIn("host", cfg)                                     # as before: --host is given to setup
        self.assertNotIn("kept from", out)

    def test_write_setup_config(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / CONFIG
            old = {"exe": "e", "args": ["-m", "a.gguf", "--ctx", "8192", "--threads", "8"],
                   "sampling": {"temperature": 0.7}, "env": {"MY_VAR": "1"}}
            p.write_text(json.dumps(old), encoding="utf-8")
            new = {"exe": "e2", "args": ["-m", "b.gguf", "--ctx", "131072"]}
            _, out = quiet(setup.write_setup_config, p, new)
            got = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(got["sampling"], {"temperature": 0.7})
            self.assertEqual(got["env"], {"MY_VAR": "1"})
            self.assertEqual(got["exe"], "e2")
            self.assertEqual(got["args"], ["-m", "b.gguf", "--ctx", "131072", "--threads", "8"])
            self.assertEqual(json.loads((Path(d) / f"{CONFIG}.bak").read_text(encoding="utf-8")), old)
            self.assertIn("args --threads", out)
            # the same again: nothing changes, so no new copy is made
            (Path(d) / f"{CONFIG}.bak").unlink()
            quiet(setup.write_setup_config, p, {"exe": "e2", "args": ["-m", "b.gguf", "--ctx", "131072"]})
            self.assertFalse((Path(d) / f"{CONFIG}.bak").exists())
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")), got)

    def test_an_unreadable_earlier_config_is_replaced_with_a_copy(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / CONFIG
            p.write_text("{cut off", encoding="utf-8")
            quiet(setup.write_setup_config, p, {"exe": "e", "args": []})
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")), {"exe": "e", "args": []})
            self.assertEqual((Path(d) / f"{CONFIG}.bak").read_text(encoding="utf-8"), "{cut off")

    def test_typed_ahead_flush(self):
        # #841: typed-ahead keys are dropped on a terminal only
        with unittest.mock.patch.object(setup.sys, "stdin") as stdin:
            stdin.isatty.return_value = False
            setup.flush_typed_ahead()                      # a pipe: nothing touched
            stdin.fileno.assert_not_called()

    def test_an_expert_profile_of_the_users_own_is_carried(self):
        # #775: an --expert-profile set by hand survives a setup run while its file exists
        with tempfile.TemporaryDirectory() as d:
            mine = Path(d) / "learned.profile"
            mine.write_bytes(b"x")
            old = {"args": ["-m", "a.gguf", "--expert-profile", str(mine)]}
            new = {"args": ["-m", "a.gguf", "--expert-profile", "/data/glm-ud-iq3_xxs.profile"]}
            self.assertEqual(setup.carry_over(old, new), ["args --expert-profile"])
            self.assertEqual(setup.flag_value(new["args"], "--expert-profile"), str(mine))
            # a file that is gone, or setup's own named again: setup's own stays
            gone = {"args": ["--expert-profile", str(Path(d) / "gone.profile")]}
            new = {"args": ["--expert-profile", "/data/glm-ud-iq3_xxs.profile"]}
            self.assertEqual(setup.carry_over(gone, new), [])
            self.assertEqual(new["args"], ["--expert-profile", "/data/glm-ud-iq3_xxs.profile"])
            self.assertEqual(setup.carry_over({"args": list(new["args"])}, new), [])

    def test_a_copy_set_up_like_the_last_one_keeps_them_too(self):
        with tempfile.TemporaryDirectory() as d:
            src = Path(d) / "other" / CONFIG
            src.parent.mkdir()
            src.write_text(json.dumps({"exe": "old", "mcp_servers": {"a": {"command": "x"}}}), encoding="utf-8")
            p = Path(d) / CONFIG
            quiet(setup.write_setup_config, p, {"exe": "new"}, src)
            self.assertEqual(json.loads(p.read_text(encoding="utf-8")),
                             {"exe": "new", "mcp_servers": {"a": {"command": "x"}}})
            self.assertFalse((Path(d) / f"{CONFIG}.bak").exists())   # nothing here was replaced


class NoBrowser(unittest.TestCase):
    """#609 (#631): --no-browser keeps the chat page from opening: "open_browser": false in the run config, no --open
    on the start's server command or in run-<model>.sh.  Without the flag nothing changes."""

    def test_at_setup(self):
        code, out, cfg = run_setup("--no-browser")
        self.assertEqual(code, 0, out)
        self.assertIs(cfg["open_browser"], False)
        code, out, cfg = run_setup()
        self.assertNotIn("open_browser", cfg)                             # not given: no key
        # setup run again without the flag keeps the choice (#629); --browser undoes it
        code, out, cfg = run_setup(configs=[(CONFIG, {"open_browser": False})])
        self.assertIs(cfg["open_browser"], False)
        code, out, cfg = run_setup("--browser", configs=[(CONFIG, {"open_browser": False})])
        self.assertIs(cfg["open_browser"], True)

    def test_run_script(self):
        with tempfile.TemporaryDirectory() as d, unittest.mock.patch.object(setup, "ROOT", Path(d)):
            text = lambda p: p.read_text(encoding="utf-8")   # noqa: E731
            self.assertIn(" --open", text(setup.write_run_script("glm-ud-iq3_xxs", Path(d) / CONFIG, 8101)))
            script = setup.write_run_script("glm-ud-iq3_xxs", Path(d) / CONFIG, 8101, False)
            self.assertNotIn("--open", text(script))
            self.assertIn(" --port 8101", text(script))

    def start_cmd(self, cfg, keep=None):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / "strata-glm"
            exe.write_bytes(b"")
            p = Path(d) / CONFIG
            p.write_text(json.dumps({"exe": str(exe), "args": ["--ctx", "131072"], **cfg}))
            call = unittest.mock.Mock(return_value=0)
            with unittest.mock.patch.object(setup, "gpus", owner_gpus), \
                    unittest.mock.patch.object(setup, "exec_server", call):
                _, out = quiet(setup.start, p, None, None, keep)
            return call.call_args[0][0], out, json.loads(p.read_text())

    def test_on_a_start(self):
        cmd, out, _ = self.start_cmd({})
        self.assertIn("--open", cmd)                                      # the default
        self.assertIn("the browser opens when it is ready", out)
        cmd, out, _ = self.start_cmd({"open_browser": False})
        self.assertNotIn("--open", cmd)
        self.assertNotIn("the browser opens", out)
        self.assertIn("no browser", out)                                  # the Settings line says so
        cmd, out, saved = self.start_cmd({}, {"open_browser": False})     # ./setup.sh --no-browser: kept from now on
        self.assertNotIn("--open", cmd)
        self.assertIs(saved["open_browser"], False)
        self.assertIn("saved for this model: no browser", out)
        cmd, out, saved = self.start_cmd({"open_browser": False}, {"open_browser": True})
        self.assertIn("--open", cmd)
        self.assertIs(saved["open_browser"], True)


if __name__ == "__main__":
    unittest.main()
