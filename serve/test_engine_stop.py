"""How the server ends its engine on an ordinary stop: Ctrl+C (SIGINT to the terminal's foreground process group),
closing the terminal (SIGHUP, after which the window takes no more output), SIGTERM (kill, docker stop, systemd), and a
stop while the engine still starts.  Each must reach the engine as STOP and QUIT - strata-glm saves its expert profile
then - or, during the start, end it.

The engine is a fake process (FAKE_ENGINE) with no signal handling of its own, as strata-glm had none: a signal that
reaches it ends it at once, and its record then has no QUIT.  The server runs as `python -m serve.server` in a process
group of its own, which the test signals as a terminal would.

    python3 -m unittest serve.test_engine_stop -v
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

from serve import server as S
from serve.test_glm_engine_protocol import free_port, write_gguf

ROOT = Path(__file__).resolve().parents[1]

FAKE_ENGINE = r'''
import os, sys, time
rec = open(os.environ["FAKE_RECORD"], "a", encoding="utf-8")

def note(x):
    rec.write(x + "\n")
    rec.flush()

sys.stdout.reconfigure(line_buffering=True)
note("PID %d" % os.getpid())
time.sleep(float(os.environ.get("FAKE_START_S", "0")))     # placing the layers, the warm-up
print("INFO engine=0.2.0 backend=fake eos=3 park_mib=0 logit_bias=1")
print("READY 4096 stop")
for line in sys.stdin:
    note(line.strip())
    if line.strip() == "QUIT":
        break
else:
    note("EOF")
time.sleep(float(os.environ.get("FAKE_QUIT_S", "0")))      # saving the profile, giving the memory back
note("BYE")
'''


def gone(pid: int) -> bool:
    """The process has ended (a zombie that its parent has not reaped yet counts as ended)."""
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as f:
            return f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return True


def wait_until(cond, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.05)
    return cond()


@unittest.skipUnless(sys.platform.startswith("linux"), "process groups and /proc: Linux")
class EngineSession(unittest.TestCase):
    def test_the_engine_runs_in_a_session_of_its_own(self):
        # Ctrl+C and the hang-up of a closed terminal go to the terminal's process group and session: the engine is in
        # neither, so only the server gets them, and it ends the engine with QUIT
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "strata-glm"
            script.write_text(f"#!{sys.executable}\n" + FAKE_ENGINE, encoding="utf-8")
            script.chmod(0o755)
            env = {**os.environ, "FAKE_RECORD": str(Path(d) / "record.txt")}
            engine = S.StrataEngine(str(script), ["--ctx", "4096"], cwd=d, env=env)
            try:
                pid = engine.proc.pid
                self.assertNotEqual(os.getpgid(pid), os.getpgrp())
                self.assertNotEqual(os.getsid(pid), os.getsid(0))
            finally:
                engine.shutdown()
            self.assertIn("QUIT", (Path(d) / "record.txt").read_text(encoding="utf-8").splitlines())
            with self.assertRaises(S.EngineDied):           # stopping: a request's restart starts nothing
                engine.restart()


class QuitWait(unittest.TestCase):
    """After QUIT the engine saves its expert profile and gives back its memory (VRAM on every card, the parked
    conversations, the mapped experts): a longer wait than the 20 s it had, still bounded, and the window says what
    it waits for."""

    def test_a_longer_bounded_wait_then_a_kill(self):
        engine = S.StrataEngine("missing-executable", [], lazy=True)
        proc = mock.Mock()
        proc.stdin, proc.stdout = io.StringIO(), io.StringIO()
        proc.poll.side_effect = lambda: -9 if proc.kill.called else None
        waits = []

        def wait(timeout=None):
            waits.append((timeout, proc.kill.called))
            if proc.kill.called:
                return -9
            raise subprocess.TimeoutExpired("engine", timeout)
        proc.wait.side_effect = wait
        engine.proc = proc
        out = io.StringIO()
        with contextlib.redirect_stdout(out), mock.patch.object(S, "ENGINE_QUIT_S", 120.0):   # the default
            engine.close()
        before_kill = sum(t for t, killed in waits if not killed)
        self.assertEqual(before_kill, 120)
        proc.terminate.assert_not_called()               # strata-glm ignores SIGTERM while the server runs
        proc.kill.assert_called_once()
        self.assertIsNone(engine.proc)
        self.assertIn("saves its expert profile", out.getvalue())
        self.assertIn("killed", out.getvalue())

    def test_main_s_stop_then_atexit_end_the_engine_once(self):
        # main()'s stop runs shutdown(), and atexit runs it again however main() ends.  When a second Ctrl+C cut the
        # first short (main() then killed the engine, which can take seconds to give its VRAM back) or it ended in
        # EngineStuck, the engine was killed: atexit must not send QUIT and wait for it again, nor say after
        # "[strata] stopped" that it saves its expert profile
        for first in (KeyboardInterrupt, S.EngineStuck):
            with self.subTest(first=first.__name__):
                engine = S.StrataEngine("missing-executable", [], lazy=True)
                proc = mock.Mock()
                proc.stdin, proc.stdout = io.StringIO(), io.StringIO()
                proc.poll.return_value = None            # killed, and still ending
                waits = []

                def wait(timeout=None):
                    waits.append(timeout)
                    if first is KeyboardInterrupt and not proc.kill.called:
                        raise KeyboardInterrupt          # the second Ctrl+C, during the wait for QUIT
                    raise subprocess.TimeoutExpired("engine", timeout)
                proc.wait.side_effect = wait
                engine.proc = proc
                with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(S, "ENGINE_QUIT_S", 120.0), \
                        self.assertRaises(first):
                    engine.shutdown()                    # main()'s stop
                proc.kill()                              # (main()'s answer to the second Ctrl+C)
                waits.clear()
                out = io.StringIO()
                with contextlib.redirect_stdout(out), mock.patch.object(S, "ENGINE_QUIT_S", 120.0):
                    engine.shutdown()                    # atexit
                self.assertEqual(waits, [])
                self.assertNotIn("saves its expert profile", out.getvalue())

    @unittest.skipUnless(sys.platform.startswith("linux"), "a fake engine process")
    def test_an_engine_slow_to_end_is_waited_for_and_the_window_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            script = Path(d) / "strata-glm"
            script.write_text(f"#!{sys.executable}\n" + FAKE_ENGINE, encoding="utf-8")
            script.chmod(0o755)
            env = {**os.environ, "FAKE_RECORD": str(Path(d) / "record.txt"), "FAKE_QUIT_S": "3"}
            engine = S.StrataEngine(str(script), ["--ctx", "4096"], cwd=d, env=env)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                engine.shutdown()
            self.assertIn("BYE", (Path(d) / "record.txt").read_text(encoding="utf-8").splitlines())
            self.assertIn("waiting for the engine to end", out.getvalue())
            self.assertIn("the engine ended", out.getvalue())


@unittest.skipUnless(sys.platform.startswith("linux"), "process groups and /proc: Linux")
class ServerStops(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        write_gguf(self.dir / "glm-00001-of-00001.gguf", name="GLM Stop Test")
        self.script = self.dir / "strata-glm"
        self.script.write_text(f"#!{sys.executable}\n" + FAKE_ENGINE, encoding="utf-8")
        self.script.chmod(0o755)
        self.record = self.dir / "record.txt"

    def start(self, start_s: float = 0.0, quit_s: float = 0.0, extra: tuple = ()):
        cfg = {"exe": str(self.script), "args": ["-m", "glm-00001-of-00001.gguf", "--ctx", "4096"],
               "cwd": str(self.dir), "log": str(self.dir / "engine.log"), "open_browser": False,
               "env": {"FAKE_RECORD": str(self.record), "FAKE_START_S": str(start_s), "FAKE_QUIT_S": str(quit_s)}}
        path = self.dir / "strata-test.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        self.port = free_port()
        # a process group of its own, as a terminal's foreground job: the test signals it as the terminal would
        proc = subprocess.Popen([sys.executable, "-m", "serve.server", "--engine", "strata", "--config", str(path),
                                 "--port", str(self.port), *extra], cwd=ROOT, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
        self.addCleanup(self.cleanup, proc)
        return proc

    def cleanup(self, proc):
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(10)
        if proc.stdout is not None:
            proc.stdout.close()
        pid = self.engine_pid()
        if pid is not None and not gone(pid):
            os.kill(pid, signal.SIGKILL)

    def lines(self) -> list[str]:
        try:
            return self.record.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []

    def engine_pid(self) -> int | None:
        pids = [int(x.split()[1]) for x in self.lines() if x.startswith("PID ")]
        return pids[-1] if pids else None

    def ready(self, proc) -> None:
        def answers():
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2):
                    return True
            except OSError:
                return False
        self.assertTrue(wait_until(lambda: answers() or proc.poll() is not None, 60) and proc.poll() is None,
                        "the server did not come up")

    def stops_with_quit(self, sig: int, close_window: bool = False) -> None:
        proc = self.start(quit_s=1.0)                    # the engine's QUIT takes a while
        self.ready(proc)
        if close_window:                                 # the window is gone: every write to it fails from now on
            proc.stdout.close()
            proc.stdout = None
        os.killpg(proc.pid, sig)
        if close_window:                                 # the kernel and the shell both hang up: no hurry from that
            time.sleep(0.3)
            os.killpg(proc.pid, sig)
        self.assertEqual(proc.wait(60), 0)
        out = proc.stdout.read() if proc.stdout is not None else ""
        self.assertIn("QUIT", self.lines(), out)         # it got QUIT (strata-glm saves its profile)
        self.assertIn("BYE", self.lines(), out)          # and the server waited for it to end
        if proc.stdout is not None:
            self.assertIn("[strata] stopped", out)

    def test_ctrl_c(self):
        self.stops_with_quit(signal.SIGINT)

    def test_sigterm(self):
        self.stops_with_quit(signal.SIGTERM)

    def test_closing_the_terminal(self):
        self.stops_with_quit(signal.SIGHUP, close_window=True)

    def test_ctrl_c_while_the_engine_starts_ends_it(self):
        proc = self.start(start_s=60.0)                  # a long warm-up
        self.assertTrue(wait_until(lambda: self.engine_pid() is not None, 60), "the engine did not start")
        os.killpg(proc.pid, signal.SIGINT)
        self.assertEqual(proc.wait(60), 0)
        self.assertIn("stopped before the model was ready", proc.stdout.read())
        self.assertTrue(wait_until(lambda: gone(self.engine_pid()), 10), "the starting engine was left running")

    def test_any_end_of_the_server_quits_the_engine(self):
        # an empty --api-key ends the server after the engine has started (exit code 2): the engine gets QUIT
        proc = self.start(extra=("--api-key", " "))
        self.assertEqual(proc.wait(60), 2)
        self.assertIn("QUIT", self.lines())


if __name__ == "__main__":
    unittest.main()
