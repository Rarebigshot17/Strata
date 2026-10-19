"""Tests for how ./setup.sh starts the server (setup.start): setup replaces itself with the server, so Ctrl+C, closing
the terminal and `kill <pid>` reach the server's own stop (STOP and QUIT to the engine), and nothing ends the server
while it stops.  A fake server stands in for serve/server.py; no GPU, no model.

    python3 tools/test_setup_stop.py
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# setup.start with this config, the fake server as the server's command and no GPUs (no nvidia-smi)
DRIVER = r'''
import sys
sys.path.insert(0, sys.argv[1])
import setup
setup.gpus = lambda: []
setup.server_command = lambda cfg_path, port, browser: [sys.executable, sys.argv[3]]
sys.exit(setup.start(setup.Path(sys.argv[2]), None))
'''

# The server's stop takes a while (QUIT to the engine, which saves its expert profile and gives its memory back)
SERVER = r'''
import os, sys, time
d = os.environ["FAKE_DIR"]
print("fake server: running", flush=True)
with open(os.path.join(d, "pid.tmp"), "w") as f:
    f.write(str(os.getpid()))
os.replace(os.path.join(d, "pid.tmp"), os.path.join(d, "pid"))
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    time.sleep(1.0)
    open(os.path.join(d, "stopped"), "w").close()
'''


class StartStops(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        (self.dir / "strata-glm").write_bytes(b"")
        (self.dir / "m-00001-of-00001.gguf").write_bytes(b"")
        self.cfg = self.dir / "strata-glm-ud-iq3_xxs.json"
        self.cfg.write_text(json.dumps({"exe": str(self.dir / "strata-glm"),
                                        "args": ["-m", str(self.dir / "m-00001-of-00001.gguf"), "--ctx", "4096"],
                                        "open_browser": False}), encoding="utf-8")
        self.server = self.dir / "server.py"
        self.server.write_text(SERVER, encoding="utf-8")

    def start(self):
        # a process group of its own, as a terminal's job: the test signals it as the terminal would
        proc = subprocess.Popen([sys.executable, "-c", DRIVER, str(ROOT), str(self.cfg), str(self.server)],
                                env={**os.environ, "FAKE_DIR": str(self.dir)}, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, start_new_session=True)
        self.addCleanup(lambda: proc.poll() is None and os.killpg(proc.pid, signal.SIGKILL))
        deadline = time.monotonic() + 60
        while not (self.dir / "pid").exists() and proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertTrue((self.dir / "pid").exists(), "the fake server did not start")
        return proc

    def test_ctrl_c_lets_the_server_finish_its_stop(self):
        proc = self.start()
        os.killpg(proc.pid, signal.SIGINT)              # Ctrl+C: the terminal signals the whole foreground group
        out, _ = proc.communicate(timeout=60)
        # setup no longer ends the server a quarter of a second after Ctrl+C
        self.assertTrue((self.dir / "stopped").exists(), out)

    def test_setup_becomes_the_server(self):
        proc = self.start()
        # the process ./setup.sh started is the server itself: `kill <pid>` reaches the server's own stop
        self.assertEqual(int((self.dir / "pid").read_text()), proc.pid)
        os.killpg(proc.pid, signal.SIGINT)
        out, _ = proc.communicate(timeout=60)
        # setup's own lines come before the server's, none lost when it replaces itself (stdout is a pipe here)
        self.assertIn("Starting", out)
        self.assertLess(out.index("Starting"), out.index("fake server: running"))


if __name__ == "__main__":
    unittest.main()
