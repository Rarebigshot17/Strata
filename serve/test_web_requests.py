"""serve/test_web_requests.py - what the chat page (serve/web/app.js) sends: its settings' request fields, run in node.

    python3 -m unittest serve.test_web_requests -v
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import unittest
from pathlib import Path

APP = Path(__file__).resolve().parent / "web" / "app.js"
NODE = shutil.which("node") or shutil.which("nodejs")


def function_source(js: str, name: str) -> str:
    """The text of a top-level `[async] function NAME(...) {...}` of app.js (it ends at the first "}" in column 0)."""
    m = re.search(r"^(?:async )?function " + re.escape(name) + r"\(.*?^\}", js, re.S | re.M)
    if m is None:
        raise AssertionError(f"app.js has no function {name}")
    return m.group(0)


@unittest.skipUnless(NODE, "needs node")
class ChatRequestFields(unittest.TestCase):
    def setUp(self):
        self.js = APP.read_text(encoding="utf-8")
        # the function whose fields "Use for other apps too" shares
        self.name = re.search(r"defaults: on \? (\w+)\(s\)", self.js).group(1)

    def fields(self, settings: dict) -> dict:
        script = function_source(self.js, self.name) + f"\nconsole.log(JSON.stringify({self.name}({json.dumps(settings)})));"
        r = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def test_thinking_on_and_off_are_both_sent(self):
        # "On" sends a level, so a run config or shared setting with reasoning_effort "none" does not win over the page
        page = {"thinking": "on", "temperature": 1.0, "top_p": 0.95, "top_k": 0, "max": "", "seed": ""}
        self.assertEqual(self.fields(page), {"temperature": 1.0, "reasoning_effort": "high", "top_p": 0.95, "top_k": 0})
        self.assertEqual(self.fields({**page, "thinking": "none", "temperature": 0, "seed": "7", "max": "64"}),
                         {"temperature": 0, "reasoning_effort": "none", "seed": 7, "max_tokens": 64})

    def test_the_chat_sends_the_same_fields(self):
        send = function_source(self.js, "send")
        self.assertIn(f"...{self.name}(settings)", send)
        self.assertNotIn("reasoning_effort", send)          # only through the shared function


if __name__ == "__main__":
    unittest.main()
