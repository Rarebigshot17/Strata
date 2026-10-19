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
    PAGE = {"thinking": "on", "temperature": 1.0, "top_p": 0.95, "top_k": 0, "max": "", "seed": ""}

    def setUp(self):
        self.js = APP.read_text(encoding="utf-8")

    def fields(self, name: str, settings: dict) -> dict:
        script = function_source(self.js, name) + f"\nconsole.log(JSON.stringify({name}({json.dumps(settings)})));"
        if name != "samplingFields":                    # the shared fields are built from the chat's
            script = function_source(self.js, "samplingFields") + "\n" + script
        r = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def test_the_chat_sends_thinking_on_and_off(self):
        # "On" sends a level, so a run config or shared setting with reasoning_effort "none" does not win over the page
        self.assertEqual(self.fields("samplingFields", self.PAGE),
                         {"temperature": 1.0, "reasoning_effort": "high", "top_p": 0.95, "top_k": 0})
        self.assertEqual(self.fields("samplingFields",
                                     {**self.PAGE, "thinking": "none", "temperature": 0, "seed": "7", "max": "64"}),
                         {"temperature": 0, "reasoning_effort": "none", "seed": 7, "max_tokens": 64})
        send = function_source(self.js, "send")
        self.assertIn("...samplingFields(settings)", send)
        self.assertNotIn("reasoning_effort", send)          # only through samplingFields

    def test_other_apps_get_thinking_off_but_not_on(self):
        # "Use for other apps too" shares the page's fields, but thinking only when it is off: a shared "on" would make
        # an Anthropic client's small calls think too, which anthropic_thinking "on_request" keeps them from (#278)
        name = re.search(r"defaults: on \? (\w+)\(s\)", self.js).group(1)
        self.assertEqual(self.fields(name, self.PAGE), {"temperature": 1.0, "top_p": 0.95, "top_k": 0})
        self.assertEqual(self.fields(name, {**self.PAGE, "thinking": "none"}),
                         {"temperature": 1.0, "reasoning_effort": "none", "top_p": 0.95, "top_k": 0})


if __name__ == "__main__":
    unittest.main()
