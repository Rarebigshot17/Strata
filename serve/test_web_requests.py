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
    PAGE = {"thinking": "max", "temperature": 1.0, "top_p": 0.95, "top_k": 0, "max": "", "seed": ""}

    def setUp(self):
        self.js = APP.read_text(encoding="utf-8")

    def run_js(self, script: str):
        r = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr)
        return json.loads(r.stdout)

    def fields(self, name: str, settings: dict) -> dict:
        script = function_source(self.js, name) + f"\nconsole.log(JSON.stringify({name}({json.dumps(settings)})));"
        if name != "samplingFields":                    # the shared fields are built from the chat's
            script = function_source(self.js, "samplingFields") + "\n" + script
        return self.run_js(script)

    def test_the_chat_sends_each_thinking_level(self):
        # Off / Low / High / Max: the level is always sent, so a run config or shared setting with another
        # reasoning_effort does not win over the page
        for level in ("none", "low", "high", "max"):
            with self.subTest(level=level):
                self.assertEqual(self.fields("samplingFields", {**self.PAGE, "thinking": level}),
                                 {"temperature": 1.0, "reasoning_effort": level, "top_p": 0.95, "top_k": 0})
        self.assertEqual(self.fields("samplingFields",
                                     {**self.PAGE, "thinking": "none", "temperature": 0, "seed": "7", "max": "64"}),
                         {"temperature": 0, "reasoning_effort": "none", "seed": 7, "max_tokens": 64})
        send = function_source(self.js, "send")
        self.assertIn("...samplingFields(settings)", send)
        self.assertNotIn("reasoning_effort", send)          # only through samplingFields

    def test_other_apps_get_thinking_off_but_no_level(self):
        # "Use for other apps too" shares the page's fields, but thinking only when it is off: a shared level would
        # make an Anthropic client's small calls think too, which anthropic_thinking "on_request" keeps them from (#278)
        name = re.search(r"defaults: on \? (\w+)\(s\)", self.js).group(1)
        for level in ("low", "high", "max"):
            with self.subTest(level=level):
                self.assertEqual(self.fields(name, {**self.PAGE, "thinking": level}),
                                 {"temperature": 1.0, "top_p": 0.95, "top_k": 0})
        self.assertEqual(self.fields(name, {**self.PAGE, "thinking": "none"}),
                         {"temperature": 1.0, "reasoning_effort": "none", "top_p": 0.95, "top_k": 0})

    def test_an_earlier_pages_setting_reads_as_a_level(self):
        # the page's thinking was Off / On, and before that none / low / medium / high: "on" is Max (the model's
        # default), medium is High; anything else is the default, Max
        got = self.run_js(function_source(self.js, "thinkingLevel") + "\nconsole.log(JSON.stringify(" + json.dumps(
            ["none", "on", "low", "medium", "high", "max", None, "", "extreme", 3]) + ".map(thinkingLevel)));")
        self.assertEqual(got, ["none", "max", "low", "high", "high", "max", "max", "max", "max", "max"])
        self.assertIn('thinking: "max"', self.js[self.js.index("const DEFAULTS"):].split("\n", 1)[0])

    def test_the_control_has_the_four_levels(self):
        html = (APP.parent / "index.html").read_text(encoding="utf-8")
        seg = html[html.index('id="s-thinking"'):]
        seg = seg[:seg.index("</div>")]
        self.assertEqual(re.findall(r'data-v="(\w+)">(\w+)<', seg),
                         [("none", "Off"), ("low", "Low"), ("high", "High"), ("max", "Max")])
        # the line under the control says what each does
        notes = self.run_js(re.search(r"^const THINKING_NOTES = \{.*?\};", self.js, re.S | re.M).group(0)
                            + "\nconsole.log(JSON.stringify(THINKING_NOTES));")
        self.assertEqual(sorted(notes), ["high", "low", "max", "none"])
        self.assertIn("default", notes["max"])


if __name__ == "__main__":
    unittest.main()
