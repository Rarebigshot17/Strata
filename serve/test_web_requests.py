"""serve/test_web_requests.py - what the chat page (serve/web/app.js) sends: its settings' request fields, and the
conversation it sends back (its send() run in node against the server, with a scripted engine).

    python3 -m unittest serve.test_web_requests -v
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server as S  # noqa: E402
from serve.glm_fixtures import GlmTokenizer, call_text, glm_template, reply_text  # noqa: E402
from serve.mcp import McpHub  # noqa: E402

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


class RecordingEngine(S.MockEngine):
    """MockEngine that keeps each pass's prompt and the tokens it gave (the reply as the model wrote it)."""

    def __init__(self, tok, scripts):
        super().__init__(tok, scripts)
        self.passes = []

    def generate(self, ids, max_new, sampling, cancel, checkpoint=None, end_ids=None, drawn=0):
        given = []
        self.passes.append((list(ids), given))
        for t in super().generate(ids, max_new, sampling, cancel, checkpoint, end_ids, drawn):
            given.append(t)
            yield t


# What send() reaches beyond the chat's own data, for node: the page's elements and drawing are stubs, fetch goes to
# the test's server.  QUESTIONS are asked one after another, each after the last answer ended.
CHAT_HARNESS = """
const BASE = %(base)s, QUESTIONS = %(questions)s, errors = [];
const nodeFetch = globalThis.fetch;
globalThis.fetch = (url, init) => nodeFetch(`${BASE}/${url}`, init);
const nodes = {};
const $ = (id) => nodes[id] || (nodes[id] = {value: "", hidden: false, disabled: false, textContent: "", style: {},
                                              lastElementChild: {}});
let messages = [], attachments = [], busy = null;
const health = {model: "m"}, mcpInfo = {tools: %(mcp_tools)d};
const settings = {thinking: "max", temperature: 0, top_p: 0.95, top_k: 0, max: "", seed: "", show: true, mcp: true};
const headers = (json) => (json ? {"Content-Type": "application/json", Origin: BASE} : {Origin: BASE});
const fmt = (n) => String(n), noop = () => {};
const renderAttachments = noop, renderChat = noop, autosize = noop, setBusy = noop, updateAssistant = noop;
const saveChat = noop, scrollDown = noop, requestAnimationFrame = () => 0, cancelAnimationFrame = noop;
const toast = (kind, title, text) => errors.push(`${title}: ${text}`);
%(functions)s
(async () => {
  for (const q of QUESTIONS) { $("input").value = q; await send(); }
  console.log(JSON.stringify({messages, errors}));
})();
"""
CHAT_FUNCTIONS = ("send", "apiMessages", "assistantMessages", "onTool", "userText", "fileBlock", "samplingFields")


@unittest.skipUnless(NODE, "needs node")
class ChatHistory(unittest.TestCase):
    """The page sends each answer back with its thinking, as the server streamed it: GLM-5.3-Flash's template keeps
    earlier turns' thinking, so the next turn's prompt starts with the last prompt and the reply as the model wrote it,
    and the engine goes on from the end of the reply instead of reading it again from the checkpoint."""

    def setUp(self):
        self.js = APP.read_text(encoding="utf-8")
        self.tok = GlmTokenizer()

    def chat(self, scripts, questions, hub=None) -> dict:
        """Ask `questions` through app.js's send(); the engine writes `scripts`, one per pass."""
        self.engine = RecordingEngine(self.tok, scripts)
        svc = S.Service(self.engine, self.tok, glm_template())
        svc.mcp = hub
        httpd = S.serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        script = CHAT_HARNESS % {"base": json.dumps(f"http://127.0.0.1:{httpd.server_address[1]}"),
                                 "questions": json.dumps(questions), "mcp_tools": len(hub.routes()) if hub else 0,
                                 "functions": "\n".join([re.search(r"^const ROUND_GAP = .*$", self.js, re.M).group(0)]
                                                        + [function_source(self.js, f) for f in CHAT_FUNCTIONS])}
        r = subprocess.run([NODE, "-e", script], capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        self.assertEqual(out["errors"], [])
        return out

    def assertGoesOn(self, before: int, after: int):
        """Pass `after`'s prompt starts with pass `before`'s prompt and the tokens that pass gave."""
        (p0, given), (p1, _) = self.engine.passes[before], self.engine.passes[after]
        self.assertTrue(given)
        self.assertEqual(self.tok.decode(p1[:len(p0) + len(given)]), self.tok.decode(p0 + given))
        self.assertEqual(p1[:len(p0) + len(given)], p0 + given)

    def test_the_next_turn_goes_on_from_the_reply(self):
        out = self.chat([reply_text("The user greets me.\n", "Hello!"), reply_text("More.", "Fine.")],
                        ["Hi", "How are you?"])
        self.assertEqual([(m["role"], m["text"]) for m in out["messages"]],
                         [("user", "Hi"), ("assistant", "Hello!"), ("user", "How are you?"), ("assistant", "Fine.")])
        self.assertEqual(len(self.engine.passes), 2)
        self.assertGoesOn(0, 1)
        self.assertIn("<think>The user greets me.\n</think>Hello!<|user|>How are you?",
                      self.tok.decode(self.engine.passes[1][0]))

    def test_after_mcp_tool_rounds_too(self):
        # each round's message carries that round's thinking, as the server wrote it into the next round's prompt
        # (run_with_mcp), and the answer after the tools carries the last round's as the model wrote it
        hub = McpHub({"fake": {"command": sys.executable, "args": [str(Path(__file__).resolve().parent /
                                                                       "mcp_fake_server.py")]}},
                     {"timeout_s": 10, "max_result_chars": 500, "max_rounds": 3})
        hub.start(wait=True)
        self.addCleanup(hub.close)
        out = self.chat([reply_text("I need the tool.", "Let me check.", call_text("fake__echo", text="one")),
                         reply_text("One more.", "", call_text("fake__add", a=2, b=3)),
                         reply_text("Both answered.\n\n", "They said one and 5."),
                         reply_text("Easy.", "Bye.")], ["check it", "thanks"], hub)
        self.assertEqual([t["state"] for t in out["messages"][1]["tools"]], ["done", "done"])
        self.assertEqual(len(self.engine.passes), 4)
        self.assertGoesOn(2, 3)
        last = self.tok.decode(self.engine.passes[3][0])
        self.assertIn("<think>I need the tool.</think>Let me check.<tool_call>fake__echo", last)
        self.assertIn("<think>One more.</think><tool_call>fake__add", last)


if __name__ == "__main__":
    unittest.main()
