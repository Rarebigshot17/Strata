"""Tests for serve/glm_server.py that need no engine and no model: GLM's reply parser, the chat template's rendering
of tools and tool results, and the stop strings.

    python -m unittest serve.test_glm_server
"""
from __future__ import annotations

import json
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve import glm_server as G  # noqa: E402

# GLM-4.7-Flash's template (llama.cpp models/templates/GLM-4.7-Flash.jinja), the format GLM-5.x uses
TEMPLATE = (ROOT / "serve" / "fixtures" / "glm47_flash.jinja")

TOOLS = [{"type": "function", "function": {"name": "read_file", "parameters": {
    "type": "object", "properties": {"path": {"type": "string"}, "lines": {"type": "integer"},
                                     "opts": {"type": "object"}}}}}]


def run_parser(text: str, in_reasoning: bool, tools=None, step: int = 1):
    p = G.GlmParser(in_reasoning, tools)
    events = []
    for i in range(0, len(text), step):
        events += p.feed(text[i:i + step])
    events += p.feed("", final=True)
    # merge consecutive text events of one kind
    out = []
    for kind, val in events:
        if out and kind == out[-1][0] and kind != "tool_call":
            out[-1] = (kind, out[-1][1] + val)
        else:
            out.append((kind, val))
    return out


class Parser(unittest.TestCase):
    def test_reasoning_then_answer(self):
        for step in (1, 3, 100):
            self.assertEqual(run_parser("I should greet.</think>\n\nHello!", True, step=step),
                             [("reasoning", "I should greet."), ("content", "Hello!")])

    def test_no_reasoning(self):
        self.assertEqual(run_parser("Just an answer.", False), [("content", "Just an answer.")])

    def test_reply_opens_its_own_think(self):
        self.assertEqual(run_parser("<think>hmm</think>Answer", False),
                         [("reasoning", "hmm"), ("content", "Answer")])

    def test_tool_call_typed_by_schema(self):
        text = ("Let me look.<tool_call>read_file<arg_key>path</arg_key><arg_value>/tmp/1.txt</arg_value>"
                "<arg_key>lines</arg_key><arg_value>20</arg_value><arg_key>opts</arg_key>"
                "<arg_value>{\"raw\": true}</arg_value></tool_call>")
        for step in (1, 7):
            ev = run_parser(text, False, TOOLS, step=step)
            self.assertEqual(ev[0], ("content", "Let me look."))
            self.assertEqual(ev[1][0], "tool_call")
            self.assertEqual(ev[1][1]["name"], "read_file")
            # "/tmp/1.txt" stays a string; 20 and the object are JSON
            self.assertEqual(json.loads(ev[1][1]["arguments"]), {"path": "/tmp/1.txt", "lines": 20,
                                                                 "opts": {"raw": True}})

    def test_string_param_that_looks_like_json(self):
        ev = run_parser("<tool_call>read_file<arg_key>path</arg_key><arg_value>123</arg_value></tool_call>",
                        False, TOOLS)
        self.assertEqual(json.loads(ev[0][1]["arguments"]), {"path": "123"})

    def test_two_calls_after_reasoning(self):
        text = ("plan</think><tool_call>a<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>\n"
                "<tool_call>b\n<arg_key>y</arg_key>\n<arg_value>two</arg_value>\n</tool_call>")
        ev = run_parser(text, True, step=2)
        self.assertEqual(ev[0], ("reasoning", "plan"))
        self.assertEqual([e[1]["name"] for e in ev[1:]], ["a", "b"])
        self.assertEqual(json.loads(ev[2][1]["arguments"]), {"y": "two"})

    def test_partial_tag_is_held_then_released(self):
        p = G.GlmParser(False)
        self.assertEqual(p.feed("x <tool"), [("content", "x ")])
        self.assertEqual(p.feed("box>"), [("content", "<toolbox>")])

    def test_unclosed_call_at_the_end(self):
        ev = run_parser("<tool_call>f<arg_key>k</arg_key><arg_value>v</arg_value>", False)
        self.assertEqual(ev[0][1]["name"], "f")


@unittest.skipUnless(TEMPLATE.exists(), "template fixture missing")
class Template(unittest.TestCase):
    def setUp(self):
        self.t = G.ChatTemplate(TEMPLATE.read_text(encoding="utf-8"))

    def test_generation_prompt_opens_thinking(self):
        out = self.t.render([{"role": "user", "content": "Hi"}])
        self.assertTrue(out.startswith("[gMASK]<sop>"))
        self.assertTrue(out.endswith("<|user|>Hi<|assistant|><think>"))
        off = self.t.render([{"role": "user", "content": "Hi"}], enable_thinking=False)
        self.assertTrue(off.endswith("<|assistant|></think>"))

    def test_tools_and_tool_results(self):
        msgs = [{"role": "user", "content": "read it"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "c1", "type": "function",
                                 "function": {"name": "read_file", "arguments": {"path": "/a", "lines": 3}}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "file text"}]
        out = self.t.render(msgs, tools=TOOLS)
        self.assertIn('"name": "read_file"', out)
        self.assertIn("<tool_call>read_file<arg_key>path</arg_key><arg_value>/a</arg_value>"
                      "<arg_key>lines</arg_key><arg_value>3</arg_value></tool_call>", out)
        self.assertIn("<|observation|><tool_response>file text</tool_response>", out)


class StopStrings(unittest.TestCase):
    def test_stop_cut_across_pieces(self):
        class FakeEngine:
            max_ctx = 100
            import threading
            lock = threading.Lock()

            def generate(self, ids, max_new, sampling, cancel):
                for t in ids:
                    yield ("token", t)
                yield ("done", {"generated": len(ids), "finish": "length"})

        class FakeTok:
            def token_bytes(self, t):
                return chr(t).encode()

        class FakeModel:
            tokenizer = FakeTok()
            eos_ids = []

        svc = G.Service.__new__(G.Service)
        svc.model, svc.engine = FakeModel(), FakeEngine()
        import threading
        out = "".join(v for k, v in svc.run([ord(c) for c in "abcSTOPdef"], 10, {}, threading.Event(), ["STOP"])
                      if k == "text")
        self.assertEqual(out, "abc")


if __name__ == "__main__":
    unittest.main()
