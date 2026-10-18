"""serve/test_reasoning_tools.py - #804 (strict form of PR #787): a tool call the model writes inside its reasoning.

A call inside the reasoning counts only when tools are declared and the name is one of them; anything else stays
reasoning text.  Without tools the parser is byte-identical to before.

    python -m unittest serve.test_reasoning_tools -v
"""
from __future__ import annotations

import json
import sys
import unittest
import unittest.mock
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import OutputParser  # noqa: E402
from serve.glm_fixtures import GlmTokenizer, call_text, glm_template  # noqa: E402
from serve.server import MockEngine, Service, serve  # noqa: E402

CTX = 8192
CALL = call_text("read_file", limit=120, path="file.txt")
OTHER = CALL.replace("read_file", "not_declared")
TOOLS = [{"name": "read_file",
          "parameters": {"properties": {"limit": {"type": "integer"}, "path": {"type": "string"}}}}]


def run(text, width, tools=TOOLS, thinking=True, stream_tools=True):
    p = OutputParser(thinking=thinking, tools=tools, stream_tools=stream_tools)
    evs = []
    for i in range(0, len(text), width):
        evs += p.feed(text[i:i + width])
    return evs + p.finish()


def joined(evs, kind):
    return "".join(e.text for e in evs if e.kind == kind)


class Parser(unittest.TestCase):
    def test_a_declared_call_in_the_reasoning_is_a_call(self):
        # nothing but whitespace after it, then </think> (#1058); GLM writes it right after the text
        for before in ("before\n", "before"):
            for after in ("\n</think>\n\nanswer", "</think>answer"):
                text = before + CALL + after
                for width in (1, 2, 7, 100_000):
                    with self.subTest(text=text, width=width):
                        evs = run(text, width)
                        calls = [e.call for e in evs if e.kind == "tool_call"]
                        self.assertEqual([(c.name, c.arguments) for c in calls],
                                         [("read_file", {"limit": 120, "path": "file.txt"})])
                        self.assertEqual(joined(evs, "reasoning"), before)
                        self.assertEqual(joined(evs, "content"), "answer")
                        self.assertFalse([e for e in evs if e.kind in ("tool_start", "tool_args")])   # never streamed

    def test_an_undeclared_name_stays_reasoning(self):
        text = "see " + OTHER + " done</think>answer"
        for width in (1, 5, 100_000):
            with self.subTest(width=width):
                evs = run(text, width)
                self.assertFalse([e for e in evs if e.kind == "tool_call"])
                self.assertEqual(joined(evs, "reasoning"), "see " + OTHER + " done")
                self.assertEqual(joined(evs, "content"), "answer")

    def test_no_tools_declared_is_unchanged(self):
        text = "before\n" + CALL + "\n</think>answer"
        for tools in (None, []):
            for width in (1, 7, 100_000):
                with self.subTest(tools=tools, width=width):
                    evs = run(text, width, tools=tools)
                    self.assertFalse([e for e in evs if e.kind == "tool_call"])
                    self.assertEqual(joined(evs, "reasoning"), "before\n" + CALL + "\n")
                    self.assertEqual(joined(evs, "content"), "answer")

    def test_an_unfinished_call_is_never_a_call(self):
        for text in ("before<tool_call>read_file",
                     "before<tool_call>read_file<arg_key>path</arg_key><arg_value>file.txt</arg_value>",
                     "before<tool_call>read_file<arg_key>path</arg_key><arg_value>file.txt</arg_value></tool_ca"):
            for width in (1, 100_000):
                with self.subTest(text=text[-20:], width=width):
                    evs = run(text, width)
                    self.assertFalse([e for e in evs if e.kind == "tool_call"])
                    self.assertEqual(joined(evs, "reasoning"), text)

    def test_the_thinking_ending_inside_a_call_makes_it_text(self):
        for thought in ("a<tool_call>read_file<arg_key>path</arg_key><arg_value>hmm",
                        "a<tool_call>read_file<arg_key>path</arg_key>hmm",
                        "a<tool_call>read_file\nhmm"):
            text = thought + "</think>the answer"
            for width in (1, 100_000):
                with self.subTest(thought=thought, width=width):
                    evs = run(text, width)
                    self.assertFalse([e for e in evs if e.kind == "tool_call"])
                    self.assertEqual(joined(evs, "reasoning"), thought)
                    self.assertEqual(joined(evs, "content"), "the answer")

    def test_a_call_in_the_answer_is_unchanged(self):
        text = "think</think>ok " + CALL
        for width in (1, 100_000):
            evs = run(text, width)
            self.assertEqual(len([e for e in evs if e.kind == "tool_call"]), 1)
            self.assertEqual(joined(evs, "reasoning"), "think")
            self.assertEqual(joined(evs, "content"), "ok ")

    def test_a_tag_that_never_closes_does_not_hold_the_thinking_back(self):
        from serve import frontend
        text = ("a\n<tool_call>read_file<arg_key>path</arg_key><arg_value>" + "thinking on and on " * 3000 +
                "</think>answer")
        p, fed = OutputParser(thinking=True, tools=TOOLS, stream_tools=True), []    # (the feed it came from, event)
        with unittest.mock.patch.object(frontend, "RCALL_MAX", 1000):
            for k, i in enumerate(range(0, len(text), 50)):
                fed += [(k, e) for e in p.feed(text[i:i + 50])]
            fed += [(len(text) // 50 + 1, e) for e in p.finish()]
        evs = [e for k, e in fed]
        first_reasoning_at = next(k for k, e in fed if e.kind == "reasoning" and len(e.text) > 100)
        self.assertLess(first_reasoning_at, 100)    # streamed after RCALL_MAX characters, not kept until </think>
        self.assertEqual(joined(evs, "reasoning"), text.split("</think>")[0])
        self.assertEqual(joined(evs, "content"), "answer")

    def test_a_partial_tag_is_held(self):
        p = OutputParser(thinking=True, tools=TOOLS)
        evs = p.feed("hello <tool_c")
        self.assertEqual(joined(evs, "reasoning"), "hello ")
        self.assertEqual(joined(p.feed("ats are nice</think>"), "reasoning"), "<tool_cats are nice")


class ThinkEnd(unittest.TestCase):
    """GLM writes the answer right after `</think>`, so every `</think>` ends the thinking, quoted or not (Qwen's #537
    rule, which kept a quoted tag as reasoning text, is gone)."""

    def test_a_quoted_tag_ends_the_thinking(self):
        # the quotes Qwen's #537 rule kept as reasoning: under GLM the first </think> ends the thinking, and the rest
        # (a second </think> too) is the answer's text
        for quote in ("in rcall, `</think>` could arrive inside the held body",
                      'the mock engine returns "</think>\\n\\n2", no tool call',
                      '```python\ntext = ("</think>\\n\\nI will use the tag now.")\n```',
                      "split it on '</think>' first"):
            text = "plan\n" + quote + "\ndone\n</think>\n\nanswer"
            head, tail = text.split("</think>", 1)
            for width in (1, 7, 100_000):
                with self.subTest(quote=quote[:24], width=width):
                    evs = run(text, width)
                    self.assertEqual(joined(evs, "reasoning"), head)
                    self.assertEqual(joined(evs, "content"), tail)

    def test_the_end_after_a_quote_ends_the_thinking(self):
        for thought in ('she said "yes"', "it is `x`", "it is 'x'", "split it on '"):
            for width in (1, 100_000):
                with self.subTest(thought=thought, width=width):
                    evs = run(thought + "</think>answer", width)
                    self.assertEqual(joined(evs, "reasoning"), thought)
                    self.assertEqual(joined(evs, "content"), "answer")

    def test_a_call_quoted_in_the_thinking_is_not_run(self):
        text = "The call looks like this:\n" + CALL + " in the answer.\n</think>" + CALL
        for width in (1, 7, 100_000):
            with self.subTest(width=width):
                evs = run(text, width)
                self.assertEqual(len([e for e in evs if e.kind == "tool_call"]), 1)      # the answer's own call
                self.assertIn(CALL + " in the answer.", joined(evs, "reasoning"))
                self.assertEqual(joined(evs, "content"), "")

    def test_an_output_that_ends_on_the_tag(self):
        for width in (1, 100_000):
            evs = run('say "</think>', width)
            self.assertEqual(joined(evs, "reasoning"), 'say "')
            self.assertEqual(joined(evs, "content"), "")


PROPS = {"path": {"type": "string"}}
WRITE = {"openai": [{"type": "function", "function": {"name": "write", "parameters": {
             "type": "object", "properties": PROPS}}}],
         "anthropic": [{"name": "write", "input_schema": {"type": "object", "properties": PROPS}}]}


def answers(script, max_tokens=500):
    """(finish reason, the call's arguments) from OpenAI and Anthropic, whole and streamed, for the model's `script`,
    and the raw Anthropic stream."""
    tok = GlmTokenizer()
    svc = Service(MockEngine(tok, script, max_context=CTX), tok, glm_template())
    httpd = serve(svc, port=0)
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    out = {}
    try:
        for api, path in (("openai", "/v1/chat/completions"), ("anthropic", "/v1/messages")):
            for stream in (False, True):
                body = {"model": "x", "max_tokens": max_tokens, "stream": stream, "tools": WRITE[api],
                        "messages": [{"role": "user", "content": "save my notes"}]}
                req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={
                    "Content-Type": "application/json", "anthropic-version": "2023-06-01"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    raw = r.read().decode()
                if stream:
                    evs = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: {")]
                    if api == "openai":
                        out[api, stream] = (evs[-1]["choices"][0]["finish_reason"], "".join(
                            (tc.get("function") or {}).get("arguments") or "" for e in evs
                            for tc in e["choices"][0]["delta"].get("tool_calls") or []))
                    else:
                        out[api, stream] = (evs[-2]["delta"]["stop_reason"], "".join(
                            e["delta"]["partial_json"] for e in evs if e["type"] == "content_block_delta"
                            and e["delta"]["type"] == "input_json_delta"))
                        out["anthropic events"] = evs
                elif api == "openai":
                    c = json.loads(raw)["choices"][0]
                    out[api, stream] = (c["finish_reason"], [tc["function"]["arguments"]
                                                             for tc in c["message"].get("tool_calls") or []])
                else:
                    m = json.loads(raw)
                    out[api, stream] = (m["stop_reason"], [b["input"] for b in m["content"] if b["type"] == "tool_use"])
    finally:
        httpd.shutdown()
        httpd.server_close()
    return out


class OverHttp(unittest.TestCase):
    def test_finish_reasons_and_block_order(self):
        script = "Let me look.\n" + call_text("write", path="file.txt") + "\n</think>done"
        a = answers(script)
        self.assertEqual((a["openai", False][0], [json.loads(x) for x in a["openai", False][1]]),
                         ("tool_calls", [{"path": "file.txt"}]))
        self.assertEqual((a["openai", True][0], json.loads(a["openai", True][1])), ("tool_calls", {"path": "file.txt"}))
        self.assertEqual(a["anthropic", False], ("tool_use", [{"path": "file.txt"}]))
        self.assertEqual(a["anthropic", True][0], "tool_use")
        # the order of the Anthropic stream's blocks: thinking, the call, the answer
        evs = a["anthropic events"]
        order = [e["content_block"]["type"] for e in evs if e["type"] == "content_block_start"]
        self.assertEqual(order, ["thinking", "tool_use", "text"])
        self.assertEqual(json.loads(a["anthropic", True][1]), {"path": "file.txt"})

    def test_a_call_quoted_in_the_thinking_is_no_call(self):
        script = "Let me look.\n" + call_text("write", path="file.txt") + " is the shape.</think>done"
        a = answers(script)
        self.assertEqual(a["openai", False], ("stop", []))
        self.assertEqual(a["anthropic", False], ("end_turn", []))
        self.assertEqual((a["openai", True], a["anthropic", True]), (("stop", ""), ("end_turn", "")))


if __name__ == "__main__":
    unittest.main()
