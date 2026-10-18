"""serve/test_cut_calls.py - #211 in the whole (non-streamed) answers: a call the output ended inside is left out, the
calls written whole before it stay, on the OpenAI and Anthropic APIs (mock engine; no model, no GPU).

    python -m unittest serve.test_cut_calls -v
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server as S  # noqa: E402
from serve.glm_fixtures import GlmTokenizer, call_text, glm_template, reply_text  # noqa: E402

PROPS = {"path": {"type": "string"}, "content": {"type": "string"}}
TOOLS = {"openai": [{"type": "function", "function": {"name": "write", "parameters": {"type": "object",
                                                                                      "properties": PROPS}}}],
         "anthropic": [{"name": "write", "input_schema": {"type": "object", "properties": PROPS}}]}
FIRST = call_text("write", path="a.txt", content="all of a")
CUT = "<tool_call>write<arg_key>path</arg_key><arg_value>b.txt</arg_value><arg_key>content</arg_key><arg_value>half"


class CutCalls(unittest.TestCase):
    def answer(self, api, script, max_tokens=500):
        """The whole answer: (finish reason, [the calls' arguments])."""
        tok = GlmTokenizer()
        httpd = S.serve(S.Service(S.MockEngine(tok, script), tok, glm_template()), port=0)
        try:
            path = {"openai": "/v1/chat/completions", "anthropic": "/v1/messages"}[api]
            body = {"model": "x", "max_tokens": max_tokens, "tools": TOOLS[api],
                    "messages": [{"role": "user", "content": "save my notes"}]}
            req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}",
                                         data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
            with urllib.request.urlopen(req, timeout=30) as r:
                d = json.loads(r.read())
        finally:
            httpd.shutdown()
            httpd.server_close()
        if api == "openai":
            choice = d["choices"][0]
            return choice["finish_reason"], [json.loads(c["function"]["arguments"])
                                             for c in choice["message"].get("tool_calls") or []]
        return d["stop_reason"], [b["input"] for b in d["content"] if b["type"] == "tool_use"]

    def test_the_whole_call_stays_and_the_cut_one_goes(self):
        script = reply_text("", "Saving both.", FIRST, CUT)
        self.assertEqual(self.answer("openai", script), ("stop", [{"path": "a.txt", "content": "all of a"}]))
        self.assertEqual(self.answer("anthropic", script), ("end_turn", [{"path": "a.txt", "content": "all of a"}]))

    def test_cut_by_the_token_limit(self):
        script = reply_text("", "", FIRST, CUT + " and more" * 50)
        n = len(GlmTokenizer().encode(reply_text("", "", FIRST, CUT)))
        self.assertEqual(self.answer("openai", script, n), ("length", [{"path": "a.txt", "content": "all of a"}]))
        self.assertEqual(self.answer("anthropic", script, n),
                         ("max_tokens", [{"path": "a.txt", "content": "all of a"}]))

    def test_whole_calls_are_all_kept(self):
        second = call_text("write", path="b.txt", content="all of b")
        want = [{"path": "a.txt", "content": "all of a"}, {"path": "b.txt", "content": "all of b"}]
        self.assertEqual(self.answer("openai", reply_text("", "", FIRST, second)), ("tool_calls", want))
        self.assertEqual(self.answer("anthropic", reply_text("", "", FIRST, second)), ("tool_use", want))


if __name__ == "__main__":
    unittest.main()
