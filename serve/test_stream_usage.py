"""serve/test_stream_usage.py - OpenAI's stream_options.include_usage on both streamed routes, against the mock
engine (no model, no GPU).

    python -m unittest serve.test_stream_usage -v
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server as S  # noqa: E402
from serve.glm_fixtures import GlmTokenizer, glm_template  # noqa: E402


class IncludeUsage(unittest.TestCase):
    def setUp(self):
        tok = GlmTokenizer()
        self.svc = S.Service(S.MockEngine(tok, "Hello there"), tok, glm_template(), model_name="glm-test")
        self.httpd = S.serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def stream(self, path, body):
        """The data lines of a streamed answer, parsed; the last one is the literal [DONE]."""
        req = urllib.request.Request(self.base + path, data=json.dumps({**body, "stream": True}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            lines = [ln[6:] for ln in r.read().decode().splitlines() if ln.startswith("data: ")]
        self.assertEqual(lines[-1], "[DONE]")
        return [json.loads(x) for x in lines[:-1]]

    CHAT = {"messages": [{"role": "user", "content": "hi"}], "reasoning_effort": "none"}
    COMPLETION = {"prompt": "[gMASK]<sop>Say", "max_tokens": 4}

    def test_chat_stream_ends_with_a_usage_chunk(self):
        chunks = self.stream("/v1/chat/completions", {**self.CHAT, "stream_options": {"include_usage": True}})
        tail, last = chunks[-1], chunks[-2]
        self.assertEqual(tail["choices"], [])
        self.assertEqual(tail["object"], "chat.completion.chunk")
        self.assertEqual(tail["id"], last["id"])
        self.assertEqual(last["choices"][0]["finish_reason"], "stop")
        self.assertEqual(tail["usage"]["completion_tokens"], len("Hello there") + 1)   # and the end-of-turn token
        # only the usage chunk carries it, as OpenAI sends it: a client that adds up the usage of every chunk
        # (LangChain's stream_usage) counted it twice when the finish chunk had it too
        self.assertEqual([c for c in chunks if c.get("usage")], [tail])

    def test_completion_stream_ends_with_a_usage_chunk(self):
        chunks = self.stream("/v1/completions", {**self.COMPLETION, "stream_options": {"include_usage": True}})
        tail, last = chunks[-1], chunks[-2]
        self.assertEqual(tail["choices"], [])
        self.assertEqual(tail["object"], "text_completion")
        self.assertEqual(last["choices"][0]["finish_reason"], "length")
        self.assertEqual(tail["usage"]["completion_tokens"], 4)
        self.assertEqual([c for c in chunks if c.get("usage")], [tail])

    def test_without_the_option_nothing_changes(self):
        for path, body in (("/v1/chat/completions", self.CHAT), ("/v1/completions", self.COMPLETION)):
            for opts in (None, {"include_usage": False}, "yes"):
                with self.subTest(path=path, stream_options=opts):
                    chunks = self.stream(path, body if opts is None else {**body, "stream_options": opts})
                    self.assertTrue(all(c["choices"] for c in chunks))
                    self.assertIn("usage", chunks[-1])


if __name__ == "__main__":
    unittest.main()
