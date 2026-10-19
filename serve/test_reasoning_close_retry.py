"""serve/test_reasoning_close_retry.py - #1053's retry (opt-in "reasoning_close_retry") and a call written in the
thinking: a reply that ends its turn on a call it wrote inside the thinking (#804) has acted; the thinking is not
closed and the turn not continued, so the call is delivered once.

    python3 -m unittest serve.test_reasoning_close_retry -v
"""
from __future__ import annotations

import contextlib
import io
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.glm_fixtures import GlmTokenizer, IdEngine, call_text, glm_template  # noqa: E402
from serve.server import Service  # noqa: E402

TOOLS = [{"name": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}]
CALL = call_text("read", path="a.txt")


def run(scripts, retry=True):
    """-> (the calls, the content, the finish, the engine passes) of one request on IdEngine(scripts)."""
    tok = GlmTokenizer()
    engine = IdEngine(tok, scripts)
    svc = Service(engine, tok, glm_template())
    svc.reasoning_close_retry = retry
    with contextlib.redirect_stdout(io.StringIO()):
        out = list(svc.run(tok.encode("hi"), True, TOOLS, 2000, {}, threading.Event()))
    evs = [x for kind, x in out if kind == "event"]
    return ([(e.call.name, e.call.arguments) for e in evs if e.kind == "tool_call"],
            "".join(e.text for e in evs if e.kind == "content"), out[-1][1]["finish"], engine.turns)


class CallInTheThinking(unittest.TestCase):
    def test_the_call_is_delivered_once(self):
        for end in ("", "\n\n", CALL):                   # (two calls: both delivered, once each)
            with self.subTest(end=end):
                calls, content, finish, passes = run([["I need the file. ", CALL, end, "<|observation|>"],
                                                      [CALL], ["The file says hi."]])
                self.assertEqual(calls, [("read", {"path": "a.txt"})] * (2 if end == CALL else 1))
                self.assertEqual((content, finish, passes), ("", "stop", 1))

    def test_without_a_call_the_thinking_is_closed_once(self):
        calls, content, finish, passes = run([["Two plus two is four."], ["4"]])
        self.assertEqual((calls, content, finish, passes), ([], "4", "stop", 2))
        calls, content, finish, passes = run([["Two plus two is four."], ["4"]], retry=False)
        self.assertEqual((calls, content, finish, passes), ([], "", "stop", 1))


if __name__ == "__main__":
    unittest.main()
