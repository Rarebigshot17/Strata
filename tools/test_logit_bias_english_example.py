"""Tests for tools/logit_bias_english_example.py: the token id it reads from a GGUF, the three request bodies it sends
and its verdict, against a fake server that reads `logit_bias` with the server's own parser (serve/logit_bias.py).
Light: no GPU, no model, a few seconds."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import logit_bias_english_example as LB  # noqa: E402
import strata_tokenizer as ST  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402
from serve import logit_bias  # noqa: E402

MERGES = ["H e", "He l", "Hel l", "Hell o"]


def write_gguf(path: Path) -> int:
    """A byte-level vocabulary where "Hello" is one token (its id is returned) and "Hi" two."""
    tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)] + ["He", "Hel", "Hell", "Hello", "<|user|>"]
    w = GGUFWriter()
    w.add("general.architecture", "glm-test")
    w.add("tokenizer.ggml.model", "gpt2")
    w.add("tokenizer.ggml.pre", "glm5")
    w.add("tokenizer.ggml.tokens", tokens)
    w.add("tokenizer.ggml.merges", MERGES)
    w.add("tokenizer.ggml.token_type", [1] * (len(tokens) - 1) + [3])
    w.write(path)
    return tokens.index("Hello")


class FakeServer(ThreadingHTTPServer):
    """Answers "Hello" unless the request bans `hello_id`, then "Hi" (or "Hello" when `obeys` is off)."""
    daemon_threads = True

    def __init__(self, hello_id: int, vocab: int, obeys: bool = True):
        super().__init__(("127.0.0.1", 0), Handler)
        self.hello_id, self.vocab, self.obeys, self.bodies = hello_id, vocab, obeys, []
        threading.Thread(target=self.serve_forever, daemon=True).start()

    def stop(self):
        self.shutdown()
        self.server_close()

    @property
    def base(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        s = self.server
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        s.bodies.append(body)
        try:
            bias = logit_bias.normalize(body.get("logit_bias"), s.vocab)
        except ValueError as e:
            out, code = {"error": {"message": str(e)}}, 400
        else:
            banned = s.obeys and bias.get(s.hello_id) == -100
            out, code = {"choices": [{"message": {"role": "assistant", "content": "Hi" if banned else "Hello"}}]}, 200
        data = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class LogitBiasExample(unittest.TestCase):
    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.dir = Path(d.name)
        self.gguf = self.dir / "m.gguf"
        self.hello = write_gguf(self.gguf)
        self.vocab = len(ST.Tokenizer.from_gguf(self.gguf).tokens)

    def run_tool(self, srv, *extra):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = LB.main(["--gguf", str(self.gguf), "--url", srv.base, "--api-key", "", *extra])
        return code, out.getvalue() + err.getvalue()

    def server(self, obeys=True):
        srv = FakeServer(self.hello, self.vocab, obeys)
        self.addCleanup(srv.stop)
        return srv

    def test_a_working_ban_passes_and_its_requests_are_the_documented_ones(self):
        srv = self.server()
        code, text = self.run_tool(srv, "--output", str(self.dir / "out"))
        self.assertEqual(code, 0, text)
        self.assertIn(f"is token(s) [{self.hello}]", text)
        self.assertEqual([b.get("logit_bias") for b in srv.bodies],
                         [None, {str(self.hello): -100}, [[self.hello, False]], None])
        for b in srv.bodies:                     # greedy, thinking off, the same question every time
            self.assertEqual((b["temperature"], b["reasoning_effort"], b["messages"]),
                             (0, "none", srv.bodies[0]["messages"]))
        rows = json.loads((self.dir / "out" / "rows.json").read_text(encoding="utf-8"))
        self.assertEqual([r["label"] for r in rows], ["before", "ban-object", "ban-pairs", "cleared"])

    def test_a_ban_the_server_ignores_fails(self):
        code, text = self.run_tool(self.server(obeys=False))
        self.assertEqual(code, 1, text)
        self.assertIn("BAD the ban changes the answer", text)

    def test_a_word_of_several_tokens_is_refused_before_any_request(self):
        srv = self.server()
        code, text = self.run_tool(srv, "--word", "Hi")
        self.assertEqual((code, srv.bodies), (2, []))
        self.assertIn("is not one token", text)


if __name__ == "__main__":
    unittest.main()
