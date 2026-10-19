"""Offline test of tools/research_run.py against the server's mock engine (no GPU, no model)."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import research_run  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402


class Recorder(MockEngine):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.samplings = []

    def generate(self, ids, max_new, sampling, cancel, *rest, **kw):
        self.samplings.append(dict(sampling))
        yield from super().generate(ids, max_new, sampling, cancel, *rest, **kw)


class ResearchRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = Recorder(tok, "an answer", max_context=8192)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.url = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        cls.dir = tempfile.TemporaryDirectory()
        d = Path(cls.dir.name)
        (d / "doc.txt").write_text("The document. " * 40, encoding="utf-8")
        (d / "q.txt").write_text("What is it?\nWho wrote it?\n\nWhen?\n", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.dir.cleanup()

    def go(self, *extra):
        d = Path(self.dir.name)
        out = d / "out.json"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = research_run.main(["--url", self.url, "--doc", str(d / "doc.txt"), "--questions", str(d / "q.txt"),
                                      "--json", str(out), *extra])
        return code, json.loads(out.read_text(encoding="utf-8")), buf.getvalue()

    def test_split_run_reports_every_question(self):
        self.engine.samplings.clear()
        code, out, text = self.go()
        self.assertEqual(code, 0, text)
        self.assertEqual(out["prefix"], "off")
        self.assertEqual([q["q"] for q in out["questions"]], [0, 1, 2])          # the blank line is skipped
        self.assertTrue(all(q["text"] for q in out["questions"]))
        self.assertEqual(len(self.engine.samplings), 3)
        self.assertIn("median first token", text)

    def test_request_bodies(self):
        split = research_run.request_body("DOC", "Q?", "split", True, 64, None)
        self.assertEqual(split["messages"], [{"role": "user", "content": "DOC"}, {"role": "user", "content": "Q?"}])
        self.assertEqual(split["strata_prefix"], {"messages": 1})
        self.assertEqual((split["temperature"], split["max_tokens"], split["stream"]), (0, 64, True))
        self.assertNotIn("model", split)
        joined = research_run.request_body("DOC", "Q?", "joined", True, 64, "m")
        self.assertEqual(joined["messages"], [{"role": "user", "content": "DOC\n\nQuestion: Q?\nAnswer:"}])
        self.assertEqual((joined["strata_prefix"], joined["model"]), ({"message": 0, "chars": 5}, "m"))
        self.assertNotIn("strata_prefix", research_run.request_body("DOC", "Q?", "split", False, 64, None))

    def test_the_server_accepts_the_prefix_field_and_the_joined_layout(self):
        self.engine.samplings.clear()
        code, out, text = self.go("--prefix", "on", "--layout", "joined", "--parallel", "2")
        self.assertEqual(code, 0, text)
        self.assertTrue(all(q["text"] for q in out["questions"]))
        self.assertEqual(len(self.engine.samplings), 3)

    def test_compare(self):
        d = Path(self.dir.name)
        self.go("--prefix", "off")
        (d / "a.json").write_text((d / "out.json").read_text(encoding="utf-8"), encoding="utf-8")
        self.go("--prefix", "on")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = research_run.main(["--compare", str(d / "a.json"), str(d / "out.json")])
        self.assertEqual(code, 0, buf.getvalue())
        self.assertIn("3 answers identical, 0 differ", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
