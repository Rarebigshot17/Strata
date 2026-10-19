"""Model discovery, on-demand loading and safe unloading over the actual HTTP API."""
import json
import io
import os
import subprocess
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

from serve.glm_fixtures import GlmTokenizer, glm_template
from serve.server import QUIT_QUIET_S, EngineDied, EngineStuck, MockEngine, Service, StrataEngine, serve


class ResidentEngine(StrataEngine):
    def __init__(self, tok):
        self.reply = MockEngine(tok, "Hello.", max_context=4096)
        self.max_context, self.info, self.last = 4096, {}, {}
        self.loaded, self.starts, self.closes = False, 0, 0
        self.unloaded = True
        self.progress = None

    def alive(self):
        return self.loaded

    def exit_code(self):
        return None

    def restart(self):
        self.loaded = True
        self.starts += 1
        self.unloaded = False

    def close(self):
        self.loaded = False
        self.closes += 1

    def unload(self):
        self.close()
        self.unloaded = True

    def generate(self, *args, **kwargs):
        yield from self.reply.generate(*args, **kwargs)


class Lifecycle(unittest.TestCase):
    def test_close_cleans_up_after_a_broken_stdin_pipe(self):
        for running in (False, True):
            with self.subTest(running=running):
                engine = StrataEngine("missing-executable", [], lazy=True)
                read_fd, write_fd = os.pipe()
                stdin = os.fdopen(write_fd, "w")
                self.addCleanup(stdin.close)
                os.close(read_fd)                       # the engine's end of the pipe is already gone
                stdin.write("pending")                  # buffered: flush() and close() will hit the broken pipe
                proc = mock.Mock(stdin=stdin, stdout=io.StringIO())
                self.addCleanup(proc.stdout.close)
                proc.poll.side_effect = [None if running else 0, 0]
                engine.proc, engine.pump, engine.log = proc, mock.Mock(), io.StringIO()
                self.addCleanup(engine.log.close)
                engine.ended, engine.progress, engine.last = False, (1, 2), {"generated": 1}

                engine.close()

                self.assertTrue(stdin.closed)
                self.assertTrue(proc.stdout.closed)
                self.assertTrue(engine.log.closed)
                engine.pump.join.assert_called_once_with(timeout=2)
                self.assertIsNone(engine.proc)
                self.assertTrue(engine.ended)
                self.assertIsNone(engine.progress)
                self.assertEqual(engine.last, {})
                if running:                             # its input is gone: it is ending, and is waited for
                    proc.terminate.assert_not_called()
                    proc.wait.assert_called_once_with(timeout=QUIT_QUIET_S)
                else:
                    proc.terminate.assert_not_called()
                    proc.wait.assert_not_called()
                proc.kill.assert_not_called()
                engine.close()                          # repeated cleanup is harmless

    def test_close_sends_stop_quit_and_eof_before_waiting(self):
        class Pipe(io.StringIO):
            sent = None

            def close(self):
                if not self.closed:
                    self.sent = self.getvalue()
                super().close()

        engine = StrataEngine("missing-executable", [], lazy=True)
        proc = mock.Mock()
        proc.stdin, proc.stdout = Pipe(), io.StringIO()
        proc.poll.side_effect = [None, 0]
        proc.wait.side_effect = lambda **kwargs: self.assertTrue(proc.stdin.closed)
        engine.proc = proc
        stdin = proc.stdin
        engine.close()
        self.assertIsNone(engine.proc)
        # the engine reads QUIT only between requests: a STOP first ends a request still running
        self.assertEqual(stdin.sent, "STOP\nQUIT\n")
        proc.kill.assert_not_called()

    def test_close_does_not_forget_a_process_still_exiting(self):
        engine = StrataEngine("missing-executable", [], lazy=True)
        proc = mock.Mock()
        proc.stdin, proc.stdout = io.StringIO(), io.StringIO()
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired("engine", 2)
        engine.proc = proc
        with self.assertRaisesRegex(EngineStuck, "still releasing"):
            engine.close()
        self.assertIs(engine.proc, proc)
        proc.terminate.assert_not_called()              # strata-glm ignores SIGTERM while the server runs
        proc.kill.assert_called_once()

    def test_close_waits_for_an_engine_slow_to_end_after_quit(self):
        # saving the expert profile and giving the memory back can take a while: neither terminated nor killed
        engine = StrataEngine("missing-executable", [], lazy=True)
        proc = mock.Mock()
        proc.stdin, proc.stdout = io.StringIO(), io.StringIO()
        proc.poll.side_effect = [None, 0]
        proc.wait.side_effect = [subprocess.TimeoutExpired("engine", QUIT_QUIET_S), 0]
        engine.proc = proc
        with mock.patch("builtins.print"):
            engine.close()
        proc.terminate.assert_not_called()
        proc.kill.assert_not_called()
        self.assertIsNone(engine.proc)

    def test_native_lazy_constructor_does_not_start_a_process(self):
        engine = StrataEngine("missing-executable", ["--ctx", "16384"], lazy=True)
        self.assertFalse(engine.alive())
        self.assertEqual(engine.max_context, 16384)
        self.assertIsNone(engine.exit_code())
        engine.close()
        # without --ctx: the engine's own default context
        engine = StrataEngine("missing-executable", ["-m", "model.gguf"], lazy=True)
        self.assertEqual(engine.max_context, 32768)
        engine.close()

    def setUp(self):
        tok = GlmTokenizer()
        self.engine = ResidentEngine(tok)
        self.svc = Service(self.engine, tok, glm_template())
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def request(self, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            response = urllib.request.urlopen(req, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            data = response.read().decode()
            return response.status, data if "event-stream" in response.headers.get("Content-Type", "") else json.loads(data)

    def test_discover_chat_unload_and_stream_reload(self):
        self.assertEqual(self.request("/api/health")[1]["service"], "strata")
        self.assertFalse(self.request("/v1/status")[1]["loaded"])
        self.assertEqual(self.request("/v1/models")[1]["data"][0]["status"]["value"], "unloaded")
        self.assertTrue(self.request("/props")[1]["models_autoload"])
        self.assertEqual(self.engine.starts, 0)
        chat = {"messages": [{"role": "user", "content": "Hello"}], "max_tokens": 32, "reasoning_effort": "none"}
        for _ in range(2):
            code, answer = self.request("/v1/chat/completions", chat)
            self.assertEqual(code, 200)
            self.assertEqual(answer["choices"][0]["message"]["content"], "Hello.")
        self.assertEqual(self.engine.starts, 1)
        self.assertFalse(self.request("/v1/unload", {})[1]["loaded"])
        code, stream = self.request("/v1/chat/completions", {**chat, "stream": True})
        self.assertEqual(code, 200)
        self.assertIn("data: [DONE]", stream)
        self.assertEqual(self.engine.starts, 2)

    def test_busy_auth_and_foreign_origin_cannot_unload(self):
        self.engine.loaded = True
        with self.svc.fifo:
            self.assertEqual(self.request("/v1/unload", {})[0], 409)
        self.svc.api_key = "local-secret"
        self.assertEqual(self.request("/v1/unload", {})[0], 401)
        self.assertEqual(self.request("/v1/unload", {}, {"Authorization": "Bearer local-secret",
                                                        "Origin": "https://other.example"})[0], 403)
        self.assertEqual(self.engine.closes, 0)
        self.assertTrue(self.engine.loaded)
        self.assertEqual(self.request("/v1/unload", {}, {"Authorization": "Bearer local-secret"})[0], 200)

    def test_an_engine_that_died_on_load_is_reported_as_such(self):
        # EngineDied is a RuntimeError: it keeps its own answer ("the next request restarts it")
        def died():
            raise EngineDied("the engine ended while it started")
        self.engine.restart = died
        code, body = self.request("/v1/chat/completions", {"messages": [{"role": "user", "content": "Hello"}]})
        self.assertEqual(code, 503)
        self.assertIn("the next request restarts it", body["error"]["message"])

    def test_explicit_load_and_model_matching(self):
        self.assertEqual(self.request("/v1/load", {"model": "other-model"})[0], 404)
        self.assertEqual(self.engine.starts, 0)
        before = time.time()
        code, status = self.request("/v1/load", {"model": self.svc.model})
        self.assertEqual(code, 200)
        self.assertTrue(status["loaded"])
        self.assertEqual(self.engine.starts, 1)
        self.assertGreaterEqual(self.svc.last_request_at, before)
        self.svc.status["queued"] = 1
        self.assertEqual(self.request("/v1/load", {})[0], 409)
        self.assertEqual(self.request("/v1/unload", {})[0], 409)
        self.assertEqual(self.engine.closes, 0)
        self.assertEqual(self.request("/v1/load", {}, {"Content-Type": "text/plain"})[0], 415)


if __name__ == "__main__":
    unittest.main()
