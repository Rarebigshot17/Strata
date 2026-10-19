"""The service and its HTTP handler under the deployment's load: one engine shared by several agents, a chatbot and
a Discord bot, over the tailnet.  Each test is one finding of the D1 review (service and HTTP group).

    python3 -m unittest serve.test_service_http -v
"""
import http.client
import json
import os
import socket
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

from serve.glm_fixtures import GlmTokenizer, glm_template
from serve.server import ConvCacheLog, EngineDied, EngineStuck, MockEngine, Service, serve


class RestartableEngine(MockEngine):
    """A MockEngine that can die and be started again, as StrataEngine (auto_load)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running, self.unloaded, self.starting, self.starts = True, False, False, 0

    def alive(self):
        return self.running

    def exit_code(self):
        return None

    restart_s = 0.0

    def restart(self):
        time.sleep(self.restart_s)
        self.running, self.unloaded = True, False
        self.starts += 1

    def generate(self, ids, *args, **kwargs):
        self.order = getattr(self, "order", []) + [self.tok.decode(ids)]
        yield from super().generate(ids, *args, **kwargs)


class Http(unittest.TestCase):
    """A server on a RestartableEngine; `self.base` is its address."""
    reply = "Hello."

    def setUp(self):
        self.tok = GlmTokenizer()
        self.engine = RestartableEngine(self.tok, self.reply, max_context=8192)
        self.svc = Service(self.engine, self.tok, glm_template())
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def call(self, path, body=None, headers=None, timeout=30):
        """-> (status, headers, body bytes)."""
        data = None if body is None else body if isinstance(body, bytes) else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.headers, r.read()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.headers, e.read()


class ModelList(Http):
    def test_a_dead_engine_that_restarts_on_use_is_listed(self):
        """The engine died between requests (OOM) with idle unload off: the next request starts it again, so
        /v1/models still lists the model (as unloaded), or a chatbot's model picker drops it."""
        self.engine.running = False
        for path in ("/v1/models", "/models"):
            code, _, body = self.call(path)
            self.assertEqual(code, 200)
            data = json.loads(body)["data"]
            self.assertEqual([m["id"] for m in data], [self.svc.model], path)
            self.assertEqual(data[0]["status"], {"value": "unloaded"})
        self.engine.starting = True                     # while it starts again
        self.assertEqual(json.loads(self.call("/v1/models")[2])["data"][0]["status"], {"value": "loading"})
        self.assertEqual(self.engine.starts, 0)         # listing it does not start it


class ConvCacheLogThreads(unittest.TestCase):
    """GET /metrics polls the engine's log on every request thread (the Monitor tab every second, a Prometheus
    scrape, a second tab): two polls at once must not count the same lines twice."""

    def test_concurrent_polls_count_each_line_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "engine.log")
            with open(path, "w", encoding="utf-8") as f:
                f.write("conversations: parked 10 tokens (1 MiB, 1 ms) for an earlier engine run\n" * 50)
                start = f.tell()
                for _ in range(200):
                    f.write("conversations: parked 100 tokens (1 MiB, 1 ms) for x\n")
                    f.write("conversations: resumed x at 90 of 120 prompt tokens\n")
            log = ConvCacheLog()
            barrier = threading.Barrier(2, timeout=1)

            class Slow:                             # both polls read the new lines before either moves on
                def __init__(self, *args):
                    self.f = open(*args)

                def __enter__(self):
                    return self

                def __exit__(self, *exc):
                    self.f.close()

                def seek(self, pos):
                    self.f.seek(pos)

                def read(self, n=-1):
                    data = self.f.read(n)
                    try:
                        barrier.wait()
                    except threading.BrokenBarrierError:
                        pass
                    return data

            out = []
            with mock.patch("serve.server.open", Slow, create=True):
                threads = [threading.Thread(target=lambda: out.append(log.poll(path, start))) for _ in range(2)]
                for th in threads:
                    th.start()
                for th in threads:
                    th.join(10)
            self.assertEqual(len(out), 2)
            self.assertEqual((log.state["parks"], log.state["restores"]), (200, 200))
            later = log.poll(path, start)            # and a later poll reads nothing again
            self.assertEqual((later["parks"], later["restores"]), (200, 200))


class LoadButton(Http):
    def test_a_failed_start_is_answered_503(self):
        """POST /load (the Monitor's Load button) when the engine cannot start: a 503 that says why, as /v1/load,
        not a connection closed with no answer."""
        self.engine.running = False
        for err in (EngineDied("the engine exited before it was ready (exit code 1)"),
                    EngineStuck("the engine did not end"), FileNotFoundError(2, "No such file", "strata-glm")):
            with self.subTest(err=type(err).__name__):
                with mock.patch.object(self.engine, "restart", side_effect=err), mock.patch("builtins.print"):
                    code, _, body = self.call("/load", {})
                self.assertEqual(code, 503, body)
                self.assertIn(str(err).split(" (")[0], json.loads(body)["error"]["message"])

    def test_a_missing_engine_executable_is_a_503_for_a_request(self):
        """The engine's executable is gone (update.sh rebuilding it) when a request restarts it: a 503, not a
        connection closed without an answer."""
        self.engine.running = False
        with mock.patch.object(self.engine, "restart", side_effect=FileNotFoundError(2, "No such file", "strata-glm")), \
                mock.patch("builtins.print"):
            code, _, body = self.call("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(code, 503, body)
        self.assertIn("could not be started", json.loads(body)["error"]["message"])


class RefusedAtTurn(Http):
    """The engine was alive when a request's load() checked it and is gone at the request's turn (it died in the
    request ahead, or POST /unload freed the card), and starting it again is refused or fails: the stream has
    begun, so the error arrives as the API's own error event - not as a second HTTP response written into it."""

    BODIES = {"/v1/chat/completions": {"messages": [{"role": "user", "content": "hi"}], "stream": True},
              "/v1/completions": {"prompt": "hi", "stream": True, "max_tokens": 8},
              "/v1/messages": {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 64, "stream": True},
              "/v1/responses": {"input": "hi", "stream": True}}

    def streamed(self, path, restart_error):
        self.engine.running = True                                  # alive when the request's load() looks
        self.assertTrue(self.svc.fifo.acquire(timeout=5))          # the request ahead holds the turn
        out = {}

        def send():
            c = http.client.HTTPConnection("127.0.0.1", self.httpd.server_address[1], timeout=30)
            try:
                c.request("POST", path, json.dumps(self.BODIES[path]), {"Content-Type": "application/json"})
                r = c.getresponse()
                out["status"], out["body"] = r.status, r.read().decode("utf-8", "replace")
            finally:
                c.close()
        th = threading.Thread(target=send)
        th.start()
        deadline = time.monotonic() + 10
        while self.svc.fifo.waiting() < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.engine.running = False                                 # it died in the request ahead
        with mock.patch.object(self.engine, "restart", side_effect=restart_error), mock.patch("builtins.print"):
            self.svc.fifo.release()
            th.join(30)
        self.assertFalse(th.is_alive())
        return out["status"], out["body"]

    def test_the_error_is_an_event_of_the_stream(self):
        from serve.server import GpuBusy
        for err in (GpuBusy("the GPU is in use by another program", retry_after=15),
                    EngineStuck("the engine could not be ended")):
            for path in self.BODIES:
                with self.subTest(err=type(err).__name__, path=path):
                    status, body = self.streamed(path, err)
                    self.assertEqual(status, 200)
                    self.assertNotIn("HTTP/1.", body)
                    self.assertIn(str(err), body)
                    if path == "/v1/messages":
                        self.assertIn("event: error", body)
                    elif path == "/v1/responses":
                        self.assertIn("event: response.failed", body)
                    else:
                        self.assertTrue(body.rstrip().endswith("data: [DONE]"), body)
                    self.assertFalse(self.svc.fifo.locked())


class ResponsesLikeChat(Http):
    """/v1/responses renders the prompt as the chat route does for the same request."""

    def prompt_end(self, path, body):
        code, _, out = self.call(path, body)
        self.assertEqual(code, 200, out)
        return self.tok.decode(self.engine.last_prompt)[-40:]

    def test_the_configs_reasoning_effort_applies(self):
        """The run config's "reasoning_effort": "none" (thinking off for requests that do not say) is used on
        /v1/responses as on the chat and Messages routes."""
        self.svc.config_effort = "none"
        chat = self.prompt_end("/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]})
        messages = self.prompt_end("/v1/messages", {"messages": [{"role": "user", "content": "x"}], "max_tokens": 64})
        responses = self.prompt_end("/v1/responses", {"input": "x"})
        self.assertTrue(chat.endswith("</think>"), chat)
        self.assertEqual(messages, chat)
        self.assertEqual(responses, chat)
        # a request's own effort still wins, and the Chat settings' come before the config's
        self.assertTrue(self.prompt_end("/v1/responses", {"input": "x", "reasoning": {"effort": "high"}})
                        .endswith("<think>"))
        self.svc.shared = {"reasoning_effort": "high"}
        self.assertTrue(self.prompt_end("/v1/responses", {"input": "x"}).endswith("<think>"))

    def test_tool_choice_forces_the_call(self):
        """tool_choice "required" or a named function makes the reply a call, as on the chat route."""
        params = {"type": "object", "properties": {"cmd": {"type": "string"}}}
        chat_tools = [{"type": "function", "function": {"name": n, "parameters": params}} for n in ("run", "read")]
        tools = [{"type": "function", "name": n, "parameters": params} for n in ("run", "read")]
        for choice, chat_choice, end in (("required", "required", "<tool_call>"),
                                         ({"type": "function", "name": "run"},
                                          {"type": "function", "function": {"name": "run"}}, "<tool_call>run")):
            with self.subTest(choice=choice):
                chat = self.prompt_end("/v1/chat/completions", {
                    "messages": [{"role": "user", "content": "x"}], "tools": chat_tools, "tool_choice": chat_choice,
                    "reasoning_effort": "none"})
                responses = self.prompt_end("/v1/responses", {"input": "x", "tools": tools, "tool_choice": choice,
                                                              "reasoning": {"effort": "none"}})
                self.assertTrue(chat.endswith(end), chat)
                self.assertEqual(responses, chat)
        # a namespace's tool named by its own name: the name the template shows
        ns = [{"type": "namespace", "name": "shell", "tools": [{"type": "function", "name": "run",
                                                                "parameters": params}]},
              {"type": "function", "name": "read", "parameters": params}]
        self.assertTrue(self.prompt_end("/v1/responses", {
            "input": "x", "tools": ns, "tool_choice": {"type": "function", "name": "run"},
            "reasoning": {"effort": "none"}}).endswith("<tool_call>shell.run"))


class ResponsesHangUp(Http):
    reply = "word " * 400

    def test_a_client_that_hangs_up_is_not_a_malformed_request(self):
        """A non-streamed /v1/responses whose client closes the connection mid-answer: the run is cancelled and
        the monitor says "disconnected" - not a 400 "the request is malformed (KeyError: 'response')"."""
        self.engine.delay = 0.02
        self.svc.api_monitor = True
        body = json.dumps({"input": "hi", "reasoning": {"effort": "none"}}).encode()
        s = socket.create_connection(("127.0.0.1", self.httpd.server_address[1]), timeout=10)
        s.sendall(b"POST /v1/responses HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                  b"Content-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
        deadline = time.monotonic() + 10
        while not self.svc.status.get("generated") and time.monotonic() < deadline:
            time.sleep(0.02)
        with mock.patch("builtins.print") as printed:
            s.close()                                               # the client hangs up mid-answer
            while time.monotonic() < deadline and not (self.svc.api_requests and
                                                       self.svc.api_requests[-1].get("finished_at")):
                time.sleep(0.05)
        record = self.svc.api_requests[-1]
        self.assertEqual(record["state"], "disconnected", record)
        self.assertNotEqual(record.get("http_status"), 400)
        self.assertFalse([c for c in printed.call_args_list if "malformed" in str(c)])


class ExtremeJson(Http):
    """Valid JSON that Python cannot take as it is (1e999 is infinity; thousands of nested arrays exceed the
    recursion limit) is a 400, not a connection closed without an answer."""

    def test_infinite_numbers_and_deep_nesting_are_400(self):
        hi = [{"role": "user", "content": "hi"}]
        cases = [("/v1/chat/completions", b'{"messages": %s, "max_tokens": 1e999}' % json.dumps(hi).encode()),
                 ("/v1/completions", b'{"prompt": "hi", "max_tokens": 1e999}'),
                 ("/v1/messages", b'{"messages": %s, "max_tokens": 1e999}' % json.dumps(hi).encode()),
                 ("/v1/chat/completions", b"[" * 100000 + b"]" * 100000),
                 ("/settings", b'{"defaults": {"top_k": 1e999}}'),
                 ("/settings", b'{"defaults": ' + b"[" * 100000 + b"]" * 100000 + b"}")]
        for path, body in cases:
            with self.subTest(path=path, body=body[:40]):
                with mock.patch("builtins.print"):
                    code, _, out = self.call(path, body)
                self.assertEqual(code, 400, out)
                self.assertIn("error", json.loads(out))

    def test_the_run_config_view_too(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.svc.config_path = os.path.join(tmp, "strata.json")
            with open(self.svc.config_path, "w", encoding="utf-8") as f:
                json.dump({"model": "m.gguf"}, f)
            for body in (b'{"set": {"ctx": 1e999}}', b'{"set": ' + b"[" * 100000 + b"]" * 100000 + b"}"):
                with self.subTest(body=body[:30]), mock.patch("builtins.print"):
                    code, _, out = self.call("/config", body)
                    self.assertEqual(code, 400, out)


class CheckedBeforeLoad(Http):
    def test_a_bad_request_is_400_without_loading_the_model(self):
        """A request the server refuses anyway is refused before an unloaded model is loaded for it (minutes for
        GLM-5.3-Flash, holding the turn of every other client)."""
        self.engine.running, self.engine.unloaded = False, True
        hi = [{"role": "user", "content": "hi"}]
        picture = [{"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type":
                                                                              "image/png", "data": "AAAA"}}]}]
        cases = [("/v1/messages", {"messages": picture, "max_tokens": 8}),
                 ("/v1/messages", {"messages": hi, "max_tokens": 8, "stop_sequences": 5}),
                 ("/v1/messages", {"messages": hi, "max_tokens": 8, "temperature": "abc"}),
                 ("/v1/messages", {"messages": hi, "max_tokens": 8, "reasoning_budget_tokens": "x"}),
                 ("/v1/messages", {"messages": hi, "max_tokens": 1e999}),
                 ("/v1/chat/completions", {"messages": hi, "stop": 5}),
                 ("/v1/chat/completions", {"messages": hi, "temperature": "abc"}),
                 ("/v1/chat/completions", {"messages": hi, "reasoning_budget_tokens": "x"}),
                 ("/v1/chat/completions", {"messages": hi, "max_tokens": 1e999}),
                 ("/v1/completions", {"prompt": "hi", "max_tokens": 1e999})]
        for path, body in cases:
            with self.subTest(path=path, body=body), mock.patch("builtins.print"):
                self.engine.running, self.engine.starts = False, 0
                code, _, out = self.call(path, json.dumps(body).replace("Infinity", "1e999").encode())
                self.assertEqual(code, 400, out)
                self.assertEqual(self.engine.starts, 0, "loaded for a request it refuses")
        self.engine.running, self.engine.starts = False, 0
        code, _, out = self.call("/v1/messages", {"messages": hi, "max_tokens": 8})   # a good one loads it
        self.assertEqual(code, 200, out)
        self.assertEqual(self.engine.starts, 1)


class ArrivalOrder(Http):
    reply = "abcd" * 10                                                 # 40 tokens (bytes)

    def test_a_request_that_arrives_during_a_restart_keeps_its_place(self):
        """B's turn restarts an engine that died after B's load() looked; C arrives during that restart and D
        after it.  C must not wait in load() behind B's whole answer and then be served after D."""
        self.engine.delay, self.engine.restart_s = 0.05, 1.0           # B's answer takes 2 s, the restart 1 s
        done = {}

        def ask(name):
            code, _, body = self.call("/v1/chat/completions", {"messages": [{"role": "user", "content": name}],
                                                               "reasoning_effort": "none"})
            done[name[-1]] = code

        self.assertTrue(self.svc.fifo.acquire(timeout=5))               # the request ahead holds the turn
        threads = {"B": threading.Thread(target=ask, args=("question B",))}
        threads["B"].start()
        deadline = time.monotonic() + 10
        while self.svc.fifo.waiting() < 1 and time.monotonic() < deadline:
            time.sleep(0.01)
        self.engine.running = False                                     # it died in the request ahead
        with mock.patch("builtins.print"):
            self.svc.fifo.release()                                     # B's turn: it starts the engine again
            time.sleep(0.3)
            threads["C"] = threading.Thread(target=ask, args=("question C",))
            threads["C"].start()                                        # C arrives during the restart
            while not self.engine.running and time.monotonic() < deadline:
                time.sleep(0.01)
            time.sleep(0.8)
            threads["D"] = threading.Thread(target=ask, args=("question D",))
            threads["D"].start()                                        # D arrives after it, while B answers
            for th in threads.values():
                th.join(30)
        self.assertEqual(done, {"B": 200, "C": 200, "D": 200})
        order = [next(x for x in "BCD" if f"question {x}" in prompt) for prompt in self.engine.order]
        self.assertEqual(order, ["B", "C", "D"])
        self.assertEqual(self.engine.starts, 1)
        self.assertEqual((self.svc.status["queued"], self.svc.fifo.waiting(), self.svc.fifo.locked()), (0, 0, False))


if __name__ == "__main__":
    unittest.main()
