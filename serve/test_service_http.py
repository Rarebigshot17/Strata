"""The service and its HTTP handler under the deployment's load: one engine shared by several agents, a chatbot and
a Discord bot, over the tailnet.  Each test is one finding of the D1 review (service and HTTP group).

    python3 -m unittest serve.test_service_http -v
"""
import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest import mock

from serve.glm_fixtures import GlmTokenizer, glm_template
from serve.server import ConvCacheLog, MockEngine, Service, serve


class RestartableEngine(MockEngine):
    """A MockEngine that can die and be started again, as StrataEngine (auto_load)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.running, self.unloaded, self.starting, self.starts = True, False, False, 0

    def alive(self):
        return self.running

    def exit_code(self):
        return None

    def restart(self):
        self.running, self.unloaded = True, False
        self.starts += 1


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


if __name__ == "__main__":
    unittest.main()
