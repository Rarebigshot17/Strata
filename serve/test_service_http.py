"""The service and its HTTP handler under the deployment's load: one engine shared by several agents, a chatbot and
a Discord bot, over the tailnet.  Each test is one finding of the D1 review (service and HTTP group).

    python3 -m unittest serve.test_service_http -v
"""
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

from serve.server import ConvCacheLog


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
