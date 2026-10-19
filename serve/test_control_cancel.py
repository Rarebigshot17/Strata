"""#879: a Stop that arrives while the engine reads the prompt (PP or HB lines, no tokens yet) is sent at once, not
after the first token, and the request's DONE is drained before the next request reads the pipe."""
import io
import os
import queue
import sys
import threading
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serve import server  # noqa: E402


class ControlCancelTest(unittest.TestCase):
    def _engine(self, lines):
        """A StrataEngine without a process: its stdin is a StringIO, its stdout lines are `lines`."""
        eng = object.__new__(server.StrataEngine)
        eng.proc = mock.Mock()
        eng.proc.poll.return_value = None
        eng.proc.stdin = io.StringIO()
        eng.ended, eng.can_stop, eng.silence_s = False, True, 0
        eng.info, eng.last = {"logit_bias": 1}, {}
        eng.lines = queue.Queue()
        for line in lines:
            eng.lines.put(line)
        return eng

    def sent(self, eng):
        return eng.proc.stdin.getvalue().splitlines()

    def test_stop_during_prompt_read(self):
        cancel = threading.Event()
        eng = self._engine(["PP 256 4096 1.0 100.0", "PP 512 4096 2.0 100.0", "PP 768 4096 3.0 100.0",
                            "DONE 0 4096 3.0 0.0 cancel 0 0 0 0 0 0 0 0.0 768 0", "T 72"])
        gen = eng.generate([1, 2, 3], 8, {}, cancel)
        self.assertIsNone(next(gen))               # the first PP line, nothing cancelled yet
        self.assertEqual(len(self.sent(eng)), 1)
        self.assertTrue(self.sent(eng)[0].startswith("GEN 8 "))
        cancel.set()
        with self.assertRaises(StopIteration):     # the next PP line: STOP goes out now, then the drain to DONE
            next(gen)
        self.assertEqual(self.sent(eng)[1:], ["STOP"])   # once
        self.assertEqual(eng.last["finish"], "cancel")   # this request's DONE was read ...
        self.assertEqual(eng.lines.get_nowait(), "T 72")  # ... and nothing after it
        self.assertEqual(eng.progress, (512, 4096))   # the line that saw the cancel still counts as progress

    def test_stop_during_a_quiet_chunk(self):
        # a long prompt chunk says only HB: a cancel is seen at the next HB line, not at the chunk's PP line
        cancel = threading.Event()
        cancel.set()
        eng = self._engine(["HB", "HB", "PP 4096 4096 9.0 100.0", "DONE 0 4096 9.0 0.0 cancel 0 0 0 0 0 0 0 0.0 4096 0"])
        self.assertEqual(list(eng.generate([1, 2, 3], 8, {}, cancel)), [])
        self.assertEqual(self.sent(eng)[1:], ["STOP"])
        self.assertIsNone(eng.progress)                 # the PP line came after the STOP: only drained, not read
        self.assertTrue(eng.lines.empty())
        self.assertEqual(eng.last["finish"], "cancel")

    def test_no_stop_without_cancel(self):
        eng = self._engine(["RESUME 2", "PP 256 256 1.0 100.0", "HB", "T 72", "T 259",
                            "DONE 2 256 1.0 2.0 stop 0 0 2 0 0 0 0 0.0 254 0"])
        self.assertEqual([t for t in eng.generate([1, 2, 3], 8, {}, threading.Event()) if t is not None], [72, 259])
        self.assertEqual(len(self.sent(eng)), 1)    # the GEN line only
        self.assertEqual((eng.last["finish"], eng.reused), ("stop", 2))


if __name__ == "__main__":
    unittest.main()
