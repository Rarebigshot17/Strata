"""#1527: the READY-read timeout (#1317) ends the engine, and the request that was loading it must end too -
a 503 with the reason, not a connection that never answers (or a misleading 400)."""
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
import json
import urllib.request
from unittest import mock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import serve.server as server                                            # noqa: E402
from serve.server import StrataEngine, Service, ChatTemplate, ByteTokenizer, serve  # noqa: E402

# never prints READY: the READY read times out and the process is killed
SILENT = r'''
import time
time.sleep(3600)
'''

# first run answers READY + one request; after --hang is touched a restart stays silent forever
HANG_ON_RESTART = r'''
import sys, time
from pathlib import Path
hang = Path(sys.argv[sys.argv.index('--hang') + 1])
if hang.exists():
    time.sleep(3600)
print('READY 4096 stop', flush=True)
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
    if line.startswith(('GEN', 'BGEN')):
        print('T 90\nDONE 1 1 0 1 stop', flush=True)
'''

# reads the experts into the page cache before READY, reporting on stderr as strata-glm does (every 2 s there): longer
# than the READY bound in all, but moving on all the time
WARMING_SLOWLY = r'''
import sys, time
for i in range(1, 9):
    sys.stderr.write('warming the experts: %.1f of 8.0 GB (%d s, 1.00 GB/s)\n' % (i, i))
    sys.stderr.flush()
    time.sleep(0.25)
print('READY 4096 stop', flush=True)
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
'''

# reports once, then only the same figure: the disk read is stuck
WARMING_STUCK = r'''
import sys, time
while True:
    sys.stderr.write('warming the experts: 1.0 of 8.0 GB (1 s, 1.00 GB/s)\n')
    sys.stderr.flush()
    time.sleep(0.1)
'''


@unittest.skipIf(os.name == "nt", "the .cmd wrapper keeps the pipe open after the kill (only the wrapper dies), so the "
                                  "close of its stdout waits for the sleeping child; a real engine is one process")
class ReadyTimeout(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self._patchers = [mock.patch.object(server, 'ENGINE_READY_S', 0.5),
                          mock.patch.object(StrataEngine, 'RESTART_RETRY_S', 0.0)]
        for p in self._patchers:
            p.start(); self.addCleanup(p.stop)

    def make_script(self, name, code):
        py = self.root / (name + '.py')
        if os.name == 'nt':
            py.write_text(code); script = self.root / (name + '.cmd')
            script.write_text(f'@"{sys.executable}" "{py}" %*\r\n')
        else:
            script = py; py.write_text('#!' + sys.executable + '\n' + code); py.chmod(0o700)
        return str(script)

    def service(self, script, extra_args=()):
        eng = StrataEngine(script, list(extra_args), lazy=True)
        self.addCleanup(eng.close)
        return Service(eng, ByteTokenizer(), ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))

    def post(self, base):
        body = {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 8}
        req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read()

    def assert503(self, base):
        t0 = time.monotonic()
        try:
            self.post(base)
            self.fail('a request on an engine that never reports READY must not succeed')
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 503)
            body = e.read().decode()
            self.assertIn('did not report READY', body)
            self.assertIn('next request restarts it', body)
        self.assertLess(time.monotonic() - t0, 60)

    def test_a_warm_up_that_moves_on_is_not_ended(self):
        # a cold start from a slower disk: 150 GB at 150 MB/s is more than the bound, and the engine says how far it is
        log = self.root / 'engine.log'
        with mock.patch('builtins.print'):
            eng = StrataEngine(self.make_script('warming', WARMING_SLOWLY), [], log=str(log))
        self.addCleanup(eng.close)
        self.assertTrue(eng.alive())
        self.assertEqual(eng.max_context, 4096)

    def test_a_warm_up_that_stops_moving_is_ended(self):
        log = self.root / 'engine.log'
        t0 = time.monotonic()
        with mock.patch('builtins.print'), self.assertRaises(server.EngineDied) as c:
            StrataEngine(self.make_script('stuck', WARMING_STUCK), [], log=str(log))
        self.assertIn('did not report READY', str(c.exception))
        self.assertLess(time.monotonic() - t0, 10)

    def test_ready_timeout_on_lazy_load_answers_the_request(self):
        # lazy start: restart()'s own bookkeeping must not crash on the missing slot_cv either (#1527)
        svc = self.service(self.make_script('silent', SILENT))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.shutdown); self.addCleanup(httpd.server_close)
        base = f'http://127.0.0.1:{httpd.server_address[1]}'
        self.assert503(base)
        self.assertFalse(svc.engine.starting)

    def test_ready_timeout_on_restart_answers_the_request(self):
        # the reporter's case: the engine had answered once, then a restart hangs on the READY read
        hang = self.root / 'hang'
        svc = self.service(self.make_script('hang_on_restart', HANG_ON_RESTART), ('--hang', str(hang)))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.shutdown); self.addCleanup(httpd.server_close)
        base = f'http://127.0.0.1:{httpd.server_address[1]}'
        status, _ = self.post(base)                              # first request: the engine works
        self.assertEqual(status, 200)
        svc.engine.proc.kill(); svc.engine.proc.wait(10)         # it dies between requests
        hang.touch()                                             # and its restart never reports READY
        self.assert503(base)
        self.assertFalse(svc.engine.starting)


class SpawnFailure(unittest.TestCase):
    """The engine's program cannot be run (missing, not executable, being rebuilt by update.sh): the start ends
    cleanly - the start's narrator stops, the log is closed, and the request that was loading it gets a 503 that says
    why, not a dropped connection."""

    def test_a_missing_engine_on_restart(self):
        root = Path(tempfile.mkdtemp()); self.addCleanup(lambda: __import__('shutil').rmtree(root, True))
        narrators = []
        with mock.patch.object(server, 'narrate_start', lambda *a, **k: narrators.append(a[3])), \
                mock.patch.object(StrataEngine, 'RESTART_RETRY_S', 0.0), mock.patch('builtins.print'):
            eng = StrataEngine(str(root / 'no-such-engine'), [], log=str(root / 'engine.log'), lazy=True)
            self.addCleanup(eng.close)
            svc = Service(eng, ByteTokenizer(), ChatTemplate(Path(__file__).parent / 'chat_template.jinja'))
            httpd = serve(svc, port=0)
            self.addCleanup(httpd.server_close); self.addCleanup(httpd.shutdown)
            req = urllib.request.Request(f'http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions',
                                         data=json.dumps({'messages': [{'role': 'user', 'content': 'hi'}],
                                                          'max_tokens': 8}).encode(),
                                         headers={'Content-Type': 'application/json'})
            with self.assertRaises(urllib.error.HTTPError) as c:
                urllib.request.urlopen(req, timeout=30)
        with c.exception as e:
            self.assertEqual(e.code, 503)
            self.assertIn('no-such-engine', e.read().decode())
        self.assertEqual(len(narrators), 3)                      # restart's three tries
        self.assertTrue(all(done.is_set() for done in narrators), 'a narrator still says "still starting"')
        self.assertTrue(eng.log.closed)
        self.assertFalse(eng.starting)


if __name__ == '__main__':
    unittest.main()
