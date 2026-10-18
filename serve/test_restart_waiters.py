"""#1012 (the old #792): requests waiting for the engine when it dies and is started again.

One engine process serves the requests one at a time, in arrival order (Service.fifo); the next request after the
engine died starts it again (Service.ensure_loaded -> StrataEngine.restart).  A request waiting for its turn when
the engine dies either goes on with the new engine or ends with a clean 503; never a hang, never a stray exception.
A fake strata-glm process (the GLM serve protocol) runs behind the real StrataEngine, Service and HTTP server.

    python3 -m unittest serve.test_restart_waiters -v
"""
import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import server
from serve.glm_fixtures import glm_template
from serve.server import ByteTokenizer, EngineDied, Service, StrataEngine, serve

# Every request is answered "ok, done." and GLM's end of turn, <|user|> (259 in server.ByteTokenizer).  Markers in
# the prompt's bytes: REFUSEME is refused (ERR, no DONE); NFSOLO ends its first request with DONE ... nonfinite and
# nothing emitted; LONGREPLY reads its prompt for 15 s first (a PP line every 0.25 s).  A STOP ends the last GEN's
# request, as the engine's does.  --log gets a line for each GEN.
FAKE = r'''
import queue, sys, threading, time
args = sys.argv[1:]
log = open(args[args.index('--log') + 1], 'a', encoding='utf-8')
sys.stdout.reconfigure(line_buffering=True)
commands, received, stopped = queue.Queue(), [0], [0]
def reader():
    for line in sys.stdin:
        line = line.strip()
        if line == 'STOP':
            stopped[0] = received[0]
            continue
        if line.startswith('GEN '):
            received[0] += 1
        commands.put(line)
    commands.put('QUIT')
threading.Thread(target=reader, daemon=True).start()
print('INFO engine=0.2.0 backend=cpu-reference eos=3 park_mib=0 logit_bias=1')
print('READY 4096 stop')
REPLY = list(b'ok, done.') + [259]
fired, gen_no = set(), 0
def is_stopped():
    return stopped[0] >= gen_no
while True:
    line = commands.get()
    if line == 'QUIT':
        break
    if not line.startswith('GEN '):
        continue
    gen_no += 1
    f = line.split()
    max_new, ids = int(f[1]), [int(x) for x in f[-1].split(',')]
    n, text = len(ids), bytes(t & 255 for t in ids)
    log.write('GEN %d\n' % n)
    log.flush()
    if b'REFUSEME' in text:
        print('ERR the prompt (%d tokens) does not fit the context (4096)' % n)
        continue
    if b'NFSOLO' in text and 'nf' not in fired:
        fired.add('nf')
        print('DONE 0 %d 5.0 0.0 nonfinite 0 0 0 0 0 0 0 0.0 %d 0' % (n, n))
        continue
    if b'LONGREPLY' in text:
        log.write('READING\n')
        log.flush()
        for k in range(1, 61):
            if is_stopped():
                break
            time.sleep(0.25)
            print('PP %d %d %.1f 100.0' % (n * k // 60, n, k * 250.0))
    out = 0
    for t in REPLY[:max_new]:
        if is_stopped():
            break
        print('T %d' % t)
        out += 1
    fin = 'stop' if out and REPLY[out - 1] == 259 else 'cancel' if is_stopped() else 'length'
    print('DONE %d %d 1.0 1.0 %s 0 0 0 0 0 0 0 0.0 %d 0' % (out, n, fin, n))
'''


class RestartWaiters(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        script = self.dir / 'strata-glm'
        script.write_text(f'#!{sys.executable}\n' + FAKE, encoding='utf-8')
        script.chmod(0o755)
        self.log = self.dir / 'requests.log'
        self.engine = StrataEngine(str(script), ['--log', str(self.log)])
        self.addCleanup(self.engine.close)
        self.engine.RESTART_RETRY_S = 0.0              # a start that fails is tried again at once
        self.svc = Service(self.engine, ByteTokenizer(), glm_template())
        self.httpd = serve(self.svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f'http://127.0.0.1:{self.httpd.server_address[1]}'

    def chat(self, text, max_tokens=64):
        body = {'messages': [{'role': 'user', 'content': text}], 'max_tokens': max_tokens, 'reasoning_effort': 'none'}
        req = urllib.request.Request(self.base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode())

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=10) as r:
            return json.loads(r.read().decode())

    def post(self, text, out, key):
        try:
            out[key] = ('ok', self.chat(text)['choices'][0]['message']['content'])
        except urllib.error.HTTPError as e:
            with e:
                out[key] = ('http', e.code, json.loads(e.read().decode())['error']['message'])
        except Exception as e:      # noqa: BLE001 - reported by the assertions
            out[key] = ('exc', repr(e))

    def waiters(self, n):
        """n requests waiting for their turn: the test holds it (as a running request would)."""
        self.assertTrue(self.svc.fifo.acquire(timeout=5))
        out, threads = {}, []
        for i in range(n):
            th = threading.Thread(target=self.post, args=(f'question {i}', out, i))
            th.start()
            threads.append(th)
        deadline = time.monotonic() + 20
        while self.svc.fifo.waiting() < n and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertEqual(self.svc.fifo.waiting(), n)
        return out, threads

    def joined(self, threads):
        for th in threads:
            th.join(60)
        self.assertFalse([th for th in threads if th.is_alive()], 'a waiter hung')

    def clean(self):
        self.assertFalse(self.svc.fifo.locked())
        self.assertEqual(self.svc.fifo.waiting(), 0)
        self.assertEqual((self.svc.status['queued'], self.svc.status['busy']), (0, False))

    def kill(self):
        self.engine.proc.kill()
        self.engine.proc.wait(timeout=10)

    def gens(self):
        return [x for x in self.log.read_text(encoding='utf-8').splitlines() if x.startswith('GEN')]

    def test_waiters_go_on_with_the_new_engine(self):
        old = self.engine.proc
        out, threads = self.waiters(3)
        self.kill()
        self.svc.fifo.release()                         # the request that held the turn is gone
        self.joined(threads)
        self.assertEqual(out, {i: ('ok', 'ok, done.') for i in range(3)})
        self.assertIsNot(self.engine.proc, old)
        self.assertTrue(self.engine.alive())
        self.assertEqual(self.engine.max_context, 4096)
        self.clean()

    def test_waiters_get_a_clean_503_when_the_engine_stays_down(self):
        out, threads = self.waiters(3)
        self.kill()
        spawn = self.engine.spawn
        self.engine.spawn = (sys.executable, [], None, None, None, False)   # python stops at --serve, before READY
        try:
            self.svc.fifo.release()
            self.joined(threads)
        finally:
            self.engine.spawn = spawn
        for i in range(3):
            self.assertEqual(out[i][:2], ('http', 503), out[i])
            self.assertIn('engine', out[i][2])
        self.clean()
        # the next request starts the engine again and is served
        self.assertEqual(self.chat('after')['choices'][0]['message']['content'], 'ok, done.')
        self.clean()

    def killed_mid_read(self, others):
        """A long prompt is being read (alone, or with `others` requests arriving meanwhile, which wait for their
        turn) when the engine dies: that request ends at once with a 503, without waiting out a drain for a DONE that
        never comes, and the next ones start the engine again."""
        out = {}
        threads = [threading.Thread(target=self.post, args=('LONGREPLY ' + 'word ' * 740, out, 'long'))]
        threads[0].start()
        deadline = time.monotonic() + 20
        while not (self.log.exists() and 'READING' in self.log.read_text(encoding='utf-8')):
            self.assertLess(time.monotonic(), deadline, 'the long request did not reach the engine')
            time.sleep(0.02)
        for i in range(others):
            th = threading.Thread(target=self.post, args=(f'short {i}', out, i))
            th.start()
            threads.append(th)
        while self.svc.fifo.waiting() < others and time.monotonic() < deadline:
            time.sleep(0.02)
        t0 = time.monotonic()
        self.kill()
        self.joined(threads)
        self.assertLess(time.monotonic() - t0, 20)
        self.assertEqual(out['long'][:2], ('http', 503), out)
        self.assertEqual({k: v for k, v in out.items() if k != 'long'}, {i: ('ok', 'ok, done.') for i in range(others)})
        self.clean()
        self.assertEqual(self.chat('after')['choices'][0]['message']['content'], 'ok, done.')

    def test_killed_while_reading_alone(self):
        self.killed_mid_read(0)

    def test_killed_while_reading_with_others_waiting(self):
        self.killed_mid_read(2)

    def test_an_old_request_meets_a_restart_in_progress(self):
        """A request that learns its engine died while restart() has already taken the process (self.proc is None):
        its death note and its send answer with EngineDied, not AttributeError (the client got no reply at all)."""
        self.kill()
        self.engine.close()                                           # what restart() does first
        self.assertIsNone(self.engine.proc)
        self.assertTrue(self.engine.death_note())                    # a sentence, no AttributeError
        self.assertFalse(self.engine.alive())
        self.assertIsNone(self.engine.exit_code())
        with self.assertRaises(EngineDied):
            list(self.engine.generate([1, 2], 8, {}, threading.Event()))

    def test_an_engine_err_answers_at_once(self):
        """#1059: the engine refuses a request up front (ERR, then idle; no DONE follows): the answer comes at once,
        not after a drain of a DONE that never comes, and the turn is given back."""
        out = {}
        t0 = time.monotonic()
        self.post('REFUSEME please', out, 0)
        self.assertLess(time.monotonic() - t0, 20)
        self.assertEqual(out[0][:2], ('http', 400), out)
        self.assertIn('does not fit', out[0][2])
        self.clean()
        self.assertEqual(self.chat('after')['choices'][0]['message']['content'], 'ok, done.')

    def test_a_killed_engine_is_started_by_the_next_request(self):
        self.kill()
        self.assertEqual(self.chat('hi')['choices'][0]['message']['content'], 'ok, done.')
        self.assertTrue(self.engine.alive())

    def test_nonfinite_before_the_first_token_is_sent_again_once(self):
        """#879: DONE ... nonfinite with nothing emitted yet: the same request goes to the engine once more."""
        self.assertEqual(self.chat('NFSOLO alone')['choices'][0]['message']['content'], 'ok, done.')
        self.assertEqual(len(self.gens()), 2, self.gens())

    def test_a_burst_of_connections_is_not_reset(self):
        """30-40 clients at once got "connection reset by peer" with the listen backlog of 5: the server listens with
        a deep backlog (STRATA_HTTP_BACKLOG, 256), so 60 simultaneous requests all get answers, one after another."""
        self.assertGreaterEqual(server.Server.request_queue_size, 64)
        errs, ok = [], []

        def one(i):
            try:
                self.assertEqual(self.chat(f'burst {i}', max_tokens=16)['choices'][0]['message']['content'],
                                 'ok, done.')
                ok.append(i)
            except Exception as e:                      # noqa: BLE001 - reported below
                errs.append(repr(e))
        threads = [threading.Thread(target=one, args=(i,)) for i in range(60)]
        for th in threads:
            th.start()
        self.joined(threads)
        self.assertEqual((len(ok), errs[:2]), (60, []))
        self.assertEqual(len(self.gens()), 60)
        self.clean()

    def test_a_request_waiting_for_a_reload_is_counted_as_queued(self):
        """#1603: with the model unloaded, a request waits for the turn to start the engine: it is queued the whole
        time, never neither queued nor running."""
        self.assertEqual(self.chat('warm')['choices'][0]['message']['content'], 'ok, done.')
        self.engine.unload()
        self.assertTrue(self.svc.fifo.acquire(timeout=5))
        res = {}
        t = threading.Thread(target=self.post, args=('reloads', res, 'r'))
        t.start()
        try:
            deadline = time.monotonic() + 10
            live = {}
            while time.monotonic() < deadline:
                live = self.get('/metrics')['live']
                if live.get('queued'):
                    break
                time.sleep(0.05)
            self.assertEqual((live.get('queued'), live.get('state')), (1, 'unloaded'), live)
        finally:
            self.svc.fifo.release()
        self.joined([t])
        self.assertEqual(res['r'], ('ok', 'ok, done.'))
        self.assertEqual(self.get('/metrics')['live']['queued'], 0)
        self.clean()


if __name__ == '__main__':
    unittest.main()
