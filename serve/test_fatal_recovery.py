"""Real pipe/process protocol regressions for the GLM engine's two kinds of ERR line.

`ERR <reason>`: the engine refused one request (malformed, does not fit) and stays alive and in step: a 400, and the
next request goes to the same process.  `ERR internal: <what>`: the engine failed (a CUDA error): the request's state
is gone, it is the server's failure (a 500, or an error event in a stream), and the engine may exit after it; the next
request then starts it again.  No DONE follows either line.  A fake strata-glm process stands in for the engine.

    python3 -m unittest serve.test_fatal_recovery -v
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

from serve.glm_fixtures import GlmTokenizer, glm_template
from serve.server import EngineDied, EngineError, Service, StrataEngine, serve

# The first GEN line fails as --mode says; every later one (and every one after a restart: the state file says the
# first was attempted) answers "Z" and GLM's end of turn, <|user|> (259 in GlmTokenizer).
FAKE = r'''
import sys, time
from pathlib import Path
state = Path(sys.argv[sys.argv.index('--state') + 1])
mode = sys.argv[sys.argv.index('--mode') + 1]
sys.stdout.reconfigure(line_buffering=True)
print('INFO engine=0.2.0 backend=cpu-reference eos=3 park_mib=0 logit_bias=1')
print('READY 4096 stop')
for line in sys.stdin:
    if line.startswith('QUIT'):
        break
    if not line.startswith('GEN '):
        if line.strip() not in ('', 'STOP'):          # as strata-glm: an unknown command is answered
            print('ERR unsupported command: ' + line.split()[0])
        continue
    n = len(line.split()[-1].split(','))
    if not state.exists():
        state.write_text('the first request was attempted')
        if mode == 'invalid':
            print('ERR the prompt (%d tokens) does not fit the context (4096)' % n)
            continue
        if mode in ('late', 'stopped'):
            print('T 90')
        if mode == 'stopped':                           # fails while the stopped request is drained
            sys.stdin.readline()
        print('ERR internal: CUDA error 700 (an illegal memory access was encountered)')
        if mode in ('internal', 'late'):
            sys.exit(1)
        if mode in ('slow-exit', 'stopped'):            # strata-glm unwinds first: a CUDA context, 100+ GB mapped
            time.sleep(2)
            sys.exit(1)
        continue
    print('T 90')
    print('T 259')
    print('DONE 2 %d 1.0 1.0 stop 0 0 0 0 0 0 0 0.0 %d 0' % (n, n))
'''


class FatalRecovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.script = self.root / 'strata-glm'
        self.script.write_text('#!' + sys.executable + '\n' + FAKE, encoding='utf-8')
        self.script.chmod(0o700)

    def engine(self, mode):
        e = StrataEngine(str(self.script), ['--state', str(self.root / 'state'), '--mode', mode])
        self.addCleanup(e.close)
        return e

    def service(self, e):
        svc = Service(e, GlmTokenizer(), glm_template())
        self.addCleanup(lambda: svc.engine.close())
        return svc

    def wait_dead(self, e):
        end = time.monotonic() + 5
        while e.alive() and time.monotonic() < end:
            time.sleep(.01)
        self.assertFalse(e.alive(), 'an engine that exited must not count as loaded for the next request')
        # alive() turns False as soon as the process is gone or its output closed; the pump notes the last ERR line
        # a moment later, and death_note needs both that and the exit code
        e.pump.join(5)
        e.proc.wait(5)

    def run_request(self, svc):
        return list(svc.run([1], False, [], 8, {}, threading.Event()))

    def content(self, events):
        return ''.join(x.text or '' for kind, x in events if kind == 'event' and x.kind == 'content')

    def test_internal_error_is_not_a_success_and_next_request_restarts(self):
        e = self.engine('internal')
        svc = self.service(e)
        with self.assertRaises(EngineError) as c:
            self.run_request(svc)
        self.assertNotIsInstance(c.exception, ValueError)          # the server's failure, not the request's
        self.assertIn('CUDA error 700', str(c.exception))
        self.wait_dead(e)
        self.assertIs(svc.engine, e, 'the failed request must not be retried automatically')
        self.assertEqual(svc.history[-1]['finish'], 'error')
        # #997 #890: the death note gives the engine's own last words, not a guess about RAM
        self.assertIn('exited (code 1) after it reported: internal: CUDA error 700', e.death_note())
        result = self.run_request(svc)
        self.assertTrue(svc.engine.alive())
        self.assertTrue(svc.loaded())
        self.assertEqual(self.content(result), 'Z')
        self.assertEqual((result[-1][1]['finish'], result[-1][1]['completion_tokens']), ('stop', 2))   # Z, <|user|>

    def test_internal_error_with_the_engine_still_running(self):
        # the engine caught the failure and dropped the request's state: it is not ended, the next request uses it
        e = self.engine('internal-alive')
        svc = self.service(e)
        pid = e.proc.pid
        with self.assertRaises(EngineError):
            self.run_request(svc)
        self.assertTrue(e.alive())
        result = self.run_request(svc)
        self.assertEqual(e.proc.pid, pid)
        self.assertEqual((self.content(result), result[-1][1]['finish']), ('Z', 'stop'))

    def test_the_request_after_a_fatal_error_starts_a_new_engine(self):
        # the engine exits after ERR internal only once it has unwound: the request queued behind the failed one must
        # start a new engine at once, not write into the dying one (EngineDied, a second failed request)
        e = self.engine('slow-exit')
        svc = self.service(e)
        with self.assertRaises(EngineError):
            self.run_request(svc)
        self.assertFalse(e.alive())
        self.assertIn('exited (code 1) after it reported: internal: CUDA error 700', e.death_note())
        result = self.run_request(svc)
        self.assertEqual((self.content(result), result[-1][1]['finish']), ('Z', 'stop'))

    def test_a_fatal_error_while_a_stopped_request_drains(self):
        e = self.engine('stopped')
        svc = self.service(e)
        gen = e.generate([1], 8, {}, threading.Event())
        self.assertEqual(next(gen), 90)
        gen.close()                                     # STOP; the engine fails and exits while it is drained
        self.assertFalse(e.alive())
        self.assertEqual(self.content(self.run_request(svc)), 'Z')

    def test_internal_error_after_the_first_tokens(self):
        e = self.engine('late')
        svc = self.service(e)
        events = []
        with self.assertRaises(EngineError):
            for item in svc.run([1], False, [], 8, {}, threading.Event()):
                events.append(item)
        self.assertEqual(self.content(events), 'Z')                  # what came before the failure was sent
        self.assertEqual(svc.history[-1]['finish'], 'error')
        self.wait_dead(e)
        self.assertEqual(self.content(self.run_request(svc)), 'Z')

    def post(self, base, body, path='/v1/chat/completions'):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read().decode()
        except urllib.error.HTTPError as err:
            with err:
                return err.code, err.read().decode()

    def health(self, base):
        with urllib.request.urlopen(base + '/health', timeout=5) as r:
            return json.load(r)

    def http(self, mode):
        svc = self.service(self.engine(mode))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        return svc, f'http://127.0.0.1:{httpd.server_address[1]}'

    def test_http_health_and_next_request_recovery(self):
        svc, base = self.http('internal')
        body = {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 8,
                'chat_template_kwargs': {'enable_thinking': False}, 'stream': True}
        status, text = self.post(base, body)
        self.assertEqual(status, 200)
        self.assertIn('"error"', text)
        self.assertIn('CUDA error 700', text)
        self.assertTrue(text.rstrip().endswith('data: [DONE]'))
        self.wait_dead(svc.engine)
        self.assertFalse(self.health(base)['loaded'])
        body['stream'] = False
        status, text = self.post(base, body)
        self.assertEqual(status, 200, text)
        reply = json.loads(text)
        self.assertEqual((reply['choices'][0]['message']['content'], reply['usage']['completion_tokens']), ('Z', 2))
        self.assertTrue(self.health(base)['loaded'])

    def test_http_refusal_is_a_400_and_the_engine_stays(self):
        svc, base = self.http('invalid')
        pid = svc.engine.proc.pid
        body = {'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 8, 'reasoning_effort': 'none'}
        status, text = self.post(base, body)
        self.assertEqual(status, 400, text)
        error = json.loads(text)['error']
        self.assertEqual(error['type'], 'invalid_request_error')
        self.assertIn('does not fit the context', error['message'])
        self.assertTrue(self.health(base)['loaded'])
        status, text = self.post(base, body)
        self.assertEqual((status, json.loads(text)['choices'][0]['message']['content']), (200, 'Z'))
        self.assertEqual(svc.engine.proc.pid, pid)

    def refusal(self, path, body):
        """`path`'s answer (not streamed) when the engine refuses the request: a fresh engine whose first GEN is
        refused."""
        (self.root / 'state').unlink(missing_ok=True)
        svc, base = self.http('invalid')
        return self.post(base, body, path)

    def test_a_refusal_is_a_400_on_the_other_apis(self):
        for path, body in (('/v1/messages', {'model': 'm', 'max_tokens': 8,
                                             'messages': [{'role': 'user', 'content': 'hi'}]}),
                           ('/v1/completions', {'prompt': 'hi', 'max_tokens': 8})):
            with self.subTest(path=path):
                status, text = self.refusal(path, body)
                self.assertEqual(status, 400, text)
                error = json.loads(text)['error']
                self.assertEqual(error['type'], 'invalid_request_error')
                self.assertIn('does not fit the context', error['message'])

    def test_a_refusal_is_a_400_on_the_responses_api(self):
        status, text = self.refusal('/v1/responses', {'model': 'm', 'input': 'hi', 'max_output_tokens': 8})
        self.assertEqual(status, 400, text)
        error = json.loads(text)['error']
        self.assertEqual(error['type'], 'invalid_request_error')
        self.assertIn('does not fit the context', error['message'])

    def test_validation_error_does_not_kill_engine(self):
        e = self.engine('invalid')
        with self.assertRaises(ValueError) as c:
            list(e.generate([1], 8, {}, threading.Event()))
        self.assertNotIsInstance(c.exception, EngineError)
        self.assertTrue(e.alive())
        self.assertEqual(list(e.generate([1], 8, {}, threading.Event())), [90, 259])   # in step: its own T lines
        self.assertEqual(e.last['finish'], 'stop')

    def test_closed_engine_rejects_without_accessing_closed_pipe(self):
        e = self.engine('internal')
        e.close()
        with self.assertRaises(EngineDied):
            list(e.generate([1], 8, {}, threading.Event()))


if __name__ == '__main__':
    unittest.main()
