import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import urllib.error
import urllib.request

from serve.logit_bias import engine_key, normalize, validate_request
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, serve
from serve.frontend import ChatTemplate


class LogitBias(unittest.TestCase):
    def test_forms_and_transport(self):
        self.assertEqual(normalize({"7": -100, "9": 2.5}, 10), normalize([[7, False], [9, 2.5]], 10))
        self.assertEqual(engine_key([[9, 2.5], [7, False]]), " logit_bias=7:-100,9:2.5")
        self.assertEqual(StrataEngine.sampling_keys({"logit_bias": {"7": -100}}), " logit_bias=7:-100")
        for value in [None, {}, []]:
            self.assertEqual(engine_key(value), "")

    def test_invalid(self):
        bad = [True, 1, "x", [1], [[1]], [[1, 0, 2]], {"-1": 0}, {"1.0": 0}, {"10": 0},
               {"1": False}, [[1, True]], [[1.0, 0]], [[True, 0]], [[1, 0], ["1", 2]],
               {"1": 101}, {"1": -101}, {"1": float('nan')}, {"1": float('inf')}, {"1": 10**1000}]
        for value in bad:
            with self.subTest(value=str(value)[:50]), self.assertRaises(ValueError):
                normalize(value, 10)
        with self.assertRaises(ValueError):
            normalize({str(i): -100 for i in range(10)}, 10)

    def test_large_list(self):
        entries = [[i, False] for i in range(103215)]
        self.assertEqual(len(normalize(entries, 248320)), 103215)
        self.assertEqual(engine_key(entries).count(':-100'), 103215)

    def test_capability(self):
        engine = SimpleNamespace(info={})                 # an engine whose INFO line has no logit_bias=1
        with self.assertRaises(ValueError):
            validate_request({'logit_bias': {'1': -100}}, engine, 10)
        self.assertEqual(validate_request({}, engine, 10), {})
        self.assertEqual(validate_request({'logit_bias': {'1': -100}}, SimpleNamespace(info={'logit_bias': 1}), 10), {1: -100})

    def test_http_rejects_before_stream(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, 'ok', max_context=4096)
        service = Service(engine, tok, ChatTemplate(Path(__file__).parent/'chat_template.jinja'))
        httpd = serve(service, port=0)
        try:
            for bias in [True, {'1': -100}, [[1, True]]]:
                data = json.dumps({'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 2,
                                   'stream': True, 'logit_bias': bias}).encode()
                req = urllib.request.Request(f'http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions',
                                             data=data, headers={'Content-Type': 'application/json'})
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(req, timeout=10)
                with caught.exception as response:
                    self.assertEqual(response.code, 400)
                    self.assertIn('logit_bias', json.loads(response.read())['error']['message'])
        finally:
            httpd.shutdown()
            httpd.server_close()

    def test_every_route_checks_it_before_the_answer(self):
        # /v1/messages and /v1/responses pass the request on as the sampling too: a bad bias failed in the engine,
        # after a stream had started
        tok = ByteTokenizer()
        engine = MockEngine(tok, 'ok', max_context=4096)
        service = Service(engine, tok, ChatTemplate(Path(__file__).parent/'chat_template.jinja'))
        httpd = serve(service, port=0)
        base = f'http://127.0.0.1:{httpd.server_address[1]}'
        msgs = [{'role': 'user', 'content': 'hi'}]
        routes = {'/v1/chat/completions': {'messages': msgs, 'max_tokens': 2},
                  '/v1/completions': {'prompt': 'hi', 'max_tokens': 2},
                  '/v1/messages': {'messages': msgs, 'max_tokens': 2},
                  '/v1/messages/count_tokens': {'messages': msgs},
                  '/v1/responses': {'input': 'hi', 'max_output_tokens': 16}}

        def post(path, body):
            req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=10) as response:
                    return response.status, response.read().decode()
            except urllib.error.HTTPError as e:
                with e:
                    return e.code, e.read().decode()
        try:
            for path, body in routes.items():
                for bias in [True, [[1, True]], {'999999999999': 1}, {'1': -100}]:   # the last: no logit_bias=1
                    if path.endswith('count_tokens') and bias == {'1': -100}:
                        continue                                  # counting runs no engine
                    with self.subTest(path=path, bias=bias):
                        code, text = post(path, {**body, 'stream': True, 'logit_bias': bias})
                        self.assertEqual(code, 400, text)
                        error = json.loads(text)['error']
                        self.assertIn('logit_bias', error['message'])
                        if path == '/v1/responses':
                            self.assertEqual(error.get('param'), 'logit_bias')
            engine.info = {'logit_bias': 1}                       # an engine that takes them: served
            for path, body in routes.items():
                with self.subTest(path=path, bias='valid'):
                    self.assertEqual(post(path, {**body, 'logit_bias': {'1': -100}})[0], 200)
        finally:
            httpd.shutdown()
            httpd.server_close()
