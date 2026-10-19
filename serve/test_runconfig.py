"""serve/test_runconfig.py - #564: the web page's Settings view of the run config (GET / POST /config), against the
mock engine: only the listed keys change, every other key of the file stays, a bad value changes nothing, and only
Strata's own page may write it (JSON, no foreign Origin, the API key when one is set).

    python -m unittest serve.test_runconfig
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve import runconfig
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]
CFG = {"exe": "build/strata-glm", "args": ["-m", "models/model-00001-of-00004.gguf", "--ctx", "65536"],
       "model_name": "glm-q3", "port": 8101, "host": "0.0.0.0", "api_key": "",
       "sampling": {"temperature": 0.7, "presence_penalty": 1.5},
       "mcp_servers": {"files": {"command": "npx", "args": ["-y", "server-filesystem", "."]}},
       "cors_origins": ["http://localhost:3000"]}


class Apply(unittest.TestCase):
    def test_only_the_named_keys_change(self):
        new, changed = runconfig.apply(CFG, {"sampling.top_p": 0.95, "reasoning_budget_tokens": 4096,
                                             "vram_reserve_gb": 1.5, "park_mib": 16384, "aliases": "glm, local",
                                             "lazy_load": True})
        self.assertEqual(changed, ["sampling.top_p", "reasoning_budget_tokens", "vram_reserve_gb", "park_mib",
                                   "aliases", "lazy_load"])
        self.assertEqual(new["sampling"], {"temperature": 0.7, "presence_penalty": 1.5, "top_p": 0.95})
        self.assertEqual(new["args"][len(CFG["args"]):], ["--vram-reserve", "1.5", "--park-mib", "16384"])
        self.assertEqual(new["aliases"], ["glm", "local"])
        for k in ("exe", "model_name", "port", "host", "api_key", "mcp_servers", "cors_origins"):
            self.assertEqual(new[k], CFG[k], k)
        self.assertEqual(CFG["sampling"], {"temperature": 0.7, "presence_penalty": 1.5})   # the input is left alone

    def test_null_is_the_default_again(self):
        cfg, _ = runconfig.apply(CFG, {"vram_reserve_gb": 2, "park_mib": 0, "fit_max_tokens": True})
        new, changed = runconfig.apply(cfg, {"vram_reserve_gb": None, "park_mib": None, "fit_max_tokens": None,
                                             "sampling.temperature": None, "idle_unload_s": None})
        self.assertEqual(changed, ["vram_reserve_gb", "park_mib", "fit_max_tokens", "sampling.temperature"])
        self.assertEqual(new["args"], CFG["args"])
        self.assertNotIn("fit_max_tokens", new)
        self.assertEqual(new["sampling"], {"presence_penalty": 1.5})
        new, _ = runconfig.apply({"sampling": {"top_k": 20}}, {"sampling.top_k": None})
        self.assertNotIn("sampling", new)                                 # an empty block goes

    def test_refused(self):
        for changes in ({"api_key": "x"}, {"host": "0.0.0.0"}, {"mcp_servers": {}}, {"before_load": "calc.exe"},
                        {"args": []}, {"exe": "evil.exe"}, {"trusted_origins": ["*"]}, {"allowed_hosts": ["*"]},
                        {"sampling.temperature": -1}, {"sampling.top_p": 0}, {"sampling.top_k": -1},
                        {"sampling.top_k": 2.5}, {"sampling.min_p": 2}, {"reasoning_budget_tokens": 1.5},
                        {"fit_max_tokens": "yes"}, {"effort_position": "end"}, {"vram_reserve_mib": 2048},
                        {"vram_reserve_gb": -1}, {"vram_reserve_gb": "2"}, {"park_mib": 1.5}, {"park_mib": -1},
                        {"idle_unload_s": True}, {"aliases": [1]}, {}, None, ["x"]):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                runconfig.apply(CFG, changes)
        with self.assertRaises(ValueError):                              # one bad value: nothing is applied
            runconfig.apply(CFG, {"fit_max_tokens": True, "sampling.top_k": -1})

    def test_only_finite_numbers(self):
        # Python's json reads NaN, Infinity and 1e400: NaN passed every range test (idle_unload_s NaN unloaded the
        # model at every idle moment), Infinity reached int() (OverflowError), and a huge park_mib or
        # vram_reserve_gb overflowed the engine's cast to size_t
        nan, inf = float("nan"), float("inf")
        for key in ("idle_unload_s", "engine_silence_s", "vram_reserve_gb", "park_mib", "reasoning_budget_tokens",
                    "sampling.temperature", "sampling.top_p", "sampling.top_k", "sampling.min_p"):
            for v in (nan, inf, -inf):
                with self.subTest(key=key, v=v), self.assertRaises(ValueError):
                    runconfig.apply(CFG, {key: v})
        for key, v in (("park_mib", 1e300), ("park_mib", 1 << 50), ("vram_reserve_gb", 1e12)):
            with self.subTest(key=key, v=v), self.assertRaises(ValueError):
                runconfig.apply(CFG, {key: v})
        new, _ = runconfig.apply(CFG, {"vram_reserve_gb": 24, "park_mib": 262144, "idle_unload_s": 86400})
        self.assertEqual(new["args"][len(CFG["args"]):], ["--vram-reserve", "24", "--park-mib", "262144"])

    def test_top_k_has_no_cap(self):                                     # the GLM sampler takes any top_k; 0 = off
        for k in (0, 64, 65, 200, 1000):
            new, _ = runconfig.apply(CFG, {"sampling.top_k": k})
            self.assertEqual(new["sampling"]["top_k"], k)

    def test_view(self):
        v = runconfig.view({**CFG, "args": CFG["args"] + ["--vram-reserve", "1.5", "--park-mib", "4096"]},
                           "x/strata-q2_0.json")
        self.assertEqual(v["file"], "strata-q2_0.json")
        got = {k["key"]: k for k in v["keys"]}
        self.assertEqual(got["sampling.temperature"]["value"], 0.7)
        self.assertEqual(got["vram_reserve_gb"]["value"], 1.5)
        self.assertEqual(got["park_mib"]["value"], 4096)
        self.assertEqual(got["park_mib"]["kind"], "number")
        self.assertNotIn("effort_position", got)
        self.assertIsNone(got["fit_max_tokens"]["value"])
        self.assertEqual(got["anthropic_thinking"]["choices"], ["model", "on_request"])
        self.assertNotIn("api_key", got)
        self.assertNotIn("mcp_servers", got)


class Http(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = Path(self.dir.name) / "strata-q2_0.json"
        self.path.write_text(json.dumps(CFG, indent=1), encoding="utf-8")
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok", max_context=4096), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.svc.config_path = str(self.path)
        self.httpd = serve(self.svc, port=0)
        self.host = f"127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.dir.cleanup()

    def call(self, method, body=None, headers=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(f"http://{self.host}/config", method=method, headers=h,
                                     data=None if body is None else (body if isinstance(body, bytes)
                                                                     else json.dumps(body).encode()))
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def saved(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_read_and_write_from_the_page(self):
        status, v = self.call("GET")
        self.assertEqual(status, 200, v)
        self.assertEqual({k["key"]: k["value"] for k in v["keys"]}["sampling.temperature"], 0.7)
        with contextlib.redirect_stdout(io.StringIO()):
            status, v = self.call("POST", {"set": {"sampling.temperature": 1.0, "open_browser": False}},
                                  {"Origin": f"http://{self.host}"})
        self.assertEqual(status, 200, v)
        self.assertEqual(v["changed"], ["sampling.temperature", "open_browser"])
        cfg = self.saved()
        self.assertEqual(cfg["sampling"]["temperature"], 1.0)
        self.assertIs(cfg["open_browser"], False)
        for k in ("mcp_servers", "cors_origins", "host", "args", "exe"):
            self.assertEqual(cfg[k], CFG[k], k)
        self.assertEqual(json.loads((self.path.parent / "strata-q2_0.json.bak").read_text(encoding="utf-8")), CFG)
        status, v = self.call("POST", {"set": {"open_browser": False}})   # no Origin (a script on this PC): allowed
        self.assertEqual((status, v["changed"]), (200, []))

    def test_only_from_strata_s_own_page(self):
        before = self.path.read_bytes()
        status, _ = self.call("POST", {"set": {"fit_max_tokens": True}}, {"Origin": "https://evil.example"})
        self.assertEqual(status, 403)
        status, _ = self.call("POST", b'{"set": {"fit_max_tokens": true}}', {"Content-Type": "text/plain"})
        self.assertEqual(status, 415)
        for body in ({"set": {"api_key": ""}}, {"set": {"mcp_servers": {"x": {"command": "calc"}}}},
                     {"set": {"sampling.top_k": -1}}, {"nothing": 1}, b"not json",
                     b'{"set": {"idle_unload_s": NaN}}', b'{"set": {"park_mib": Infinity}}',
                     b'{"set": {"park_mib": 1e400}}', b'{"set": {"vram_reserve_gb": -Infinity}}'):
            status, b = self.call("POST", body, {"Origin": f"http://{self.host}"})
            self.assertEqual(status, 400, (body, b))
        self.assertEqual(self.path.read_bytes(), before)                 # nothing was written
        self.assertFalse((self.path.parent / "strata-q2_0.json.bak").exists())

    def test_the_api_key(self):
        self.svc.api_key = "secret"
        self.assertEqual(self.call("GET")[0], 401)
        self.assertEqual(self.call("POST", {"set": {"fit_max_tokens": True}})[0], 401)
        with contextlib.redirect_stdout(io.StringIO()):
            status, _ = self.call("POST", {"set": {"fit_max_tokens": True}}, {"Authorization": "Bearer secret"})
        self.assertEqual(status, 200)
        self.assertIs(self.saved()["fit_max_tokens"], True)

    def test_without_a_run_config(self):
        self.svc.config_path = None
        self.assertEqual(self.call("GET")[0], 404)
        self.assertEqual(self.call("POST", {"set": {"fit_max_tokens": True}})[0], 404)

    def test_the_page_has_the_view(self):
        web = ROOT / "serve" / "web"
        html, js = (web / "index.html").read_text(encoding="utf-8"), (web / "app.js").read_text(encoding="utf-8")
        for el in ("cfg-card", "cfg-form", "cfg-save", "cfg-msg"):
            self.assertIn(f'id="{el}"', html)
        self.assertIn('fetch("config", {method: "POST", headers: headers(true)', js)


if __name__ == "__main__":
    unittest.main()
