"""serve/server.py against the GLM engine's protocol and model files, without a GPU or a model.

A fake engine process (FAKE_ENGINE below) speaks strata-glm's serve protocol: INFO with park_mib and logit_bias,
READY, then for each GEN line RESUME, PP, HB, T and DONE (or ERR / ERR internal), and its "conversations:" lines on
stderr; it records its argv and every line it reads.  It is driven through StrataEngine and, over HTTP, through the
Service.  The other tests use the GLM chat template (serve/fixtures/glm47_flash.jinja) with a byte tokenizer, and a
small GGUF that holds a GLM-style tokenizer and template.

    python3 -m unittest serve.test_glm_engine_protocol -v
"""
from __future__ import annotations

import inspect
import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from serve import server as S  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402

GLM_TEMPLATE = ROOT / "serve" / "fixtures" / "glm47_flash.jinja"

FAKE_ENGINE = r'''
import os, sys
sys.stdout.reconfigure(line_buffering=True)
log = open(os.environ["FAKE_LOG"], "a", encoding="utf-8")

def note(x):
    log.write(x + "\n")
    log.flush()

def err(x):
    sys.stderr.write(x + "\n")
    sys.stderr.flush()

note("ARGV " + " ".join(sys.argv[1:]))
eos = int(os.environ.get("FAKE_EOS", "259"))
err("GPU 0: layers 0-46, 11.2 GB")
err("expert cache: 640 slots, 9000 MiB")
print("INFO engine=0.2.0 backend=cpu-reference eos=3 park_mib=512 logit_bias=1 expert_slots=640 expert_cache_mib=9000")
print("READY 4096 stop")
for line in sys.stdin:
    line = line.strip()
    note(line)
    if line == "QUIT":
        break
    if not line.startswith("GEN "):
        continue
    try:
        mode = open(os.environ["FAKE_MODE"], encoding="utf-8").read().strip()
    except OSError:
        mode = "ok"
    n = len(line.rsplit(" ", 1)[1].split(","))
    if mode == "err":
        print("ERR the prompt does not fit the context")
        continue
    if mode == "internal":
        print("ERR internal: CUDA error 700 (an illegal memory access)")
        continue
    print("RESUME 2")
    print("PP %d %d 1.5 2000.0" % (n, n))
    print("HB")
    for t in (72, 105):
        print("T %d" % t)
        print("HB")
    print("T %d" % eos)
    print("DONE 3 %d 1.5 2.0 stop 0 0 2 30 40 0 0 0.0 %d 0" % (n, n - 2))
    err("conversations: parked 10 tokens (0.500 MiB, 1 ms) for a new conversation")
    err("conversations: resumed a checkpoint at 2 of %d prompt tokens (1 ms)" % n)
    err("conversations: 1 parked, 2 checkpoints, 1.500 of 512.0 MiB")
'''


class GlmBytes(S.ByteTokenizer):
    """The byte tokenizer with GLM's reasoning and tool tags as type-4 tokens (matched always)."""
    SPECIALS = S.ByteTokenizer.SPECIALS + ["<think>", "</think>", "<tool_call>", "</tool_call>"]
    ALWAYS = ("<think>", "</think>", "<tool_call>", "</tool_call>")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class FakeEngine:
    """A StrataEngine on FAKE_ENGINE, with its record (argv and the lines it read), its mode file and its log."""

    def __init__(self, d: str, cfg_extra: dict | None = None):
        self.dir = Path(d)
        self.script = self.dir / "strata-glm"
        self.script.write_text(f"#!{sys.executable}\n" + FAKE_ENGINE, encoding="utf-8")
        self.script.chmod(0o755)
        self.record, self.mode, self.log = self.dir / "record.txt", self.dir / "mode.txt", self.dir / "engine.log"
        cfg = {"exe": str(self.script), "args": ["-m", "model-00001-of-00003.gguf", "--ctx", "4096"],
               **(cfg_extra or {})}
        env = {**S.child_env(cfg), "FAKE_LOG": str(self.record), "FAKE_MODE": str(self.mode)}
        self.args = S.engine_args(cfg, [257, 259, 261])
        self.engine = S.StrataEngine(cfg["exe"], self.args, cwd=d, log=str(self.log), env=env)

    def set_mode(self, mode: str) -> None:
        self.mode.write_text(mode, encoding="utf-8")

    def lines(self) -> list[str]:
        return self.record.read_text(encoding="utf-8").splitlines()


def close_quietly(engine) -> None:
    try:
        engine.close()
    except Exception:  # noqa: BLE001 - a test's cleanup
        pass


class EngineArgsTests(unittest.TestCase):
    def test_eos_ids_and_split_are_added(self):
        cfg = {"args": ["-m", "m.gguf", "--ctx", "8192"], "gpu": [0, 2, 1], "split": [24, 12, 10]}
        self.assertEqual(S.engine_args(cfg, [154820, 154827, 154829]),
                         ["-m", "m.gguf", "--ctx", "8192", "--split", "24,12,10", "--eos-ids", "154820,154827,154829"])

    def test_the_configs_own_eos_ids_and_split_win(self):
        cfg = {"args": ["-m", "m.gguf", "--eos-ids", "1", "--split", "1,1"], "split": "3,4"}
        self.assertEqual(S.engine_args(cfg, [5, 6]), cfg["args"])

    def test_split_must_match_the_cards(self):
        with self.assertRaises(ValueError):
            S.split_value({"gpu": [0, 1, 2], "split": [24, 12]})
        with self.assertRaises(ValueError):
            S.split_value({"split": [24, 0]})

    def test_gpus_are_visible_in_the_order_given(self):
        env = S.child_env({"gpu": [2, 0, 1]})
        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "2,0,1")
        self.assertEqual(env["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")

    def test_context_and_model_from_the_args(self):
        e = S.StrataEngine("strata-glm", ["-m", "GLM-00001-of-00004.gguf", "--ctx", "65536"], cwd="/models",
                           lazy=True)
        self.assertEqual(e.max_context, 65536)
        self.assertEqual(e.model_path, "/models/GLM-00001-of-00004.gguf")

    def test_sampling_keys(self):
        keys = S.StrataEngine.sampling_keys({"temperature": 1.0, "top_p": 0.95, "top_k": 0, "seed": 7,
                                             "logit_bias": {"7": -100}})
        self.assertIn(" top_k=0", keys)                 # 0 is sent: off, the whole vocabulary
        self.assertIn(" top_p=0.95", keys)
        self.assertTrue(keys.endswith(" logit_bias=7:-100"))
        self.assertIn(" top_k=200", S.StrataEngine.sampling_keys({"top_k": 200}))   # no cap
        for gone in ("pin=", "ckpt=", "cvec", "pcie", "spec_min_p", "aux_cpus"):
            self.assertNotIn(gone, keys)


class FakeEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fake = FakeEngine(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(close_quietly, self.fake.engine)

    def run_gen(self, ids, sampling=None, checkpoint=None):
        cancel = threading.Event()
        return [t for t in self.fake.engine.generate(ids, 16, sampling or {}, cancel, checkpoint=checkpoint)
                if t is not None]

    def test_start_reads_info_and_ready(self):
        e = self.fake.engine
        self.assertTrue(e.alive())
        self.assertEqual(e.max_context, 4096)
        self.assertTrue(e.can_stop)
        self.assertEqual(e.info["park_mib"], 512)       # an integer
        self.assertEqual(e.info["logit_bias"], 1)
        self.assertEqual(e.info["version"], "0.2.0")
        self.assertEqual(e.info["backend"], "cpu-reference")
        argv = self.fake.lines()[0]
        self.assertTrue(argv.startswith("ARGV --serve -m model-00001-of-00003.gguf --ctx 4096"), argv)
        self.assertIn("--eos-ids 257,259,261", argv)

    def test_gen_line_and_done(self):
        toks = self.run_gen([1, 2, 3, 4], {"temperature": 1.0, "top_k": 0, "logit_bias": {"9": 2.5}}, checkpoint=3)
        self.assertEqual(toks, [72, 105, 259])           # the end-of-turn id comes through; HB lines are skipped
        gen = next(x for x in self.fake.lines() if x.startswith("GEN "))
        self.assertTrue(gen.startswith("GEN 16 cp=3 temperature=1.0 top_k=0 logit_bias=9:2.5 "), gen)
        self.assertTrue(gen.endswith(" 1,2,3,4"), gen)
        last = self.fake.engine.last
        self.assertEqual((last["generated"], last["prompt_tokens"], last["finish"]), (3, 4, "stop"))
        self.assertEqual((last["reused"], last["hits"], last["lookups"], last["prompt_read"]), (2, 30, 40, 2))
        self.assertNotIn("drafts_offered", last)
        self.assertEqual(self.fake.engine.reused, 2)     # RESUME
        self.assertEqual(self.fake.engine.progress, (4, 4))

    def test_no_checkpoint_no_cp_key(self):
        self.run_gen([1, 2])
        gen = next(x for x in self.fake.lines() if x.startswith("GEN "))
        self.assertNotIn("cp=", gen)

    def test_err_is_a_refusal(self):
        self.fake.set_mode("err")
        with self.assertRaises(ValueError) as c:
            self.run_gen([1, 2])
        self.assertNotIsInstance(c.exception, S.EngineError)
        self.assertIn("does not fit", str(c.exception))
        self.fake.set_mode("ok")
        self.assertEqual(self.run_gen([1, 2]), [72, 105, 259])   # the engine kept step

    def test_err_internal_is_an_engine_error(self):
        self.fake.set_mode("internal")
        with self.assertRaises(S.EngineError) as c:
            self.run_gen([1, 2])
        self.assertNotIsInstance(c.exception, ValueError)
        self.assertIn("CUDA error 700", str(c.exception))

    def test_close_sends_stop_before_quit(self):
        self.run_gen([1, 2])
        self.fake.engine.close()
        self.assertIsNone(self.fake.engine.proc)
        self.assertEqual(self.fake.lines()[-2:], ["STOP", "QUIT"])

    def test_request_line(self):
        self.run_gen([1, 2, 3, 4])
        line = S.request_line(self.fake.engine.last)
        self.assertTrue(line.startswith("[strata] request prompt 4 cached 2 output 3 prompt_read 2 ms"), line)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tok = GlmBytes()
        self.svc = S.Service(S.MockEngine(self.tok, "Hi."), self.tok, ChatTemplate(GLM_TEMPLATE))
        self.messages = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello"},
                         {"role": "assistant", "content": "Hi."}, {"role": "user", "content": "And now?"}]

    def conversation_ids(self, kwargs):
        base = self.svc.template.render(self.messages, tools=None, add_generation_prompt=False, **kwargs)
        return self.tok.encode(base, parse_special=True)

    def test_checkpoint_before_the_generation_prompt(self):
        ids, thinking, _ = self.svc.prepare(self.messages, None, {})
        cp = self.svc.checkpoint.at
        self.assertTrue(thinking)                        # the prompt ends with <|assistant|><think>
        self.assertEqual(ids[:cp], self.conversation_ids({}))
        self.assertEqual(self.tok.decode(ids[cp:]), "<|assistant|><think>")

    def test_thinking_off_and_a_forced_call(self):
        ids, thinking, _ = self.svc.prepare(self.messages, None, {"enable_thinking": False}, force="<tool_call>get")
        cp = self.svc.checkpoint.at
        self.assertFalse(thinking)                       # ... ends with <|assistant|></think>
        self.assertEqual(ids[:cp], self.conversation_ids({"enable_thinking": False}))
        self.assertEqual(self.tok.decode(ids[cp:]), "<|assistant|></think><tool_call>get")

    def test_the_run_sends_it_once(self):
        ids, thinking, max_new = self.svc.prepare(self.messages, None, {})
        cp = self.svc.checkpoint.at
        list(self.svc.run(ids, thinking, None, max_new, {}, threading.Event()))
        self.assertEqual(self.svc.engine.last_checkpoint, cp)
        self.assertIsNone(self.svc.checkpoint.at)       # taken by that run, not left for another

    def test_the_next_turn_goes_on_from_the_checkpoint(self):
        # the next turn's prompt starts with this one's up to the checkpoint; after it the reply is written again
        # without its reasoning, so the engine's state after the generation prompt could not have been continued
        turn1 = self.messages[:2]
        ids1, _, _ = self.svc.prepare(turn1, None, {})
        cp1 = self.svc.checkpoint.at
        turn2 = turn1 + [{"role": "assistant", "content": "Hello!", "reasoning_content": "Greet back."},
                         {"role": "user", "content": "More"}]
        ids2, _, _ = self.svc.prepare(turn2, None, {})
        self.assertEqual(ids2[:cp1], ids1[:cp1])
        self.assertNotEqual(ids2[:len(ids1)], ids1)
        self.assertEqual(self.tok.decode(ids2[cp1:]), "<|assistant|></think>Hello!<|user|>More<|assistant|><think>")
        self.assertGreater(self.svc.checkpoint.at, cp1)

    def test_no_checkpoint_when_the_conversation_is_not_a_prefix(self):
        self.svc.template = ChatTemplate(
            source="{{ messages[0].content }}{% if add_generation_prompt %}!{% else %}?{% endif %}")
        ids, thinking, max_new = self.svc.prepare([{"role": "user", "content": "x"}], None, {})
        self.assertEqual(self.tok.decode(ids), "x!")
        self.assertIsNone(self.svc.checkpoint.at)
        list(self.svc.run(ids, thinking, None, max_new, {}, threading.Event()))
        self.assertIsNone(self.svc.engine.last_checkpoint)   # GEN without cp=

    def test_tool_results_are_text(self):
        msgs = self.messages + [{"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "ls", "arguments": {"path": "."}}}]},
            {"role": "tool", "content": [{"type": "text", "text": "a.txt"}, {"type": "image_url", "image_url": {}}]}]
        ids, _, _ = self.svc.prepare(msgs, None, {})
        text = self.tok.decode(ids)
        self.assertIn("<tool_response>a.txt" + S.IMAGE_NOTE + "</tool_response>", text)

    def test_a_picture_is_refused(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "what is this?"},
                                              {"type": "image_url", "image_url": {"url": "data:,"}}]}]
        with self.assertRaises(ValueError) as c:
            self.svc.prepare(msgs, None, {})
        self.assertIn("text only", str(c.exception))


class StopIdTests(unittest.TestCase):
    def test_glm_turn_ends(self):
        tok = GlmBytes()
        svc = S.Service(S.MockEngine(tok, "x"), tok, ChatTemplate(GLM_TEMPLATE))
        want = {tok.encode(t, parse_special=True)[0] for t in ("<|endoftext|>", "<|user|>", "<|observation|>")}
        self.assertEqual(svc.stop_ids, want)

    def test_the_ggufs_eos_eot_eom_ids(self):
        tok = GlmBytes()
        tok.special_ids = {"tokenizer.ggml.eos_token_id": 262, "tokenizer.ggml.eot_token_id": 259,
                           "tokenizer.ggml.bos_token_id": 256}
        self.assertEqual(S.end_of_turn_ids(tok), [259, 261, 262])

    def test_mock_engine_ends_with_user(self):
        tok = GlmBytes()
        eng = S.MockEngine(tok, "Hi")
        self.assertEqual(list(eng.generate([1], 99, {}, threading.Event()))[-1],
                         tok.encode("<|user|>", parse_special=True)[0])

    def test_byte_tokenizer_is_glm_shaped(self):
        tok = S.ByteTokenizer()
        self.assertEqual(tok.ALWAYS, ())
        ids = tok.encode("[gMASK]<sop><|user|>hi<|assistant|>", parse_special=True)
        self.assertEqual(ids, [256, 257, 259, ord("h"), ord("i"), 260])
        self.assertEqual(len(tok.encode("<|user|>")), len("<|user|>"))   # a control token only with parse_special


class FifoTests(unittest.TestCase):
    def test_arrival_order(self):
        fifo, order = S.Fifo(), []
        fifo.acquire()
        threads = []

        def waiter(i):
            with fifo:
                order.append(i)
        for i in range(6):
            t = threading.Thread(target=waiter, args=(i,))
            t.start()
            threads.append(t)
            deadline = time.monotonic() + 5
            while fifo.waiting() < i + 1 and time.monotonic() < deadline:   # each one queued before the next
                time.sleep(0.005)
        fifo.release()
        for t in threads:
            t.join(5)
        self.assertEqual(order, list(range(6)))

    def test_a_waiter_that_leaves_lets_the_next_one_up(self):
        fifo = S.Fifo()
        fifo.acquire()
        first, second = fifo.join(), fifo.join()
        self.assertFalse(fifo.wait(first, 0.05))
        fifo.leave(first)
        fifo.release()
        self.assertTrue(fifo.wait(second, 1.0))
        self.assertFalse(fifo.acquire(blocking=False))
        fifo.release()
        self.assertTrue(fifo.acquire(blocking=False))

    def test_the_turn_follows_the_ticket_not_the_thread(self):
        fifo, order, threads = S.Fifo(), [], []
        first = fifo.join()
        self.assertTrue(fifo.wait(first, 0))
        tickets = [fifo.join() for _ in range(5)]

        def worker(i):
            deadline = time.monotonic() + 10
            while not fifo.wait(tickets[i], 0.05):
                if time.monotonic() > deadline:
                    return
            order.append(i)
            fifo.release()
        for i in reversed(range(5)):                     # started in the opposite order to their tickets
            threads.append(threading.Thread(target=worker, args=(i,)))
            threads[-1].start()
        fifo.release()
        for t in threads:
            t.join(15)
        self.assertEqual(order, [0, 1, 2, 3, 4])
        self.assertEqual(fifo.waiting(), 0)

    def test_a_request_closed_while_queued_gives_up_its_place(self):
        tok = GlmBytes()
        svc = S.Service(S.MockEngine(tok, "x"), tok, ChatTemplate(GLM_TEMPLATE))
        svc.fifo.acquire()
        with mock.patch.object(S, "QUEUE_BEAT_S", 0.0):
            run = svc.run([1, 2], False, None, 8, {}, threading.Event())
            self.assertEqual(next(run), ("ping", None))  # a keep-alive while it waits
            self.assertEqual(svc.fifo.waiting(), 1)
            run.close()                                  # the client left
            self.assertEqual((svc.fifo.waiting(), svc.status["queued"]), (0, 0))
            cancel = threading.Event()
            run = svc.run([1, 2], False, None, 8, {}, cancel)
            self.assertEqual(next(run), ("ping", None))
            cancel.set()                                 # cancelled while it waits: it ends without the turn
            self.assertEqual([e[1]["finish"] for e in run], ["cancel"])
        self.assertEqual((svc.fifo.waiting(), svc.status["queued"]), (0, 0))
        self.assertTrue(svc.fifo.locked())               # still the holder's: the cancelled one never took it
        self.assertEqual(svc.engine.last_prompt, [])     # neither reached the engine
        svc.fifo.release()
        events = list(svc.run([1, 2], False, None, 8, {}, threading.Event()))   # the next one gets the turn
        self.assertEqual(events[-1][1]["finish"], "stop")
        self.assertFalse(svc.fifo.locked())

    def test_a_cancelled_request_leaves_the_queue(self):
        tok = GlmBytes()
        svc = S.Service(S.MockEngine(tok, "x"), tok, ChatTemplate(GLM_TEMPLATE))
        svc.fifo.acquire()
        cancel = threading.Event()
        cancel.set()
        events = list(svc.run([1, 2], False, None, 8, {}, cancel))
        self.assertEqual(events[-1][1]["finish"], "cancel")
        self.assertEqual(svc.fifo.waiting(), 0)
        self.assertEqual(svc.status["queued"], 0)
        svc.fifo.release()


class HttpBase(unittest.TestCase):
    """A Service on an ephemeral port; post() and get() return (status, JSON or text)."""

    def start(self, svc):
        self.svc = svc
        self.httpd = S.serve(svc, port=0)
        self.addCleanup(self.httpd.server_close)
        self.addCleanup(self.httpd.shutdown)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def call(self, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                status, raw = r.status, r.read().decode()
        except urllib.error.HTTPError as e:
            status, raw = e.code, e.read().decode()
        try:
            return status, json.loads(raw)
        except ValueError:
            return status, raw


class CompletionsTests(HttpBase):
    def setUp(self):
        tok = GlmBytes()
        self.engine = S.MockEngine(tok, "Hello there. STOP here")
        self.start(S.Service(self.engine, tok, ChatTemplate(GLM_TEMPLATE), model_name="glm-test"))

    def test_non_stream(self):
        status, d = self.call("/v1/completions", {"prompt": "[gMASK]<sop>The quick", "max_tokens": 5})
        self.assertEqual(status, 200, d)
        self.assertEqual(d["object"], "text_completion")
        self.assertEqual(d["model"], "glm-test")
        self.assertEqual(d["choices"][0]["text"], "Hello")
        self.assertEqual(d["choices"][0]["finish_reason"], "length")
        prompt = self.svc.tok.encode("[gMASK]<sop>The quick", parse_special=True)
        self.assertEqual(self.engine.last_prompt, prompt)  # the prompt as given, special tokens parsed
        self.assertEqual(d["usage"], {"prompt_tokens": len(prompt), "completion_tokens": 5,
                                      "total_tokens": len(prompt) + 5, "prompt_tokens_details": {"cached_tokens": 0}})

    def test_stop_string_and_end_of_turn(self):
        status, d = self.call("/completions", {"prompt": "x", "stop": [" STOP"]})
        self.assertEqual((status, d["choices"][0]["text"], d["choices"][0]["finish_reason"]),
                         (200, "Hello there.", "stop"))
        status, d = self.call("/v1/completions", {"prompt": "x"})
        self.assertEqual((d["choices"][0]["text"], d["choices"][0]["finish_reason"]), ("Hello there. STOP here", "stop"))

    def test_stream(self):
        status, raw = self.call("/v1/completions", {"prompt": "x", "max_tokens": 3, "stream": True})
        self.assertEqual(status, 200)
        lines = [x[6:] for x in raw.split("\n\n") if x.startswith("data: ")]
        self.assertEqual(lines[-1], "[DONE]")
        chunks = [json.loads(x) for x in lines[:-1]]
        self.assertEqual("".join(c["choices"][0]["text"] for c in chunks), "Hel")
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "length")
        self.assertEqual(chunks[-1]["usage"]["completion_tokens"], 3)

    def test_bad_prompt_is_a_400(self):
        status, d = self.call("/v1/completions", {"prompt": 5})
        self.assertEqual(status, 400)
        self.assertIn("prompt", d["error"]["message"])

    def test_slot_save_is_501_and_vram_is_gone(self):
        status, d = self.call("/slots/0?action=save", {"filename": "a.bin"})
        self.assertEqual(status, 501)
        self.assertEqual(d["error"]["type"], "not_supported_error")
        self.assertEqual(self.call("/v1/vram")[0], 404)
        self.assertEqual(self.call("/v1/vram", {})[0], 404)

    def test_every_slot_action_is_501_and_reaches_no_engine(self):
        for path in ("/slots/0?action=restore", "/slots/0?action=save", "/slots/1?action=save",
                     "/slots/0?action=erase"):
            status, d = self.call(path, {"filename": "a.bin"})
            self.assertEqual((status, d["error"]["type"]), (501, "not_supported_error"), path)
        self.assertEqual(self.engine.last_prompt, [])

    def test_health_says_text_only(self):
        status, d = self.call("/health")
        self.assertEqual((status, d["images"]), (200, False))

    def test_chat_alias(self):
        status, d = self.call("/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                                   "max_tokens": 4})
        self.assertEqual(status, 200, d)
        self.assertEqual(d["object"], "chat.completion")


class EngineOverHttpTests(HttpBase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fake = FakeEngine(self.tmp.name)
        self.addCleanup(close_quietly, self.fake.engine)
        self.tok = GlmBytes()
        self.start(S.Service(self.fake.engine, self.tok, ChatTemplate(GLM_TEMPLATE), model_name="glm-test"))

    def test_metrics_engine_and_conversation_cache(self):
        status, d = self.call("/v1/completions", {"prompt": "[gMASK]<sop>abc", "max_tokens": 8})
        self.assertEqual((status, d["choices"][0]["text"], d["choices"][0]["finish_reason"]), (200, "Hi", "stop"))
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:               # the engine's stderr lines reach its log a moment later
            status, m = self.call("/metrics")
            if m["conversation_cache"]["checkpoints"]:
                break
            time.sleep(0.05)
        eng = m["engine"]
        self.assertEqual((eng["model"], eng["max_context"], eng["version"], eng["backend"]),
                         ("glm-test", 4096, "0.2.0", "cpu-reference"))
        self.assertEqual((eng["park_mib"], eng["expert_slots"], eng["expert_cache_mib"]), (512, 640, 9000))
        cc = m["conversation_cache"]
        self.assertEqual((cc["enabled"], cc["budget_mib"]), (True, 512))
        self.assertEqual((cc["parked"], cc["checkpoints"], cc["bytes"]), (1, 2, int(1.5 * 2**20)))
        self.assertEqual((cc["parks"], cc["restores"], cc["evictions"]), (1, 1, 0))
        self.assertEqual((cc["last_event"], cc["last_tokens"]), ("resumed", 2))
        self.assertIsInstance(cc["last_at"], float)
        n = len(self.tok.encode("[gMASK]<sop>abc", parse_special=True))
        self.assertEqual((cc["requests"], cc["requests_reused"], cc["reused_tokens"], cc["prompt_tokens"]),
                         (1, 1, 2, n))
        self.assertEqual((cc["last_prompt"], cc["last_reused"]), (n, 2))
        req = m["requests"][0]
        self.assertEqual(req["hit_rate"], 0.75)          # hits / lookups from DONE
        self.assertNotIn("drafts_offered", req)
        t = m["totals"]
        self.assertEqual((t["hits"], t["lookups"], t["requests"], t["reused"]), (30, 40, 1, 2))
        for key in ("since", "prompt_tokens", "prompt_ms", "decode_ms", "output_tokens"):
            self.assertIn(key, t)

    def test_err_internal_is_a_500(self):
        self.fake.set_mode("internal")
        status, d = self.call("/v1/completions", {"prompt": "abc", "max_tokens": 4})
        self.assertEqual(status, 500)
        self.assertEqual(d["error"]["type"], "server_error")
        self.assertIn("CUDA error 700", d["error"]["message"])
        status, d = self.call("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                                       "max_tokens": 4})
        self.assertEqual(status, 500, d)

    def test_err_internal_in_a_stream_is_an_error_event(self):
        self.fake.set_mode("internal")
        status, raw = self.call("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                                         "max_tokens": 4, "stream": True})
        self.assertEqual(status, 200)
        errors = [json.loads(x[6:]) for x in raw.split("\n\n") if x.startswith("data: {\"error\"")]
        self.assertEqual(len(errors), 1, raw)
        self.assertIn("CUDA error 700", errors[0]["error"]["message"])
        self.assertTrue(raw.rstrip().endswith("data: [DONE]"))

    def test_err_is_a_400(self):
        self.fake.set_mode("err")
        status, d = self.call("/v1/completions", {"prompt": "abc", "max_tokens": 4})
        self.assertEqual(status, 400)
        self.assertIn("does not fit", d["error"]["message"])

    def test_sampling_defaults_reach_the_gen_line(self):
        self.call("/v1/completions", {"prompt": "abc", "max_tokens": 4})
        gen = [x for x in self.fake.lines() if x.startswith("GEN ")][-1]
        self.assertIn(" temperature=1.0 top_p=0.95 ", gen)   # GLM's, when neither the config nor the request sets them
        self.call("/v1/completions", {"prompt": "abc", "max_tokens": 4, "temperature": 0, "top_k": 0})
        gen = [x for x in self.fake.lines() if x.startswith("GEN ")][-1]
        self.assertNotIn("temperature=", gen)            # 0: greedy, the engine's default
        self.assertIn(" top_k=0 ", gen)


class SharedSettingsTests(unittest.TestCase):
    def test_top_k_and_thinking_off(self):
        self.assertEqual(S.clean_shared_defaults({"top_k": 0, "reasoning_effort": "none"}),
                         {"top_k": 0, "reasoning_effort": "none"})
        self.assertEqual(S.clean_shared_defaults({"top_k": 200.0})["top_k"], 200)
        for bad in ({"top_k": -1}, {"top_k": 2.5}, {"experimental_speed_projection": True}):
            with self.assertRaises(ValueError):
                S.clean_shared_defaults(bad)

    def test_config_sampling_top_k(self):
        self.assertEqual(S.sampling_defaults_from_config({"sampling": {"top_k": 0}}), {"top_k": 0})
        self.assertEqual(S.sampling_defaults_from_config({"sampling": {"top_k": 100}}), {"top_k": 100})
        with self.assertRaises(SystemExit):
            S.sampling_defaults_from_config({"sampling": {"top_k": -2}})


# ------------------------------------------------------------------------------------------------ the model's GGUF
CONTROL = ["[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<|endoftext|>"]
USER_DEFINED = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>",
                "</arg_value>", "<tool_response>", "</tool_response>"]


def write_gguf(path: Path, template: str | None = None, name: str | None = "GLM Test", eos: str = "<|endoftext|>"):
    """A GGUF with no tensors: a byte-level vocabulary (no merges), GLM's special tokens, an eos id and a template."""
    from gguf_writer import GGUFWriter
    from strata_tokenizer import BYTE_TO_UNICODE
    tokens = [BYTE_TO_UNICODE[b] for b in range(256)] + CONTROL + USER_DEFINED
    w = GGUFWriter()
    w.add("general.architecture", "glm-test")
    if name is not None:
        w.add("general.name", name)
    w.add("tokenizer.ggml.model", "gpt2")
    w.add("tokenizer.ggml.pre", "glm5")
    w.add("tokenizer.ggml.tokens", tokens)
    w.add("tokenizer.ggml.token_type", [1] * 256 + [3] * len(CONTROL) + [4] * len(USER_DEFINED))
    w.add("tokenizer.ggml.eos_token_id", tokens.index(eos), "u32")
    if template is not None:
        w.add("tokenizer.chat_template", template)
    w.write(path)
    return tokens


class ModelFilesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def test_tokenizer_template_name_and_turn_ends(self):
        path = self.dir / "glm-00001-of-00002.gguf"
        tokens = write_gguf(path, template=GLM_TEMPLATE.read_text(encoding="utf-8"), eos="<|endoftext|>")
        files = S.ModelFiles(path)
        self.assertEqual(files.name, "GLM Test")
        self.assertEqual(files.template, GLM_TEMPLATE.read_text(encoding="utf-8"))
        tok = files.tokenizer
        self.assertEqual(tok.pre, "glm5")
        self.assertEqual(tok.encode("<|user|>hi", parse_special=True), [tokens.index("<|user|>"), ord("h"), ord("i")])
        self.assertEqual(S.end_of_turn_ids(tok), sorted(tokens.index(t) for t in S.END_OF_TURN))
        # every special token's text is protected inside a message: control and user-defined
        svc = S.Service(S.MockEngine(tok, "x"), tok, ChatTemplate(GLM_TEMPLATE))
        for t in CONTROL + USER_DEFINED:
            self.assertIn(t, svc.literals)

    def test_eos_id_of_the_gguf_joins_the_turn_ends(self):
        path = self.dir / "glm.gguf"
        tokens = write_gguf(path, eos="</tool_call>", name=None)
        files = S.ModelFiles(path)
        self.assertEqual(files.name, S.MODEL_NAME)
        self.assertIsNone(files.template)
        self.assertIn(tokens.index("</tool_call>"), S.end_of_turn_ids(files.tokenizer))

    def test_not_a_tokenizer(self):
        from gguf_writer import GGUFWriter
        w = GGUFWriter()
        w.add("general.architecture", "glm-test")
        w.write(self.dir / "weights-00002-of-00002.gguf")
        with self.assertRaises(ValueError) as c:
            S.ModelFiles(self.dir / "weights-00002-of-00002.gguf")
        self.assertIn("first shard", str(c.exception))

    def test_template_override_and_fallback(self):
        path = self.dir / "glm.gguf"
        write_gguf(path)
        files = S.ModelFiles(path)
        self.assertEqual(S.chat_template_for(files, str(GLM_TEMPLATE)).source, GLM_TEMPLATE.read_text(encoding="utf-8"))
        fallback = S.chat_template_for(files, None)      # the GGUF has none
        self.assertEqual(fallback.source, (ROOT / "serve" / "chat_template.jinja").read_text(encoding="utf-8"))
        with self.assertRaises(SystemExit):
            S.chat_template_for(files, str(self.dir / "missing.jinja"))

    @unittest.skipUnless("source" in inspect.signature(ChatTemplate).parameters,
                         "ChatTemplate(source=...) comes with the GLM frontend")
    def test_the_ggufs_template(self):
        path = self.dir / "glm.gguf"
        write_gguf(path, template=GLM_TEMPLATE.read_text(encoding="utf-8"))
        tpl = S.chat_template_for(S.ModelFiles(path), None)
        self.assertEqual(tpl.source, GLM_TEMPLATE.read_text(encoding="utf-8"))


class MainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)
        write_gguf(self.dir / "glm-00001-of-00001.gguf", name="GLM Main Test")

    def config(self, **extra) -> Path:
        cfg = {"exe": sys.executable, "args": ["-m", "glm-00001-of-00001.gguf", "--ctx", "512"],
               "cwd": str(self.dir), "log": str(self.dir / "engine.log"), **extra}
        path = self.dir / "strata-test.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        return path

    def test_an_engine_that_does_not_start_is_a_clean_error(self):
        # python itself is the "engine": it stops at --serve, an option it does not know, before any READY
        r = subprocess.run([sys.executable, "-m", "serve.server", "--engine", "strata", "--config", str(self.config()),
                            "--port", str(free_port())], cwd=ROOT, capture_output=True, text=True, timeout=120)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("the engine did not start", r.stderr)
        self.assertNotIn("Traceback", r.stderr + r.stdout)

    def test_tokenizer_flag_is_gone(self):
        r = subprocess.run([sys.executable, "-m", "serve.server", "--tokenizer", "pack/full/tokenizer"], cwd=ROOT,
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)
        self.assertIn("GGUF", r.stderr)

    def test_mock_with_the_models_tokenizer_and_name(self):
        port = free_port()
        proc = subprocess.Popen([sys.executable, "-m", "serve.server", "--engine", "mock", "--port", str(port),
                                 "--config", str(self.config(chat_template=str(GLM_TEMPLATE)))],
                                cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            deadline, models = time.monotonic() + 60, None
            while time.monotonic() < deadline and proc.poll() is None:
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as r:
                        models = json.loads(r.read())
                    break
                except OSError:
                    time.sleep(0.2)
            self.assertIsNotNone(models, proc.stdout.read() if proc.poll() is not None else "no answer")
            self.assertEqual(models["data"][0]["id"], "GLM Main Test")
        finally:
            proc.terminate()
            proc.wait(30)
            proc.stdout.close()


if __name__ == "__main__":
    unittest.main()
