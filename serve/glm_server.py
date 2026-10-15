"""serve/glm_server.py - an OpenAI-compatible server for GLM-5.3-Flash on Strata's GLM engine.

    python -m serve.glm_server --model MODEL.gguf [--engine build/strata-glm] [--host 127.0.0.1] [--port 8101]
                               [--ctx 32768] [--api-key KEY] [--cors] [-- ENGINE ARGS...]

The engine (`strata-glm --serve`, or `strata-glm-ref --serve` for tests without a GPU) stays resident and takes one
request at a time over the line protocol in include/strata/glm/serve_loop.hpp; requests that arrive meanwhile wait
their turn. The prompt is the chat template stored in the GGUF, rendered with Jinja the way Hugging Face renders it,
and tokenized with the GGUF's own vocabulary (tools/strata_tokenizer.py). GLM's reply format is parsed back into
OpenAI's: `<think>...</think>` becomes `reasoning_content`, and `<tool_call>name<arg_key>k</arg_key><arg_value>v
</arg_value>...</tool_call>` becomes `tool_calls`.

Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions, POST /v1/completions. Streaming is SSE, with a
keep-alive comment every 10 s while a request waits or its prompt is being read.
"""
from __future__ import annotations

import argparse
import codecs
import collections
import json
import os
import pathlib
import queue
import re
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import jinja2  # noqa: E402
import jinja2.sandbox  # noqa: E402

from gguf_reader import GGUFFile  # noqa: E402
from strata_tokenizer import Tokenizer  # noqa: E402

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
CALL_OPEN, CALL_CLOSE = "<tool_call>", "</tool_call>"
STOP_TEXTS = ("<|endoftext|>", "<|user|>", "<|observation|>")
KEEPALIVE_S = 10.0


# ================================================================ the model's files

class ModelInfo:
    """The GGUF's tokenizer, chat template, end-of-turn tokens and name."""

    def __init__(self, path: str):
        md = GGUFFile(pathlib.Path(path)).metadata
        self.metadata = md
        self.tokenizer = Tokenizer(list(md["tokenizer.ggml.tokens"]), list(md.get("tokenizer.ggml.merges") or []),
                                   list(md.get("tokenizer.ggml.token_type") or []) or None,
                                   md.get("tokenizer.ggml.pre", "glm5"))
        self.template = md.get("tokenizer.chat_template")
        if not self.template:
            raise ValueError("the GGUF has no tokenizer.chat_template")
        ids = self.tokenizer.ids
        self.eos_ids = sorted({ids[t] for t in STOP_TEXTS if t in ids} |
                              ({int(md["tokenizer.ggml.eos_token_id"])} if "tokenizer.ggml.eos_token_id" in md else set()))
        name = md.get("general.name")
        self.name = name if name and name != "n/a" else pathlib.Path(path).stem.split("-0000")[0]


def _tojson(value, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


def _raise(msg):
    raise jinja2.exceptions.TemplateError(msg)


class ChatTemplate:
    """The GGUF's Jinja template, in the environment Hugging Face's apply_chat_template uses."""

    def __init__(self, source: str):
        env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                                           extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = _tojson
        env.globals["raise_exception"] = _raise
        env.globals["strftime_now"] = lambda fmt: time.strftime(fmt)
        self.tpl = env.from_string(source)

    def render(self, messages, tools=None, add_generation_prompt=True, **kwargs) -> str:
        return self.tpl.render(messages=messages, tools=tools or None, add_generation_prompt=add_generation_prompt,
                               **kwargs)


# ================================================================ GLM's reply format -> OpenAI's

class GlmParser:
    """Turns the decoded reply into events, as text arrives:
    ("reasoning", text), ("content", text), ("tool_call", {"name", "arguments"}).

    Text that could still be the start of a tag (`<thi`, `</tool_c`) is held until the next piece decides it."""

    def __init__(self, in_reasoning: bool, tools=None):
        self.mode = "reasoning" if in_reasoning else "content"
        self.buf = ""
        self.started = in_reasoning
        self.strip_nl = False                        # newlines right after </think> or </tool_call> are not content
        self.string_params = {}
        for t in tools or []:
            fn = t.get("function", t) if isinstance(t, dict) else {}
            props = ((fn.get("parameters") or {}).get("properties") or {}) if isinstance(fn, dict) else {}
            self.string_params[fn.get("name")] = {k for k, v in props.items()
                                                  if isinstance(v, dict) and v.get("type") == "string"}

    @staticmethod
    def _hold(text: str, tags) -> int:
        """How many characters at the end of `text` could begin one of `tags`."""
        best = 0
        for tag in tags:
            for n in range(min(len(tag) - 1, len(text)), 0, -1):
                if text.endswith(tag[:n]):
                    best = max(best, n)
                    break
        return best

    def feed(self, text: str, final: bool = False):
        self.buf += text
        events = []
        while True:
            if self.mode == "reasoning":
                i = self.buf.find(THINK_CLOSE)
                if i >= 0:
                    if i:
                        events.append(("reasoning", self.buf[:i]))
                    self.buf = self.buf[i + len(THINK_CLOSE):]
                    self.mode = "content"
                    self.started = self.strip_nl = True
                    continue
                keep = 0 if final else self._hold(self.buf, (THINK_CLOSE,))
                if len(self.buf) > keep:
                    events.append(("reasoning", self.buf[:len(self.buf) - keep]))
                    self.buf = self.buf[len(self.buf) - keep:]
                break
            if self.mode == "content":
                if self.strip_nl:
                    self.buf = self.buf.lstrip("\n")
                    if not self.buf and not final:
                        break
                    self.strip_nl = False
                if not self.started:                 # a reply that opens its own <think> (thinking was not forced)
                    stripped = self.buf.lstrip()
                    if stripped.startswith(THINK_OPEN):
                        self.buf = stripped[len(THINK_OPEN):]
                        self.started = True
                        self.mode = "reasoning"
                        continue
                    if not final and THINK_OPEN.startswith(stripped):
                        break                        # could still become <think>
                    self.started = True
                i = self.buf.find(CALL_OPEN)
                if i >= 0:
                    if self.buf[:i].strip():
                        events.append(("content", self.buf[:i]))
                    self.buf = self.buf[i + len(CALL_OPEN):]
                    self.mode = "call"
                    continue
                keep = 0 if final else self._hold(self.buf, (CALL_OPEN,))
                if len(self.buf) > keep:
                    events.append(("content", self.buf[:len(self.buf) - keep]))
                    self.buf = self.buf[len(self.buf) - keep:]
                break
            # mode == "call"
            i = self.buf.find(CALL_CLOSE)
            if i < 0:
                if final and self.buf.strip():       # an unclosed call at the end: parse what is there
                    events.append(("tool_call", self._call(self.buf)))
                    self.buf = ""
                break
            events.append(("tool_call", self._call(self.buf[:i])))
            self.buf = self.buf[i + len(CALL_CLOSE):]
            self.mode = "content"
            self.started = self.strip_nl = True
        return events

    def _call(self, body: str) -> dict:
        j = body.find("<arg_key>")
        name = (body if j < 0 else body[:j]).strip()
        strings = self.string_params.get(name, set())
        args = {}
        for k, v in re.findall(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", body, re.S):
            k = k.strip()
            if k in strings:
                args[k] = v
                continue
            try:
                args[k] = json.loads(v)
            except ValueError:
                args[k] = v
        return {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}


# ================================================================ the engine process

class EngineDied(RuntimeError):
    pass


class Fifo:
    """The engine runs one request at a time; the others wait their turn in arrival order (a Lock wakes any of
    them, so a steady stream of agent requests could keep the chat waiting)."""

    def __init__(self):
        self._cv = threading.Condition()
        self._waiting: collections.deque = collections.deque()
        self._busy = False

    def join(self) -> object:
        ticket = object()
        with self._cv:
            self._waiting.append(ticket)
        return ticket

    def wait(self, ticket, timeout: float) -> bool:
        """True once it is this ticket's turn (the caller then holds the engine until release())."""
        with self._cv:
            if not self._cv.wait_for(lambda: not self._busy and self._waiting[0] is ticket, timeout):
                return False
            self._waiting.popleft()
            self._busy = True
            return True

    def leave(self, ticket):
        """A waiter that gave up."""
        with self._cv:
            try:
                self._waiting.remove(ticket)
            except ValueError:
                pass
            self._cv.notify_all()

    def release(self):
        with self._cv:
            self._busy = False
            self._cv.notify_all()

    def waiting(self) -> int:
        with self._cv:
            return len(self._waiting)


class Engine:
    def __init__(self, cmd: list[str], log_path: str | None):
        self.cmd = cmd
        self.log = open(log_path, "a", encoding="utf-8") if log_path else subprocess.DEVNULL
        self.queue = Fifo()                          # one request at a time; the others wait here, in order
        self.lines: queue.Queue = queue.Queue()
        self.max_ctx = 0
        self.info = {}
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True,
                                     bufsize=1)
        threading.Thread(target=self._pump, daemon=True).start()
        while True:
            line = self.lines.get()
            if line is None:
                raise EngineDied(f"the engine exited before it was ready (exit code {self.proc.poll()})")
            if line.startswith("INFO "):
                for kv in line.split()[1:]:
                    k, _, v = kv.partition("=")
                    self.info[k] = v
            elif line.startswith("READY "):
                self.max_ctx = int(line.split()[1])
                break

    def _pump(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def generate(self, ids, max_new, sampling: dict, cancel: threading.Event):
        """Yields ("token", id), ("progress", (done, total)), ("heartbeat", None), and finally ("done", stats)."""
        keys = "".join(f" {k}={v}" for k, v in sampling.items() if v is not None)
        try:
            self.proc.stdin.write(f"GEN {int(max_new)}{keys} {','.join(str(int(t)) for t in ids)}\n")
            self.proc.stdin.flush()
        except OSError:
            raise EngineDied("the engine is not running") from None
        stopped = False
        while True:
            try:
                line = self.lines.get(timeout=1.0)
            except queue.Empty:
                if cancel.is_set() and not stopped:
                    self._stop()
                    stopped = True
                yield ("heartbeat", None)
                continue
            if line is None:
                raise EngineDied("the engine stopped unexpectedly")
            if cancel.is_set() and not stopped:
                self._stop()
                stopped = True
            if line.startswith("T "):
                if not stopped:
                    yield ("token", int(line[2:]))
            elif line.startswith("PP "):
                f = line.split()
                yield ("progress", (int(f[1]), int(f[2])))
            elif line.startswith("DONE "):
                f = line.split()
                yield ("done", {"generated": int(f[1]), "prompt": int(f[2]), "prompt_ms": float(f[3]),
                                "decode_ms": float(f[4]), "finish": f[5], "reused": int(f[8]),
                                "hits": int(f[9]), "lookups": int(f[10])})
                return
            elif line.startswith("ERR "):
                raise ValueError(line[4:])

    def _stop(self):
        try:
            self.proc.stdin.write("STOP\n")
            self.proc.stdin.flush()
        except OSError:
            pass

    def close(self):
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            self.proc.kill()


# ================================================================ one request

class BadRequest(ValueError):
    pass


def _sampling(req: dict) -> dict:
    """The request's sampling settings as engine keys; GLM's recommended temperature 1.0 / top-p 0.95 when unset."""
    def num(k):
        v = req.get(k)
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise BadRequest(f"'{k}' must be a number")
        return v
    t = num("temperature")
    s = {"temperature": 1.0 if t is None else float(t), "top_p": num("top_p"), "top_k": num("top_k"),
         "min_p": num("min_p"), "penalty_repeat": num("repetition_penalty"),
         "penalty_freq": num("frequency_penalty"), "penalty_present": num("presence_penalty"), "seed": num("seed")}
    if s["top_p"] is None and s["temperature"] > 0:
        s["top_p"] = 0.95
    if s["top_k"] is not None:
        s["top_k"] = int(s["top_k"])
    if s["seed"] is not None:
        s["seed"] = int(s["seed"])
    return s


class Service:
    def __init__(self, model: ModelInfo, engine: Engine, api_key: str | None, cors: bool):
        self.model = model
        self.engine = engine
        self.template = ChatTemplate(model.template)
        self.api_key = api_key
        self.cors = cors

    # ---- prompt
    def chat_prompt(self, req: dict):
        msgs = req.get("messages")
        if not isinstance(msgs, list) or not msgs:
            raise BadRequest("'messages' must be a non-empty list")
        tools = req.get("tools") if req.get("tool_choice") != "none" else None
        kwargs = dict(req.get("chat_template_kwargs") or {})
        thinking = req.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") in ("enabled", "disabled"):
            kwargs.setdefault("enable_thinking", thinking["type"] == "enabled")
        if "enable_thinking" in kwargs and "thinking" not in kwargs:
            kwargs["thinking"] = kwargs["enable_thinking"]
        if req.get("reasoning_effort") is not None:
            kwargs.setdefault("reasoning_effort", req["reasoning_effort"])
        # tool call arguments arrive as a JSON string in OpenAI's format; the template wants the object
        norm = []
        for m in msgs:
            if not isinstance(m, dict):
                raise BadRequest("every message must be an object")
            m = dict(m)
            if isinstance(m.get("tool_calls"), list):
                calls = []
                for tc in m["tool_calls"]:
                    tc = json.loads(json.dumps(tc))
                    fn = tc.get("function") or {}
                    if isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"] or "{}")
                        except ValueError:
                            fn["arguments"] = {}
                    calls.append(tc)
                m["tool_calls"] = calls
            norm.append(m)
        try:
            text = self.template.render(norm, tools=tools, add_generation_prompt=True, **kwargs)
        except jinja2.exceptions.TemplateError as e:
            raise BadRequest(f"the chat template refused the request: {e}") from None
        ids = self.model.tokenizer.encode(text, parse_special=True)
        return text, ids, tools, self._checkpoint_at(norm, tools, kwargs, text, ids)

    def _checkpoint_at(self, msgs, tools, kwargs, text, ids):
        """Where the engine should checkpoint this prompt: before the generation prompt (`<|assistant|><think>`).
        The next turn re-renders the reply (without its reasoning) after that point, so the engine's state after
        the reply cannot be continued, but a checkpoint there can. Only the generation prompt is tokenized again:
        it begins with a special token, so the tokens before it are the conversation's. A wrong guess costs
        nothing but the checkpoint's use: the engine saves the state of whatever prefix it is given."""
        try:
            base = self.template.render(msgs, tools=tools, add_generation_prompt=False, **kwargs)
        except jinja2.exceptions.TemplateError:
            return None
        if not base or not text.startswith(base) or len(base) == len(text):
            return None
        tail = self.model.tokenizer.encode(text[len(base):], parse_special=True)
        if not tail or len(tail) >= len(ids) or ids[len(ids) - len(tail):] != tail:
            return None
        return len(ids) - len(tail)

    # ---- the generation, as events
    def run(self, ids, max_new, sampling, cancel, stop_strings):
        """Yields ("text", str), ("heartbeat", None), ("done", stats) - stop strings applied."""
        dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
        tok = self.model.tokenizer
        eos = set(self.model.eos_ids)
        emitted = ""
        held = ""
        stats = None
        stop_hit = False
        q = self.engine.queue
        ticket, turn = q.join(), False
        try:
            while not (turn := q.wait(ticket, 1.0)):          # another request is running: wait, with heartbeats
                if cancel.is_set():
                    return
                yield ("heartbeat", None)
        finally:
            if not turn:                                      # cancelled, or the client left (GeneratorExit)
                q.leave(ticket)
        try:
            for kind, val in self.engine.generate(ids, max_new, sampling, cancel):
                if kind == "token":
                    if val in eos:
                        continue
                    piece = dec.decode(tok.token_bytes(val))
                    if not piece or stop_hit:
                        continue
                    held += piece
                    if stop_strings:
                        cut = min((held.find(s) for s in stop_strings if s in held), default=-1)
                        if cut >= 0:
                            held = held[:cut]
                            stop_hit = True
                            cancel.set()
                        keep = 0 if stop_hit else max((n for s in stop_strings
                                                       for n in range(min(len(s) - 1, len(held)), 0, -1)
                                                       if held.endswith(s[:n])), default=0)
                    else:
                        keep = 0
                    out, held = held[:len(held) - keep], held[len(held) - keep:]
                    if out:
                        emitted += out
                        yield ("text", out)
                elif kind in ("progress", "heartbeat"):
                    yield ("heartbeat", None)
                elif kind == "done":
                    stats = val
            tail = dec.decode(b"", final=True)
            if not stop_hit and (held or tail):
                yield ("text", held + tail)
        finally:
            q.release()
        if stop_hit and stats is not None:
            stats["finish"] = "stop"
        yield ("done", stats)

    def max_new(self, req: dict, n_prompt: int) -> int:
        room = self.engine.max_ctx - n_prompt - 1
        if room <= 0:
            raise BadRequest(f"the prompt is {n_prompt} tokens; the context is {self.engine.max_ctx}")
        m = req.get("max_completion_tokens", req.get("max_tokens"))
        if m is None:
            return room
        if isinstance(m, bool) or not isinstance(m, int) or m < 0:
            raise BadRequest("'max_tokens' must be a non-negative integer")
        if m > room:
            raise BadRequest(f"the prompt ({n_prompt} tokens) plus max_tokens ({m}) exceeds the context "
                             f"({self.engine.max_ctx})")
        return m


def _finish_reason(stats, had_tool_call: bool) -> str:
    if had_tool_call:
        return "tool_calls"
    f = (stats or {}).get("finish")
    return "length" if f == "length" else "stop"


# ================================================================ HTTP

def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "strata-glm"

        def log_message(self, fmt, *args):
            sys.stderr.write("[glm-server] " + (fmt % args) + "\n")

        # ---- plumbing
        def _cors(self):
            if svc.cors:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, code: int, obj):
            body = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)

        def _error(self, code: int, msg: str, typ: str = "invalid_request_error"):
            self._json(code, {"error": {"message": msg, "type": typ}})

        def _authorized(self) -> bool:
            if not svc.api_key:
                return True
            auth = self.headers.get("Authorization", "")
            key = auth[7:] if auth.startswith("Bearer ") else self.headers.get("x-api-key", "")
            return key in svc.api_key.split(",")

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > 64 * 1024 * 1024:
                raise BadRequest("the request needs a JSON body (at most 64 MiB)")
            try:
                obj = json.loads(self.rfile.read(n))
            except ValueError:
                raise BadRequest("the body is not valid JSON") from None
            if not isinstance(obj, dict):
                raise BadRequest("the body must be a JSON object")
            return obj

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):
            if self.path in ("/health", "/v1/health"):
                ok = svc.engine.alive()
                return self._json(200 if ok else 503, {"status": "ok" if ok else "engine stopped",
                                                       "context": svc.engine.max_ctx, **svc.engine.info})
            if not self._authorized():
                return self._error(401, "invalid API key", "authentication_error")
            if self.path in ("/v1/models", "/models"):
                return self._json(200, {"object": "list", "data": [{
                    "id": svc.model.name, "object": "model", "created": int(time.time()), "owned_by": "strata",
                    "context_length": svc.engine.max_ctx}]})
            self._error(404, f"no endpoint {self.path}")

        def do_POST(self):
            if not self._authorized():
                return self._error(401, "invalid API key", "authentication_error")
            try:
                req = self._body()
                if self.path in ("/v1/chat/completions", "/chat/completions"):
                    return self._chat(req)
                if self.path in ("/v1/completions", "/completions"):
                    return self._completion(req)
                return self._error(404, f"no endpoint {self.path}")
            except BadRequest as e:
                return self._error(400, str(e))
            except EngineDied as e:
                return self._error(503, str(e), "server_error")
            except ValueError as e:
                return self._error(400, str(e))

        # ---- SSE
        def _sse_start(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self._cors()
            self.end_headers()
            self.close_connection = True

        def _sse(self, obj) -> bool:
            try:
                self.wfile.write(b"data: " + json.dumps(obj, ensure_ascii=False).encode() + b"\n\n")
                self.wfile.flush()
                return True
            except OSError:
                return False

        def _keepalive(self) -> bool:
            try:
                self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
                return True
            except OSError:
                return False

        # ---- chat
        def _chat(self, req):
            text, ids, tools, cp = svc.chat_prompt(req)
            max_new = svc.max_new(req, len(ids))
            sampling = _sampling(req)
            sampling["cp"] = cp                      # an engine key, not sampling: where to checkpoint
            stops = req.get("stop") or []
            stops = [stops] if isinstance(stops, str) else [s for s in stops if isinstance(s, str) and s]
            parser = GlmParser(text.rstrip().endswith(THINK_OPEN), tools)
            cancel = threading.Event()
            cid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            stream = bool(req.get("stream"))
            usage_in_stream = bool((req.get("stream_options") or {}).get("include_usage"))

            def chunk(delta, finish=None):
                return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": svc.model.name,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

            content, reasoning, calls = [], [], []
            stats = None
            if stream:
                self._sse_start()
                if not self._sse(chunk({"role": "assistant", "content": ""})):
                    return
            last_beat = time.monotonic()
            try:
                for kind, val in svc.run(ids, max_new, sampling, cancel, stops):
                    if kind == "done":
                        stats = val
                        events = parser.feed("", final=True)
                    elif kind == "text":
                        events = parser.feed(val)
                    else:
                        events = []
                        if stream and time.monotonic() - last_beat >= KEEPALIVE_S:
                            last_beat = time.monotonic()
                            if not self._keepalive():
                                cancel.set()
                    for ev, payload in events:
                        if ev == "reasoning":
                            reasoning.append(payload)
                            delta = {"reasoning_content": payload}
                        elif ev == "content":
                            content.append(payload)
                            delta = {"content": payload}
                        else:
                            call = {"index": len(calls), "id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                                    "function": payload}
                            calls.append(call)
                            delta = {"tool_calls": [call]}
                        if stream and not self._sse(chunk(delta)):
                            cancel.set()
            except (EngineDied, ValueError) as e:
                if not stream:
                    raise
                self._sse({"error": {"message": str(e), "type": "server_error"}})
                return
            finish = _finish_reason(stats, bool(calls))
            usage = {"prompt_tokens": len(ids), "completion_tokens": (stats or {}).get("generated", 0),
                     "total_tokens": len(ids) + (stats or {}).get("generated", 0)}
            if stream:
                self._sse(chunk({}, finish))
                if usage_in_stream:
                    self._sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                               "model": svc.model.name, "choices": [], "usage": usage})
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    pass
                return
            msg = {"role": "assistant", "content": "".join(content) or (None if calls else "")}
            if reasoning:
                msg["reasoning_content"] = "".join(reasoning)
            if calls:
                msg["tool_calls"] = [{k: v for k, v in c.items() if k != "index"} for c in calls]
            self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": svc.model.name,
                             "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage,
                             "timings": stats})

        # ---- plain completion
        def _completion(self, req):
            prompt = req.get("prompt")
            if not isinstance(prompt, str):
                raise BadRequest("'prompt' must be a string")
            ids = svc.model.tokenizer.encode(prompt, parse_special=True)
            max_new = svc.max_new(req, len(ids)) if req.get("max_tokens") is not None else min(
                256, svc.engine.max_ctx - len(ids) - 1)
            sampling = _sampling(req)
            stops = req.get("stop") or []
            stops = [stops] if isinstance(stops, str) else [s for s in stops if isinstance(s, str) and s]
            cancel = threading.Event()
            cid = "cmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            stream = bool(req.get("stream"))
            out, stats = [], None
            if stream:
                self._sse_start()
            for kind, val in svc.run(ids, max_new, sampling, cancel, stops):
                if kind == "text":
                    out.append(val)
                    if stream and not self._sse({"id": cid, "object": "text_completion", "created": created,
                                                 "model": svc.model.name,
                                                 "choices": [{"index": 0, "text": val, "finish_reason": None}]}):
                        cancel.set()
                elif kind == "done":
                    stats = val
            finish = "length" if (stats or {}).get("finish") == "length" else "stop"
            usage = {"prompt_tokens": len(ids), "completion_tokens": (stats or {}).get("generated", 0),
                     "total_tokens": len(ids) + (stats or {}).get("generated", 0)}
            if stream:
                self._sse({"id": cid, "object": "text_completion", "created": created, "model": svc.model.name,
                           "choices": [{"index": 0, "text": "", "finish_reason": finish}]})
                try:
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except OSError:
                    pass
                return
            self._json(200, {"id": cid, "object": "text_completion", "created": created, "model": svc.model.name,
                             "choices": [{"index": 0, "text": "".join(out), "finish_reason": finish}],
                             "usage": usage})

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the GGUF (its first shard)")
    ap.add_argument("--engine", default=str(ROOT / "build" / "strata-glm"), help="strata-glm (or strata-glm-ref)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--ctx", type=int, default=32768, help="context length in tokens")
    ap.add_argument("--api-key", default=os.environ.get("STRATA_API_KEY"),
                    help="required as 'Authorization: Bearer KEY'; several keys separated by commas")
    ap.add_argument("--cors", action="store_true", help="allow browser apps on other origins (Open WebUI direct)")
    ap.add_argument("--log", default=None, help="the engine's stderr (default: inherit nothing)")
    ap.add_argument("engine_args", nargs=argparse.REMAINDER, help="after --: passed to the engine")
    a = ap.parse_args(argv)
    if a.host not in ("127.0.0.1", "localhost", "::1") and not a.api_key:
        ap.error("listening beyond this machine needs --api-key (or STRATA_API_KEY)")

    print(f"[glm-server] reading the tokenizer and template from {a.model}", flush=True)
    model = ModelInfo(a.model)
    extra = [x for x in a.engine_args if x != "--"]
    cmd = [a.engine, "-m", a.model, "--serve", "--ctx", str(a.ctx),
           "--eos-ids", ",".join(str(i) for i in model.eos_ids), *extra]
    print(f"[glm-server] starting the engine: {' '.join(cmd)}", flush=True)
    engine = Engine(cmd, a.log)
    print(f"[glm-server] engine ready ({engine.info.get('backend', '?')}), context {engine.max_ctx}", flush=True)
    svc = Service(model, engine, a.api_key, a.cors)
    httpd = ThreadingHTTPServer((a.host, a.port), make_handler(svc))
    httpd.daemon_threads = True
    print(f"[glm-server] OpenAI-compatible API on http://{a.host}:{a.port}/v1 (model '{model.name}')", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
