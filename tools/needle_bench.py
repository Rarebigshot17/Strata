"""tools/needle_bench.py - needle-in-a-haystack recall through a running Strata server, with the engine's speeds.

A long text is built from files in this repository (its docs and source code, and the llama.cpp docs setup downloads
into third_party/), a code word is hidden in it at a chosen depth, and the model is asked for it through the normal
API (thinking off, greedy). One line per test: length, depth, found or not, prompt tokens, the engine's prompt and
decode speeds (the answer's `timings`), time. The code word counts as found in the answer or in its reasoning: it is
random, so it can only come from the text. With thinking off the model can still reason in the answer first at long
lengths, so the answer may run to --answer-tokens (300); a line whose answer was cut there says so.

    python tools/needle_bench.py                                  # 32K at depths 10/50/90, http://127.0.0.1:8101
    python tools/needle_bench.py --lengths 32k,128k --depths 10,50,90 --url http://127.0.0.1:8102
    python tools/needle_bench.py --lengths 8k,256k --decode-tokens 256 --out needles.json --api-key KEY

The lengths are targets, 2% under so that "128k" fits a 128K context. With --gguf (the model's first shard) the text
is cut to that many tokens by the model's own tokenizer (serve/'s, tools/strata_tokenizer.py; run the tool with the
server's Python, .venv/bin/python, which has its `regex` module). Without it the text is cut by characters, at about
3.2 per token to start with and then at the rate each answer's real token count shows (the first length can come out
10% or more short), and the line reports the real count.
A prompt the server refuses as too long for its context (it says how many tokens it has, before reading any) is cut
to fit and sent again. A length the server's context cannot hold is skipped.

--decode-tokens N adds one more turn after each length's last depth: the conversation so far (the text, the question,
the model's answer) and a request for a summary, N tokens long, to measure decoding with the whole text in the
context; --decode-turns K makes that K turns, each asking to continue the last (decoding as the VRAM expert cache
adapts to the new text). Each should find the conversation in the server's cache; when the cache gave back less than
90% of the prompt, the rest of the run leaves these turns out (they would read the whole text again).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [("docs", "*.md"), ("src", "*.cpp"), ("src", "*.cu"), ("include", "*.hpp"), ("serve", "*.py"), ("tools", "*.py"),
           ("third_party/llama.cpp/docs", "*.md")]
WORDS = ["amber", "falcon", "quartz", "willow", "copper", "harbor", "saffron", "glacier", "orchid", "lantern",
         "meadow", "cobalt", "juniper", "tundra", "velvet", "ember"]
CHARS_PER_TOKEN = 3.2
QUESTION = "\n\nWhat is the secret code word mentioned in the text above? Reply with the code word only."
SUMMARY = "Now summarize the text above in about 400 words."
MORE = "Continue the summary."
TOO_LONG = re.compile(r"exceeds the context window: prompt \((\d+) tokens\)")


def haystack(n_chars: int) -> str:
    """Deterministic filler: the repository's text files in a fixed order, repeated if needed."""
    files = []
    for d, pat in SOURCES:
        files += sorted((ROOT / d).rglob(pat))
    parts, total = [], 0
    while total < n_chars:
        for f in files:
            try:
                t = f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            parts.append(f"\n\n=== {f.relative_to(ROOT).as_posix()} ===\n{t}")
            total += len(parts[-1])
            if total >= n_chars:
                break
        if not files:
            raise SystemExit("no text files found to build the haystack from")
    return "".join(parts)[:n_chars]


def ask(url: str, key: str, messages: list[dict], max_tokens: int, timeout: float) -> tuple[dict, float]:
    """One chat request -> (the answer's JSON, seconds).  urllib.error.HTTPError carries a refusal's body."""
    body = {"model": "strata", "max_tokens": max_tokens, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False}, "messages": messages}
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    t0 = time.time()
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return out, time.time() - t0


def refused_tokens(e: urllib.error.HTTPError) -> int | None:
    """The prompt's token count from a "too long for the context" refusal, else None."""
    try:
        m = TOO_LONG.search(e.read().decode("utf-8", "replace"))
    except OSError:
        return None
    return int(m.group(1)) if m else None


def speeds(out: dict) -> dict:
    """The answer's prompt tokens, how many of them the cache held, and the engine's own prompt and decode rates."""
    u, t = out.get("usage") or {}, out.get("timings") or {}
    return {"prompt_tokens": u.get("prompt_tokens"),
            "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"), "read": t.get("prompt_n"),
            "prompt_ms": t.get("prompt_ms"), "prompt_tok_s": t.get("prompt_per_second"),
            "decoded": t.get("predicted_n"), "decode_tok_s": t.get("predicted_per_second")}


def rate(s: dict) -> str:
    def f(v, unit):
        return f"{v:,.1f} {unit}" if isinstance(v, (int, float)) else "-"
    return (f"read {s['read'] if s['read'] is not None else '-'} at {f(s['prompt_tok_s'], 'tok/s')}, "
            f"decode {f(s['decode_tok_s'], 'tok/s')}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8101")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--lengths", default="32k", help="comma-separated, e.g. 1k,32k,128k,256k,512k")
    ap.add_argument("--depths", default="10,50,90", help="where the code word sits, percent of the text")
    ap.add_argument("--answer-tokens", type=int, default=300, help="the longest answer to the question")
    ap.add_argument("--decode-tokens", type=int, default=0, help="a summary turn of this many tokens after each length")
    ap.add_argument("--decode-turns", type=int, default=1, help="how many summary turns, each continuing the last")
    ap.add_argument("--gguf", help="the model's first shard: size each text by the model's tokenizer")
    ap.add_argument("--ids-out", help="with --gguf and one length: write the text's token ids there (for strata-glm "
                                      "--tokens-file) and stop, no server needed")
    ap.add_argument("--timeout", type=float, default=6 * 3600, help="seconds to wait for one answer")
    ap.add_argument("--out", help="also write the results as JSON")
    a = ap.parse_args()
    if a.ids_out:
        if not a.gguf or "," in a.lengths:
            ap.error("--ids-out takes --gguf and one length")
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from strata_tokenizer import Tokenizer
        tok = Tokenizer.from_gguf(a.gguf)
        L = a.lengths
        n = int(float(L.lower().rstrip("k")) * 1024) if L.lower().endswith("k") else int(L)
        ids = tok.encode(haystack(int(n * 4.5)))       # more than enough text: no text runs above 4.5 chars a token
        if len(ids) < n:
            ids = tok.encode(haystack(int(n * 9)))
        Path(a.ids_out).write_text(" ".join(str(i) for i in ids[:n]) + "\n", encoding="utf-8")
        print(f"wrote {min(n, len(ids)):,} token ids to {a.ids_out}")
        return 0
    try:
        req = urllib.request.Request(a.url.rstrip("/") + "/metrics",
                                     headers={"Authorization": "Bearer " + a.api_key} if a.api_key else {})
        with urllib.request.urlopen(req, timeout=10) as r:
            ctx = int((json.loads(r.read()).get("engine") or {}).get("max_context") or 0)
    except (OSError, ValueError):
        ctx = 0
    tok = None
    if a.gguf:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from strata_tokenizer import Tokenizer
        tok = Tokenizer.from_gguf(a.gguf)
    rnd = random.Random(7)
    cpt = CHARS_PER_TOKEN
    decode_turn = a.decode_tokens > 0
    rows, found = [], 0
    for L in a.lengths.split(","):
        tokens = int(float(L.lower().rstrip("k")) * 1024) if L.lower().endswith("k") else int(L)
        tokens = int(tokens * 0.98)                        # "128k" fits a 128K context with room to answer
        if ctx and tokens + 200 > ctx:
            print(f"{L:>5}: skipped (the server's context is {ctx})")
            continue
        if tok is not None:                                # the rate of this length's own text, to 0.5%
            for _ in range(5):
                text = haystack(int(tokens * cpt))
                n = len(tok.encode(text + QUESTION))
                cpt *= tokens / n
                if abs(n - tokens) <= 0.005 * tokens:
                    break
            print(f"{L:>5}: {n:,} tokens of text by the model's tokenizer", flush=True)
        depths = [int(x) for x in a.depths.split(",")]
        for k, d in enumerate(depths):
            word = f"{rnd.choice(WORDS)}-{rnd.choice(WORDS)}-{rnd.randint(100, 999)}"
            needle = f"\nThe secret code word for this text is: {word}. Remember it.\n"
            for attempt in range(3):
                text = haystack(int(tokens * cpt))
                cut = int(len(text) * d / 100)
                cut = text.rfind("\n", 0, cut) + 1 or cut      # at a line start
                prompt = text[:cut] + needle + text[cut:] + QUESTION
                messages = [{"role": "user", "content": prompt}]
                try:
                    out, secs = ask(a.url, a.api_key, messages, a.answer_tokens, a.timeout)
                    err = None
                except urllib.error.HTTPError as e:
                    n = refused_tokens(e) if e.code == 400 else None
                    if n and attempt < 2:                        # too long: cut to the rate it showed, and again
                        cpt = len(prompt) / n * 0.995
                        print(f"{L:>5} depth {d:>3}%: {n:,} tokens is too long for the context, cut to fit", flush=True)
                        continue
                    err = e
                except OSError as e:
                    err = e
                break
            if err is not None:
                print(f"{L:>5} depth {d:>3}%: request failed ({err})", flush=True)
                rows.append({"length": L, "depth": d, "error": str(err)})
                continue
            msg, finish = out["choices"][0]["message"], out["choices"][0].get("finish_reason")
            answer, reasoning = msg.get("content") or "", msg.get("reasoning_content") or ""
            s = speeds(out)
            if s["prompt_tokens"] and tok is None:
                cpt = len(prompt) / s["prompt_tokens"]          # the next length is cut at this text's real rate
            ok = word in answer or word in reasoning
            found += ok
            cut_off = " (the answer was cut at --answer-tokens)" if finish == "length" else ""
            print(f"{L:>5} depth {d:>3}%: {'FOUND' if ok else 'missed'}  ({s['prompt_tokens'] or 0:,} prompt tokens, "
                  f"{rate(s)}, {secs:.0f} s)" + ("" if ok else f"{cut_off}  answer ends: {answer.strip()[-80:]!r}"),
                  flush=True)
            rows.append({"length": L, "depth": d, **s, "seconds": round(secs, 1), "found": ok, "finish": finish,
                         "word": word, "answer": answer.strip()[:600], "reasoning": reasoning.strip()[:600]})
            if decode_turn and k == len(depths) - 1:
                messages.append({"role": "assistant", "content": answer})
                for turn in range(1, a.decode_turns + 1):
                    messages.append({"role": "user", "content": SUMMARY if turn == 1 else MORE})
                    try:
                        out2, secs2 = ask(a.url, a.api_key, messages, a.decode_tokens, a.timeout)
                    except OSError as e:
                        print(f"{L:>5} summary turn {turn}: request failed ({e})", flush=True)
                        rows.append({"length": L, "turn": turn, "error": str(e)})
                        break
                    s2 = speeds(out2)
                    print(f"{L:>5} summary turn {turn}: {s2['prompt_tokens'] or 0:,} prompt tokens, "
                          f"{s2['cached'] or 0:,} from the cache, {rate(s2)}, {secs2:.0f} s", flush=True)
                    rows.append({"length": L, "turn": turn, **s2, "seconds": round(secs2, 1)})
                    if not s2["prompt_tokens"] or (s2["cached"] or 0) < 0.9 * s2["prompt_tokens"]:
                        decode_turn = False
                        print("      the cache did not hold the conversation: no more summary turns", flush=True)
                        break
                    reply = out2["choices"][0]["message"].get("content") or ""
                    messages.append({"role": "assistant", "content": reply})
    done = [r for r in rows if "found" in r]
    print(f"{found} of {len(done)} found")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0 if done and found == len(done) else 1


if __name__ == "__main__":
    sys.exit(main())
