#!/usr/bin/env python3
"""A client that stops reading a streamed answer early must not leave its tokens to the next request.

  STRATA_API_KEY=... python3 tools/early_close_test.py [http://127.0.0.1:8101]
"""
import json, os, sys, threading, urllib.request

URL = (sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8101") + "/v1/chat/completions"
KEY = os.environ.get("STRATA_API_KEY") or os.environ.get("STRATA_KEY", "")     # the server's name, or the old one


def req(content, stream, max_tokens=300):
    body = json.dumps({"model": "m", "max_tokens": max_tokens, "temperature": 0, "stream": stream,
                       "messages": [{"role": "user", "content": content}],
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    return urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json",
                                                           "Authorization": f"Bearer {KEY}"})


def early_close(content, keep_lines=5):
    r = urllib.request.urlopen(req(content, True), timeout=600)
    for _ in range(keep_lines):
        r.readline()
    r.close()                                   # the client goes away mid-answer


def answer(content):
    with urllib.request.urlopen(req(content, False, 60), timeout=600) as r:
        m = json.load(r)["choices"][0]["message"]
        return (m.get("content") or "") + " " + (m.get("reasoning_content") or m.get("reasoning") or "")


ok = True
# alone, then the same with a second request sent just before it (the server answers one request at a time: the
# early close happens while a request is running or waiting its turn)
for label, background in (("solo", False), ("busy", True)):
    bg = None
    if background:
        bg = threading.Thread(target=answer, args=("Count slowly from 1 to 60, one number per line.",))
        bg.start()
    early_close("Describe the weather in Montpellier in great detail, at length.")
    a = answer("Reply with exactly one word: the color of the sky on a clear day.")
    good = "blue" in a.lower() and "montpellier" not in a.lower()
    ok &= good
    print(f"{label}: {'OK' if good else 'STALE'} -> {a[:120]!r}", flush=True)
    if bg:
        bg.join()
sys.exit(0 if ok else 1)
