#!/usr/bin/env python3
"""Ban one token through the HTTP API of a running Strata server and check that the ban works and ends with its request.

    python tools/logit_bias_english_example.py --gguf /path/to/the/first/shard.gguf [--url http://127.0.0.1:8101]

The same greedy request ("one English greeting, one word only", thinking off) is sent four times: as it is, with
`logit_bias` banning the word's token as an object ({"<id>": -100}), the same ban as a pair list ([[<id>, false]]),
and as it is again.  The token id comes from the model's own tokenizer (tools/strata_tokenizer.py reads it from the
GGUF; docs/LOGIT_BIAS.md).  It passes when the first answer has the word, both bans give one and the same other
answer, and the last answer is the first one again (the ban belongs to its request only).  A ban removes one token,
not the word: the model may still spell it with other tokens, which is reported but not a failure.
--output DIR writes every request and answer to DIR/rows.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strata_tokenizer import Tokenizer  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="the model's GGUF (its first shard): the tokenizer")
    ap.add_argument("--url", default="http://127.0.0.1:8101")
    ap.add_argument("--api-key", default=os.environ.get("STRATA_KEY", ""))
    ap.add_argument("--word", default="Hello", help="the word to ban: it must be one token of the model")
    ap.add_argument("--output", type=Path, help="write every request and answer here (rows.json)")
    a = ap.parse_args(argv)

    tok = Tokenizer.from_gguf(a.gguf)
    ids = tok.encode(a.word)
    print(f"{a.word!r} is token(s) {ids}", flush=True)
    if len(ids) != 1:
        print(f"{a.word!r} is not one token of this model: pick another word with --word", file=sys.stderr)
        return 2
    banned = ids[0]
    base = {"model": "strata", "max_tokens": 8, "temperature": 0, "reasoning_effort": "none",
            "messages": [{"role": "user", "content": "Reply with one English greeting, one word only."}]}
    headers = {"Content-Type": "application/json", **({"Authorization": "Bearer " + a.api_key} if a.api_key else {})}
    rows = []

    def call(label, **extra):
        body = dict(base, **extra)
        req = urllib.request.Request(a.url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.loads(r.read())
        text = out["choices"][0]["message"].get("content") or ""
        rows.append({"label": label, "request": body, "response": out})
        print(f"{label:12s} {text!r}", flush=True)
        return text

    try:
        before = call("before")
        obj = call("ban-object", logit_bias={str(banned): -100})
        pairs = call("ban-pairs", logit_bias=[[banned, False]])
        cleared = call("cleared")
    except urllib.error.HTTPError as e:
        print(f"the server refused request {len(rows) + 1}: HTTP {e.code} {e.read()[:300].decode('utf-8', 'replace')}",
              file=sys.stderr)
        return 1
    if a.output:
        a.output.mkdir(parents=True, exist_ok=True)
        (a.output / "rows.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    if a.word not in before:
        print(f"the answer without a ban does not have {a.word!r}: pick a word the model answers with", file=sys.stderr)
        return 2
    checks = {"the ban changes the answer": obj != before,
              "the object and the pair list ban the same": obj == pairs,
              "the next request is not biased": cleared == before}
    for what, ok in checks.items():
        print(f"{'ok ' if ok else 'BAD'} {what}")
    if a.word in obj:
        print(f"note: the banned answer still spells {a.word!r} with other tokens")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
