#!/usr/bin/env python3
"""The server's tokenizer (tools/strata_tokenizer.py) against llama.cpp's, on one GGUF's vocabulary.

    python tools/glm53_tokcheck.py MODEL.gguf LLAMA_TOKENIZE_BIN

Tokenizes each string of a corpus (Unicode, digits, code, whitespace runs, GLM's special tokens) both ways, with
special tokens parsed, and prints every disagreement. Exit 0 when all agree.
"""
from __future__ import annotations

import ast
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from gguf_reader import GGUFFile  # noqa: E402
from strata_tokenizer import Tokenizer  # noqa: E402

CORPUS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "numbers: 1 12 123 1234 12345 123456 3.14159 -42 1e10",
    "def main():\n    return x  # comment\n\n\tindented\ttabs",
    "   leading and trailing spaces   ",
    "line one\nline two\r\nline three\n\n\n",
    "it's they're we've I'm you'll he'd IT'S",
    "Unicode: café naïve Zürich São Paulo ÅÄÖ",
    "中文测试，日本語のテスト、한국어 테스트",
    "emoji 😀🚀👍🏽 and symbols ∑∫√∞ ≠ ≤",
    "combining: é ä ñ",
    "[gMASK]<sop><|system|>You are helpful.<|user|>Hi<|assistant|><think>reason</think>answer",
    "<tool_call>read_file<arg_key>path</arg_key><arg_value>/tmp/a.txt</arg_value></tool_call>",
    "<|observation|><tool_response>{\"ok\": true}</tool_response>",
    "text<|endoftext|>more",
    "a<think>b</think>c<tool_call>",
    "<|user|",
    "   \n  \n",
    "x" * 300,
    "https://example.com/path?q=1&r=two#frag",
    "{\"key\": [1, 2, {\"nested\": null}]}",
]


def llama_ids(binary: str, model: str, text: str) -> list[int]:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
        f.write(text)
        path = f.name
    try:
        out = subprocess.run([binary, "-m", model, "-f", path, "--ids", "--log-disable", "--no-escape"],
                             capture_output=True, text=True, check=True).stdout
    finally:
        pathlib.Path(path).unlink()
    line = [l for l in out.splitlines() if l.startswith("[")][-1]
    return ast.literal_eval(line)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    model, binary = sys.argv[1], sys.argv[2]
    md = GGUFFile(pathlib.Path(model)).metadata
    tok = Tokenizer(list(md["tokenizer.ggml.tokens"]), list(md.get("tokenizer.ggml.merges") or []),
                    list(md.get("tokenizer.ggml.token_type") or []) or None, md.get("tokenizer.ggml.pre", "glm5"))
    bad = 0
    for text in CORPUS:
        mine = tok.encode(text, parse_special=True)
        ref = llama_ids(binary, model, text)
        if mine != ref:
            bad += 1
            print(f"DIFFERS {text[:60]!r}\n  strata:    {mine[:40]}\n  llama.cpp: {ref[:40]}")
        elif tok.decode(mine) != text:
            bad += 1
            print(f"ROUND TRIP {text[:60]!r} -> {tok.decode(mine)[:60]!r}")
    print(f"{len(CORPUS) - bad}/{len(CORPUS)} strings agree ({md.get('tokenizer.ggml.pre')}, "
          f"{len(md['tokenizer.ggml.tokens'])} tokens)")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
