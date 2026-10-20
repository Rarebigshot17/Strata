#!/usr/bin/env python3
"""What serve/server.py makes of a GLM GGUF's tokenizer and chat template: no GPU, no engine, nothing is changed.

    python tools/glm53_template_probe.py MODEL.gguf [--template-out FILE]

MODEL.gguf is the model's first shard (it holds the tokenizer and the template). Prints:
  1. GLM's tags and turn tokens: each one's id and token type (3 control, 4 user-defined; tool calls are read from the
     special tokens only when every tag is one), and the end-of-turn ids the engine stops at;
  2. the server's start line about where tool calls are read from;
  3. the template's whole source (also written to --template-out);
  4. the prompt the server renders for a few requests (thinking left to the model, off, low, medium, high and max; a
     tool, a call and its result; a forced call with thinking off; the same over the Anthropic API), as text, with
     whether it leaves the thinking open, its token count and the engine's checkpoint.
Paste the whole output back: the server is built for the template in the GGUF and this shows how it reads it.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from serve import server as S  # noqa: E402
from serve.frontend import OUTPUT_TAGS, anthropic_to_messages, forced_call, openai_to_messages  # noqa: E402

TURN_TOKENS = ("[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>", "<|endoftext|>",
               "<tool_response>", "</tool_response>")

TOOLS = [{"type": "function", "function": {
    "name": "read_file", "description": "Read a file",
    "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "lines": {"type": "integer"}},
                   "required": ["path"]}}}]
CHAT = [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}]
TOOL_TURNS = [{"role": "user", "content": "Show me notes.txt"},
              {"role": "assistant", "content": "", "reasoning_content": "I should read it.",
               "tool_calls": [{"id": "c1", "type": "function",
                               "function": {"name": "read_file", "arguments": '{"path": "notes.txt", "lines": 3}'}}]},
              {"role": "tool", "tool_call_id": "c1", "content": "milk\neggs\nbread"},
              {"role": "user", "content": "Thanks. Anything else?"}]
REQUESTS = [
    ("openai: thinking left to the model", "openai", {"messages": CHAT}),
    ("openai: reasoning_effort none (thinking off)", "openai", {"messages": CHAT, "reasoning_effort": "none"}),
    ("openai: reasoning_effort low", "openai", {"messages": CHAT, "reasoning_effort": "low"}),
    ("openai: reasoning_effort medium", "openai", {"messages": CHAT, "reasoning_effort": "medium"}),
    ("openai: reasoning_effort high", "openai", {"messages": CHAT, "reasoning_effort": "high"}),
    ("openai: reasoning_effort max", "openai", {"messages": CHAT, "reasoning_effort": "max"}),
    ("openai: a tool, a call with its reasoning, its result, the next question", "openai",
     {"messages": TOOL_TURNS, "tools": TOOLS}),
    ("openai: a forced call (tool_choice required) with thinking off", "openai",
     {"messages": TOOL_TURNS[:1], "tools": TOOLS, "tool_choice": "required", "reasoning_effort": "none"}),
    ("anthropic: thinking enabled with budget_tokens", "anthropic",
     {"max_tokens": 1024, "thinking": {"type": "enabled", "budget_tokens": 4096},
      "messages": [{"role": "user", "content": "What is 2+2?"}]}),
    ("anthropic: thinking disabled", "anthropic",
     {"max_tokens": 1024, "thinking": {"type": "disabled"}, "messages": [{"role": "user", "content": "What is 2+2?"}]}),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("gguf", help="the model's first shard")
    ap.add_argument("--template-out", help="write the template's source to this file")
    a = ap.parse_args()

    files = S.ModelFiles(a.gguf)
    tok = files.tokenizer
    template = S.chat_template_for(files, None)
    types = getattr(tok, "token_types", None) or []

    print("== 1. tags and turn tokens (type 3 control, 4 user-defined; '-' not a special token)")
    for t in OUTPUT_TAGS + TURN_TOKENS:
        i = tok.special_tokens.get(t)
        ty = types[i] if i is not None and i < len(types) else "-"
        print(f"  {t:20} id {i if i is not None else '-':>7}   type {ty}")
    print("  end-of-turn ids (the engine's --eos-ids):", S.end_of_turn_ids(tok),
          "=", [tok.decode([i]) for i in S.end_of_turn_ids(tok)])
    print("  model name:", files.name, "  pre-tokenizer:", getattr(tok, "pre", "?"))

    print("\n== 2. the server's start line")
    print(" ", S.tag_mode_line(tok))

    print("\n== 3. the template (from the GGUF)" if files.template else "\n== 3. the GGUF has no template: the fallback")
    source = files.template or (ROOT / "serve" / "chat_template.jinja").read_text(encoding="utf-8")
    print(source)
    if a.template_out:
        pathlib.Path(a.template_out).write_text(source, encoding="utf-8")
        print(f"(written to {a.template_out})")

    print("\n== 4. prompts the server renders")
    svc = S.Service(S.MockEngine(tok, "x"), tok, template)
    for title, api, req in REQUESTS:
        print(f"\n-- {title}")
        try:
            if api == "openai":
                messages, tools, kw = openai_to_messages(req)
            else:
                messages, tools, kw = anthropic_to_messages(S.anthropic_budget(req), svc.anthropic_think_unasked)
            force = forced_call(req.get("tool_choice"), tools)    # the opening of the call the reply must make
            ids, thinking, _ = svc.prepare(messages, tools, kw, max_new=64, force=force)
            cp = svc.checkpoint.at
            print(f"   template kwargs {kw}; thinking {thinking}; {len(ids)} tokens; checkpoint at {cp}")
            print(tok.decode(ids))
        except Exception as e:  # noqa: BLE001 - every request is shown, a failure with its reason
            print(f"   FAILED: {type(e).__name__}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
