"""serve/glm_fixtures.py - shared pieces for the serve tests on GLM-5.3-Flash's format (no model, no GPU).

    from serve.glm_fixtures import GlmTokenizer, glm_template, call_text, reply_text, free_port

GlmTokenizer is server.py's byte tokenizer with GLM's reasoning and tool tags as type-4 tokens (matched always,
as the real GGUF's tags are), so a MockEngine script can hold them as one token each. glm_template() is the
fallback template the server ships (GLM-5.3-Flash's own, from the real GGUF). call_text() and reply_text() write a reply
the way GLM writes it, for MockEngine scripts.  IdEngine's scripts can also hold plain() text: a tag's text as
ordinary tokens, the way a model writes it when it copies the tag from its context.
"""
from __future__ import annotations

import json
import socket
from pathlib import Path

from serve.frontend import (CALL_END, CALL_START, KEY_END, KEY_START, RESPONSE_END, RESPONSE_START, THINK_END,
                            THINK_START, VALUE_END, VALUE_START, ChatTemplate)
from serve.server import ByteTokenizer, MockEngine

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_PATH = ROOT / "serve" / "chat_template.jinja"

GLM_TAGS = (THINK_START, THINK_END, CALL_START, CALL_END, KEY_START, KEY_END, VALUE_START, VALUE_END,
            RESPONSE_START, RESPONSE_END)


class GlmTokenizer(ByteTokenizer):
    """One id per UTF-8 byte; GLM's control tokens (ids 256-262, matched with parse_special) and its tags (ids from
    263, matched always)."""
    SPECIALS = ByteTokenizer.SPECIALS + list(GLM_TAGS)
    ALWAYS = GLM_TAGS


def glm_template() -> ChatTemplate:
    """The chat template the server falls back to (serve/chat_template.jinja)."""
    return ChatTemplate(TEMPLATE_PATH)


def call_text(name: str, **args) -> str:
    """A tool call as GLM writes it: string values raw, every other value as JSON."""
    body = "".join(f"{KEY_START}{k}{KEY_END}{VALUE_START}{v if isinstance(v, str) else json.dumps(v)}{VALUE_END}"
                   for k, v in args.items())
    return f"{CALL_START}{name}{body}{CALL_END}"


def reply_text(reasoning: str | None, content: str = "", *calls: str) -> str:
    """A reply after the generation prompt: the reasoning and its </think> (the prompt opened <think>; None for a
    reply with thinking off, whose prompt closed it already), then the content and the calls."""
    head = "" if reasoning is None else reasoning + THINK_END
    return head + content + "".join(calls)


def plain(text: str) -> list[int]:
    """`text` as ordinary tokens only (one id per UTF-8 byte), never a special token: a tag's text as a model writes it
    when it copies it from a tool result, a web page or the user (the prompt holds it as ordinary text, #537)."""
    return list(text.encode("utf-8"))


class IdEngine(MockEngine):
    """MockEngine whose scripts are lists of pieces: text (encoded as MockEngine encodes it, GLM's tags as their
    special ids) and lists of ids (plain()), taken as they are.  `scripts`: one list per request, the last repeats."""

    def __init__(self, tokenizer, scripts: list[list], max_context: int = 32768):
        super().__init__(tokenizer, "", max_context)
        end = tokenizer.encode("<|user|>", parse_special=True)
        self.scripts = [[t for piece in script for t in (piece if isinstance(piece, list) else
                                                         tokenizer.encode(piece, parse_special=True))] + end
                        for script in scripts]
        self.script = self.scripts[0]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
