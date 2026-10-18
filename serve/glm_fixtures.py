"""serve/glm_fixtures.py - shared pieces for the serve tests on GLM-5.3-Flash's format (no model, no GPU).

    from serve.glm_fixtures import GlmTokenizer, glm_template, call_text, reply_text, free_port

GlmTokenizer is server.py's byte tokenizer with GLM's reasoning and tool tags as type-4 tokens (matched always,
as the real GGUF's tags are assumed to be), so a MockEngine script can hold them as one token each. glm_template()
is the fallback template the server ships (GLM-4.7-Flash's format). call_text() and reply_text() write a reply
the way GLM writes it, for MockEngine scripts.
"""
from __future__ import annotations

import json
import socket
from pathlib import Path

from serve.frontend import (CALL_END, CALL_START, KEY_END, KEY_START, RESPONSE_END, RESPONSE_START, THINK_END,
                            THINK_START, VALUE_END, VALUE_START, ChatTemplate)
from serve.server import ByteTokenizer

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


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
