"""serve/frontend.py - plan v0.3 P8: the request/response layer between API clients and the engine.

Everything here is text: no model, no GPU. The engine consumes token ids and produces text deltas; this module

  * normalizes OpenAI Chat Completions and Anthropic Messages requests into the chat template's message form,
  * renders the model's own Jinja chat template (GLM's, from the GGUF; `serve/chat_template.jinja` is GLM-4.7-Flash's,
    the same format) exactly as Hugging Face does,
  * parses the streamed output incrementally into reasoning (`<think>...</think>`), content and tool calls in
    GLM's form (`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`), never emitting a
    partial tag to the client.

Design note: plan v0.3 describes a single C++ process. The frontend is Python first because the template is Jinja
and the formats change often; the engine boundary is token ids in, text deltas out, which keeps a later C++ port
(or a pybind wrapper) a local change.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment

LOGGER = logging.getLogger(__name__)

# GLM's tags (special tokens of its tokenizer): the reasoning, a tool call and its arguments, a tool's result
THINK_START = "<think>"
THINK_END = "</think>"
CALL_START = "<tool_call>"
CALL_END = "</tool_call>"
KEY_START = "<arg_key>"
KEY_END = "</arg_key>"
VALUE_START = "<arg_value>"
VALUE_END = "</arg_value>"
RESPONSE_START = "<tool_response>"
RESPONSE_END = "</tool_response>"


# ------------------------------------------------------------------------------------------------ template
class TemplateRequestError(jinja2.exceptions.TemplateError, ValueError):
    """The template refused the request's messages (e.g. "No user query found in messages."): a ValueError, so the
    client gets a 400 with the template's message instead of a dropped connection (#365)."""


class ChatTemplate:
    """The model's chat template, rendered with the same Jinja settings as transformers' apply_chat_template.
    `path`: a template file; `source`: the template's text (the GGUF's tokenizer.chat_template)."""

    def __init__(self, path: str | Path | None = None, *, source: str | None = None):
        def raise_exception(message):
            raise TemplateRequestError(message)

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        if source is None:
            if path is None:
                raise ValueError("ChatTemplate needs a template file or the template's text")
            source = Path(path).read_text(encoding="utf-8")
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        # transformers gives templates the date this way (a system prompt that states today's date)
        env.globals["strftime_now"] = lambda fmt: time.strftime(fmt)
        self.source = source
        self.template = env.from_string(self.source)
        self.caps = self._detect_caps()

    def render(self, messages: list[dict], tools: list[dict] | None = None, add_generation_prompt: bool = True,
               **kwargs) -> str:
        # Do not use completed empty assistant turns as examples for the next reply. Keep the final message.
        # (#843.  STRATA_KEEP_EMPTY_TURNS=1 renders them as before.)
        messages = [m for i, m in enumerate(messages)
                    if i == len(messages) - 1 or os.environ.get("STRATA_KEEP_EMPTY_TURNS") == "1" or not (isinstance(m, dict) and m.get("role") == "assistant"
                                                      and not _text_of(m.get("content")).strip()
                                                      and not _has_image(m.get("content")) and not m.get("tool_calls"))]
        return self.template.render(messages=messages, tools=tools, add_generation_prompt=add_generation_prompt,
                                    **kwargs)

    def _detect_caps(self) -> dict[str, bool]:
        """llama.cpp's capability names, checked at load time against this template and GLM's tool-call format.
        These are rendering hints, not a guarantee that the model will follow a request."""
        def render(messages, tools=None, **kwargs):
            try:
                return self.render(messages, tools=tools, **kwargs)
            except Exception as exc:                # noqa: BLE001 - a hint only: never stop the server for a probe
                # A template may reject a role or feature (a custom one may fail in any way): the feature is off,
                # discovery still answers for the others, and the start-up goes on.
                LOGGER.debug("chat template capability probe failed: %s", exc, exc_info=True)
                return ""

        user = {"role": "user", "content": "strata_caps_user"}
        tools = [{"name": f"strata_caps_call_{i}", "description": "strata_caps_description",
                  "parameters": {"type": "object", "properties": {"arg": {"type": "string"}}}}
                 for i in range(2)]
        tool_prompt = render([user], tools)

        def calls_supported(count):
            calls = [{"name": f"strata_caps_call_{i}", "arguments": {"arg": f"strata_caps_arg_{i}"}}
                     for i in range(count)]
            messages = [user, {"role": "assistant", "content": "",
                               "tool_calls": [{"function": call} for call in calls]}]
            replies = [f"strata_caps_result_{i}" for i in range(count)]
            messages += [{"role": "tool", "content": reply} for reply in replies] + [user]
            prompt = render(messages, tools)
            # The instructions show the call format with placeholder names, which is not model output. Read only
            # complete calls to our probe functions, the way OutputParser reads a call.
            bodies = re.findall(r"<tool_call>(\s*strata_caps_call_\d+.*?)</tool_call>", prompt, re.S)
            try:
                parsed = [parse_tool_call(body, tools) for body in bodies]
            except ValueError:
                return False
            return [{"name": call.name, "arguments": call.arguments} for call in parsed] == calls \
                and all(reply in prompt for reply in replies)

        # GLM's template keeps the reasoning of turns before the last user message only when clear_thinking is
        # false: a template that keeps it by default or when asked to that way preserves reasoning
        history = [user, {"role": "assistant", "content": "strata_caps_answer",
                          "reasoning_content": "strata_caps_reasoning"}, user]
        return {"supports_tools": all(s in tool_prompt for s in
                                      ("strata_caps_call_0", "strata_caps_description", CALL_START, KEY_START)),
                "supports_tool_calls": calls_supported(1),
                "supports_system_role": "strata_caps_system" in render(
                    [{"role": "system", "content": "strata_caps_system"}, user]),
                "supports_parallel_tool_calls": calls_supported(2),
                "supports_preserve_reasoning": any("strata_caps_reasoning" in render(history, **kwargs)
                                                   for kwargs in ({}, {"clear_thinking": False}))}


# ------------------------------------------------------------------------------------------------ requests
# The content-part types the readers take as text (a part with no type that has a "text" field reads as text too)
TEXT_PARTS = ("text", "input_text", None)


def _text_of(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") in
                   TEXT_PARTS)


IMAGE_PARTS = ("image_url", "input_image", "image")

# Thinking levels.  GLM's template switches thinking on or off (enable_thinking=false for none); it has no levels.
# low, medium and high still reach it as reasoning_effort, with thinking on, for a template that reads them.  Clients
# spell these many ways; everything maps onto those four.
EFFORT = {"none": None, "off": None, "minimal": None, "disabled": None, "false": None,
          "low": "low", "medium": "medium", "high": "high", "xhigh": "high", "max": "high", "maximum": "high"}


def effort_kwargs(value) -> dict:
    """A reasoning effort as given by a client -> the template's kwargs.  Unknown values are a 400, not a crash."""
    if value is None or value == "":
        return {}
    if value is False:
        return {"enable_thinking": False}
    key = str(value).strip().lower()
    if key not in EFFORT:
        raise ValueError(f"unknown reasoning effort {value!r}: use none, low, medium or high")
    level = EFFORT[key]
    return {"enable_thinking": False} if level is None else {"reasoning_effort": level}


def budget_effort(tokens) -> dict:
    """Anthropic's thinking budget (budget_tokens) -> a level: under 2K low, under 8K medium, else high."""
    try:
        n = int(tokens)
    except (TypeError, ValueError):
        return {}
    return {"reasoning_effort": "low" if n < 2048 else "medium" if n < 8192 else "high"}


# The names the server gives the template itself: a client's chat_template_kwargs cannot replace them
RESERVED_KWARGS = ("messages", "tools", "add_generation_prompt", "raise_exception", "strftime_now")


def _off(value) -> bool:
    """enable_thinking as a client sends it: false, 0, null, or the words "false", "off", "no", "0" turn it off."""
    return not value or (isinstance(value, str) and value.strip().lower() in ("false", "off", "no", "0"))


def merge_template_kwargs(kwargs: dict, extra) -> dict:
    """`kwargs` (from the request's own effort or thinking fields) with the client's chat_template_kwargs on top - the
    vLLM / llama.cpp convention, {"chat_template_kwargs": {"enable_thinking": false, "clear_thinking": false}}.  Every
    key reaches the template (GLM's reads clear_thinking: false keeps the reasoning of earlier turns), except the
    names the server sets itself; reasoning_effort is read as the request's own effort, and enable_thinking false
    turns thinking off whatever the effort says."""
    if extra is None:
        return kwargs
    if not isinstance(extra, dict):
        raise ValueError("chat_template_kwargs must be an object")
    out = dict(kwargs)
    if "reasoning_effort" in extra:
        out = {k: v for k, v in out.items() if k not in ("enable_thinking", "reasoning_effort")}
        out.update(effort_kwargs(extra["reasoning_effort"]))
    if "enable_thinking" in extra:
        if _off(extra["enable_thinking"]):
            out.pop("reasoning_effort", None)
            out["enable_thinking"] = False
        else:
            out["enable_thinking"] = True
    for k, v in extra.items():
        if k in RESERVED_KWARGS:
            LOGGER.debug("chat_template_kwargs.%s is set by the server; the client's value is left out", k)
        elif k not in ("enable_thinking", "reasoning_effort"):
            out[k] = v
    return out


def _has_image(content) -> bool:
    return isinstance(content, list) and any(isinstance(p, dict) and p.get("type") in IMAGE_PARTS for p in content)


def _image_source(part: dict) -> str:
    """An image part's source as one string: a data: URL, an http(s) URL or a local file path.
    OpenAI: {"type": "image_url", "image_url": {"url": ...}} (or "image_url": "..."), Responses-style
    {"type": "input_image", "image_url": ...}; Anthropic: {"type": "image", "source": {"type": "base64",
    "media_type": ..., "data": ...}} or {"source": {"type": "url", "url": ...}}."""
    if part.get("type") == "image":
        src = part.get("source") or {}
        if src.get("type") == "base64":
            return f"data:{src.get('media_type', 'image/png')};base64,{src.get('data', '')}"
        return src.get("url") or src.get("path") or ""
    url = part.get("image_url")
    if isinstance(url, dict):
        url = url.get("url")
    return url or ""


def _parts_of(content):
    """Message content for the template: a string when there is no image, otherwise the template's list form - text
    items and image items, in order - whose image items carry their source (the template renders the text items;
    images_of finds the pictures)."""
    if not _has_image(content):
        return _text_of(content)
    items = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in IMAGE_PARTS:
            items.append({"type": "image", "source": _image_source(part)})
        elif part.get("type") in TEXT_PARTS and "text" in part:
            items.append({"type": "text", "text": part.get("text", "")})
    return items


# The blocks a messages request may carry: text, the image parts, thinking, tool_use and tool_result.
# redacted_thinking is allowed only to be dropped without an error: another server's encrypted reasoning, which
# carries nothing this server can read - the same reason responses.py's _reasoning_text gives for another server's
# encrypted_content.
ANTHROPIC_BLOCKS = ("text",) + IMAGE_PARTS + ("thinking", "tool_use", "tool_result", "redacted_thinking")


def _check_parts(content, path: str, kinds):
    """A request's content parts (on chat) or content blocks (on messages), looked at while the request is converted.
    A part whose type this server does not read (a file, audio, a document, Anthropic's tool_reference,
    search_result, server_tool_use / web_search_tool_result blocks, ...) is left out of the prompt exactly as it always
    was, with a debug line: real clients (Claude Code, Codex, opencode, Cline, Continue, Open WebUI) send such parts
    next to text the model can answer from, so refusing them would break requests that work.  A "refusal" part is read
    as its text, as /v1/responses does.  Content that is not a list, and a part that is not an object, passes through
    unchanged."""
    if not isinstance(content, list):
        return content
    out = content
    for j, part in enumerate(content):
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "refusal":
            if out is content:
                out = list(content)
            out[j] = {"type": "text", "text": part.get("refusal") or ""}
        elif kind not in kinds:
            LOGGER.debug("%s[%d].type: content part of type %r is not read by this server; left out of the prompt",
                         path, j, kind)
    return out


# A tool result's picture, in the text the model reads instead (see _tool_text)
IMAGE_OMITTED = "[image omitted: the model reads text only]"


def _tool_text(content) -> str:
    """A tool result's content as one string, as GLM's template takes it (a list would be rendered item by item as
    Python reprs): its text parts in order, with IMAGE_OMITTED in place of each picture.  The model reads no images,
    and a tool's picture (Claude Code's Read of an image file) must not fail the request: clients send the whole
    history again every turn."""
    if not _has_image(content):
        return _text_of(content)
    return "".join(IMAGE_OMITTED if p.get("type") in IMAGE_PARTS else p.get("text", "") for p in content
                   if isinstance(p, dict) and (p.get("type") in IMAGE_PARTS or p.get("type") in TEXT_PARTS))


def images_of(messages: list[dict]) -> list[str]:
    """The image sources of the conversation's user and assistant messages, in order (a tool's pictures are already
    text, see _tool_text)."""
    return [item["source"] for m in messages if isinstance(m.get("content"), list)
            for item in m["content"] if item.get("type") == "image"]


# #537 / #2026: special-token text inside a message is plain text.  The tokenizer matches GLM's tags (<think>,
# </think>, <tool_call>, <arg_key>, <tool_response>, ...) and its control tokens (<|user|>, <|observation|>, [gMASK],
# ...) as their special ids everywhere, so a user quoting "</think>", a tool result that contains "</tool_response>" or
# a file that contains "<|user|>" handed the model a real end of reasoning, of a result or of the turn.  Before the
# template is rendered such texts are swapped for private-use characters, and the server encodes the spans they mark
# as ordinary text.
GLM_TAGS = (THINK_START, THINK_END, CALL_START, CALL_END, KEY_START, KEY_END, VALUE_START, VALUE_END, RESPONSE_START,
            RESPONSE_END)
# The tags of a call.  An assistant message's content keeps them: a client replaying the model's raw call text sends
# the tokens the model wrote.
CALL_TAGS = (CALL_START, CALL_END, KEY_START, KEY_END, VALUE_START, VALUE_END)
MARK0 = 0xF0E01              # the first mark (plane 15's private use area)
LITERAL_TAGS = {tag: chr(MARK0 + k) for k, tag in enumerate(GLM_TAGS)}
LITERAL_MARKS = {v: k for k, v in LITERAL_TAGS.items()}


def literal_tags(texts) -> dict[str, str]:
    """LITERAL_TAGS plus a mark for each text in `texts`: the server passes every special token of the tokenizer
    (GGUF token types 3 and 4: <|user|>, <|observation|>, [gMASK], ... and GLM's tags), whose text written inside a
    message - a file an agent reads, a pasted chat template - would otherwise be that token.  Longest first, as the
    tokenizer matches them: a text inside a longer one is not marked before it."""
    tags = dict(LITERAL_TAGS)
    for text in texts:
        if text and text not in tags:
            tags[text] = chr(MARK0 + len(tags))
    return dict(sorted(tags.items(), key=lambda t: -len(t[0])))


def _mark(text: str, tags: dict[str, str] = LITERAL_TAGS) -> str:
    for tag, mark in tags.items():
        text = text.replace(tag, mark)
    return text


def _mark_deep(v, tags: dict[str, str] = LITERAL_TAGS):
    if isinstance(v, str):
        return _mark(v, tags)
    if isinstance(v, dict):
        return {k: _mark_deep(x, tags) for k, x in v.items()}
    if isinstance(v, list):
        return [_mark_deep(x, tags) for x in v]
    return v


def _has_tag(v, tags: dict[str, str] = LITERAL_TAGS) -> bool:
    if isinstance(v, str):
        return any(tag in v for tag in tags)
    if isinstance(v, dict):
        return any(_has_tag(x, tags) for x in v.values())
    if isinstance(v, list):
        return any(_has_tag(x, tags) for x in v)
    return False


def mark_think_literals(messages: list[dict], tools: list[dict] | None, tags: dict[str, str] = LITERAL_TAGS):
    """#537: (messages, tools) with every special-token text in their text swapped for its mark (LITERAL_TAGS, or
    `tags`: literal_tags() adds the tokenizer's special tokens), and whether there was one (False: no change, the same
    objects back - a prompt without them renders as it always did).  An assistant message's content keeps its call
    tags, and one that opens with a whole <think>...</think> block (clients that send the reasoning inline) keeps
    that block's two tags as the model's markers: the template reads them to split the reasoning off."""
    if not _has_tag(messages, tags) and not _has_tag(tools, tags):
        return messages, tools, False
    own = {t: v for t, v in tags.items() if t not in CALL_TAGS}     # what an assistant's content keeps as written
    out = []
    for m in messages:
        assistant = m.get("role") == "assistant"
        marked = {k: v if k == "role" else _mark_deep(v, own if assistant and k == "content" else tags)
                  for k, v in m.items()}
        content = m.get("content")
        if assistant and isinstance(content, str) and content.lstrip().startswith(THINK_START) \
                and THINK_END in content:
            i, j = content.index(THINK_START) + len(THINK_START), content.index(THINK_END)
            marked["content"] = (content[:i] + _mark(content[i:j], own) + THINK_END +
                                 _mark(content[j + len(THINK_END):], own))
        out.append(marked)
    return out, _mark_deep(tools, tags), True


_LITERAL_MARK_RE = re.compile("|".join(LITERAL_MARKS))


def unmark_think_literals(prompt: str, tags: dict[str, str] = LITERAL_TAGS) -> tuple[str, list[tuple[int, int]]]:
    """The rendered prompt with the marks turned back into the texts they stand for, and the (start, end) spans of
    those texts in it: the server encodes them as ordinary text (the tokenizer's encode_plain_spans)."""
    marks = LITERAL_MARKS if tags is LITERAL_TAGS else {v: k for k, v in tags.items()}
    mark_re = _LITERAL_MARK_RE if tags is LITERAL_TAGS else re.compile("|".join(map(re.escape, marks)))
    out, spans, pos, n = [], [], 0, 0
    for m in mark_re.finditer(prompt):
        out.append(prompt[pos:m.start()])
        n += m.start() - pos
        tag = marks[m.group(0)]
        out.append(tag)
        spans.append((n, n + len(tag)))
        n += len(tag)
        pos = m.end()
    out.append(prompt[pos:])
    return "".join(out), spans


def _late_system_to_user(messages: list[dict]) -> list[dict]:
    """Some chat templates take a system message only at the start ("System message must be at the beginning").
    Clients also send them mid-conversation - Claude Code's hook context as {"role": "system"} after the first user
    turn, some OpenAI clients a late "developer" message (issue #56) - so those become user messages, in place:
    merging them into the first one would change the prompt's start and cost the conversation cache every turn."""
    return [dict(m, role="user") if m.get("role") == "system" and i > 0 else m for i, m in enumerate(messages)]


def _object_list(value, name: str) -> list[dict]:
    """#460: a request's "messages" (or a message's "tool_calls") as a list of objects.  Some clients send the array
    double-encoded, as a JSON string, which used to be iterated character by character and crashed on m.get: such a
    string is decoded.  Anything that is still not a list of objects is a ValueError, which the server answers with
    a 400 naming the field.  None (or no field) is an empty list, as a missing field always was."""
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise ValueError(f"{name} must be a list of objects (a string was sent that is not JSON)") from None
    if not isinstance(value, list) or not all(isinstance(m, dict) for m in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


_TOOL_ORDERS: "OrderedDict[frozenset, tuple]" = OrderedDict()   # a set of tool names -> the order it first came in
_TOOL_ORDERS_LOCK = threading.Lock()
_TOOL_ORDERS_MAX = 64


def _stable_tool_order(tools: list[dict], name_of) -> list[dict]:
    """The same set of tools comes back in the order it was first seen.  Clients send their tools in the order their
    MCP servers finished registering, which can differ from one request (or one client restart) to the next, and the
    template renders the tool list before the system prompt: one swap moves the shared prefix back thousands of
    tokens and the whole conversation is read again (opencode issue 23571).  A request's first sight of a tool set
    is untouched (so its prompt is byte for byte what the client sent); later requests with the same names in
    another order are put back in that first order.  Duplicate names, or a single tool: left as sent."""
    if len(tools) < 2:
        return tools
    names = [name_of(t) for t in tools]
    if len(set(names)) != len(names):
        return tools
    key = frozenset(names)
    with _TOOL_ORDERS_LOCK:
        order = _TOOL_ORDERS.get(key)
        if order is None:
            _TOOL_ORDERS[key] = tuple(names)
            while len(_TOOL_ORDERS) > _TOOL_ORDERS_MAX:
                _TOOL_ORDERS.popitem(last=False)
            return tools
        _TOOL_ORDERS.move_to_end(key)
    pos = {n: k for k, n in enumerate(order)}
    return sorted(tools, key=lambda t: pos[name_of(t)])


def _tool_list(value, wrapper: str | None) -> list[dict]:
    """#592: a request's "tools" as a list of tool objects, each with a name - in the OpenAI shape
    {"type": "function", "function": {"name": ...}} (`wrapper` "function"; a bare {"name": ...} is still taken), or
    in the Anthropic shape {"name": ...} (`wrapper` None).  A value that is not (a string such as "auto", a list of
    names, an object without a name) is a ValueError - a 400 naming the field - where it used to take the request
    thread down with no reply at all.  No value (or an empty one) is no tools, as always.  A tool's schema -
    "parameters" in the OpenAI shape, "input_schema" in the Anthropic one - is an object when it is there, for the
    same reason: a string or a list passed and raised later instead, on the model's first call of that tool."""
    if not value:
        return []
    shape = ('{"type": "function", "function": {"name": ..., "parameters": {...}}}' if wrapper else
             '{"name": ..., "input_schema": {...}}')
    try:
        tools = _object_list(value, "tools")
    except ValueError:
        raise ValueError(f"tools must be a list of tool objects, each {shape}") from None
    key = "parameters" if wrapper else "input_schema"
    for i, t in enumerate(tools):
        fn = t.get(wrapper, t) if wrapper and t.get("type") == wrapper else t
        if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
            raise ValueError(f"tools[{i}] has no name: each tool must be {shape}")
        schema = fn.get(key)
        if schema is not None and not isinstance(schema, dict):
            # it reached parse_tool_call and the stream parser as the tool's schema, whose .get("properties") raised
            # AttributeError in the request thread - after the 200 and whatever the model had said before the call
            raise ValueError(f'tools[{i}] ({fn["name"]}): "{key}" must be an object (the JSON schema of its '
                             f"parameters), not {type(schema).__name__}")
    return _stable_tool_order(tools, lambda t: (t.get(wrapper, t) if wrapper and t.get("type") == wrapper else t)["name"])


def tool_arguments(raw) -> dict:
    """A call's arguments from a client's history -> the mapping the template renders (both APIs).  A JSON object
    (as text or already parsed) is used as it is; empty is {}; anything else (cut-off or malformed JSON, a list, a
    number) is kept as text under "arguments" instead of refusing the whole conversation (#510)."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            value = json.loads(raw)
        except ValueError:
            return {"arguments": raw}
        return value if isinstance(value, dict) else {"arguments": raw}
    return {"arguments": json.dumps(raw, ensure_ascii=False)}


def openai_to_messages(req: dict) -> tuple[list[dict], list[dict] | None, dict]:
    """OpenAI Chat Completions -> (template messages, template tools, template kwargs)."""
    messages = []
    for i, m in enumerate(_object_list(req.get("messages"), "messages")):
        role = m.get("role")
        if role == "developer":
            role = "system"
        # an image stays an image item in a user or assistant message (images_of finds it) and becomes a note in a
        # tool result (_tool_text); the system prompt takes text only, as _text_of reads
        images = role in ("user", "tool", "assistant")
        content = _check_parts(m.get("content"), f"messages[{i}].content",
                               TEXT_PARTS + IMAGE_PARTS if images else TEXT_PARTS)
        out = {"role": role, "content": _tool_text(content) if role == "tool" else
               _parts_of(content) if images else _text_of(content)}
        if m.get("reasoning_content"):
            out["reasoning_content"] = m["reasoning_content"]
        if m.get("tool_calls"):
            calls = []
            for c in _object_list(m["tool_calls"], "tool_calls"):
                fn = c.get("function", c)
                if not isinstance(fn, dict):
                    raise ValueError("tool_calls must be a list of objects (each with a \"function\" object)")
                calls.append({"function": {"name": fn.get("name"), "arguments": tool_arguments(fn.get("arguments"))}})
            out["tool_calls"] = calls
        messages.append(out)
    tools = [t.get("function", t) if t.get("type") == "function" else t
             for t in _tool_list(req.get("tools"), "function")] or None
    kwargs = {}
    # OpenAI Chat Completions: "reasoning_effort"; Responses style: "reasoning": {"effort": ...}
    reasoning = req.get("reasoning") if isinstance(req.get("reasoning"), dict) else {}
    kwargs.update(effort_kwargs(req.get("reasoning_effort") or reasoning.get("effort")))
    kwargs = merge_template_kwargs(kwargs, req.get("chat_template_kwargs"))
    return _late_system_to_user(messages), tools, kwargs


BILLING_HEADER = "x-anthropic-billing-header:"


def pin_billing_stamp(system: str) -> str:
    """Claude Code starts its system prompt with `x-anthropic-billing-header: cc_version=2.1.170.bf4;
    cc_entrypoint=sdk-cli; cch=b145e;`.  cch changes on EVERY request and the version's 4th part on every session, so
    the prompt changed ~22K tokens in (after the tool list) on every turn and the conversation cache could only reuse
    up to its last 16K checkpoint: half of every agent prompt was read again.  Both stamps are pinned to f's, as
    llama.cpp does (ggml-org/llama.cpp#21793); only a header at the very start of the system text is touched, and
    only inside its first 160 characters."""
    if not system.startswith(BILLING_HEADER):
        return system
    s = list(system)
    cch = system.find("cch=", len(BILLING_HEADER))
    if 0 <= cch <= 160:
        v, end = cch + 4, system.find(";", cch + 4)
        if end > v and end - v <= 16:
            s[v:end] = "f" * (end - v)
    cv = system.find("cc_version=")
    if 0 <= cv <= 160:
        v, end = cv + len("cc_version="), system.find(";", cv)
        if v < end and end - v <= 64:
            parts = system[v:end].split(".")
            if len(parts) > 3:
                tail = v + len(".".join(parts[:3])) + 1
                s[tail:end] = "f" * (end - tail)
    return "".join(s)


def anthropic_to_messages(req: dict, think_unasked: bool = True) -> tuple[list[dict], list[dict] | None, dict]:
    """Anthropic Messages -> (template messages, template tools, template kwargs).  `think_unasked`: a request
    without "thinking", an effort or a budget gets the template's default (it thinks), as through 0.1.31; False
    renders it without thinking (#278, the config's "anthropic_thinking": "on_request")."""
    messages = []
    system = req.get("system")
    if system:
        messages.append({"role": "system",
                         "content": pin_billing_stamp(_text_of(_check_parts(system, "system", TEXT_PARTS)))})
    for i, m in enumerate(_object_list(req.get("messages"), "messages")):
        content = _check_parts(m.get("content"), f"messages[{i}].content", ANTHROPIC_BLOCKS)
        if isinstance(content, str):
            messages.append({"role": m["role"], "content": content})
            continue
        if m.get("role") == "user" and _has_image(content) and not any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in content):
            messages.append({"role": "user", "content": _parts_of(content)})
            continue
        text, reasoning, calls, parts = [], [], [], []
        for j, block in enumerate(content or []):
            kind = block.get("type")
            if kind == "text":
                text.append(block.get("text", ""))
                parts.append(block)
            elif kind in IMAGE_PARTS:
                parts.append(block)
            elif kind == "thinking":
                reasoning.append(block.get("thinking", ""))
            elif kind == "tool_use":
                calls.append({"function": {"name": block.get("name"), "arguments": tool_arguments(block.get("input"))}})
            elif kind == "tool_result":
                # one string, a tool's picture (Claude Code's Read of an image file) as a note, as on the OpenAI path
                messages.append({"role": "tool", "content": _tool_text(_check_parts(
                    block.get("content"), f"messages[{i}].content[{j}].content", TEXT_PARTS + IMAGE_PARTS))})
        if text or calls or reasoning or parts:
            # an image sent beside tool results stays in this turn instead of being dropped
            out = {"role": m["role"], "content": _parts_of(parts) if _has_image(parts) else "".join(text)}
            if reasoning:
                out["reasoning_content"] = "".join(reasoning)
            if calls:
                out["tool_calls"] = calls
            messages.append(out)
    tools = [{"name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema", {})}
             for t in _tool_list(req.get("tools"), None)] or None
    kwargs = {}
    # Anthropic: "thinking": {"type": "disabled"} or {"type": "enabled", "budget_tokens": N};
    # "output_config": {"effort": "low" | "medium" | "high"}
    thinking = req.get("thinking")
    effort = (req.get("output_config") or {}).get("effort") if isinstance(req.get("output_config"), dict) else None
    if isinstance(thinking, dict) and thinking.get("type") == "disabled":
        kwargs["enable_thinking"] = False
    elif effort:
        kwargs.update(effort_kwargs(effort))
    elif isinstance(thinking, dict) and thinking.get("budget_tokens"):
        kwargs.update(budget_effort(thinking["budget_tokens"]))
    elif thinking is None and not req.get("reasoning_budget_tokens") and not think_unasked:
        # Opt-in (the config's "anthropic_thinking": "on_request"; the default thinks as 0.1.31 did, since a
        # client that never asks would otherwise lose the thinking on every turn).  Anthropic's thinking is
        # opt-in there. Claude Code's helper calls (a session title, a topic check) ask for none
        # and allow a few dozen tokens, which the model otherwise spent thinking and answered with no text at all.
        # A config's reasoning_effort still applies: Service.with_shared sets output_config before this runs.  A
        # request that gives its own reasoning_budget_tokens (#123) asks for thinking, so it thinks as before.
        kwargs["enable_thinking"] = False
    kwargs = merge_template_kwargs(kwargs, req.get("chat_template_kwargs"))
    return _late_system_to_user(messages), tools, kwargs


# ------------------------------------------------------------------------------------------------ output parser
# The tags OutputParser reads in a reply (<tool_response> and </tool_response> are the template's, never the model's)
OUTPUT_TAGS = (THINK_START, THINK_END, CALL_START, CALL_END, KEY_START, KEY_END, VALUE_START, VALUE_END)
# A "<" written by an ordinary token, as the server hands it to OutputParser when the tokenizer has every one of
# OUTPUT_TAGS as a special token (server.output_tags): the model writes its own tags as those tokens, so the same text
# spelled with ordinary tokens - a call or a </think> copied from a tool result, a web page or the user, whose tags
# #537 put in the prompt as ordinary text - never forms a tag.  A lone surrogate, which decoded text never holds;
# every event gives it back as "<".
ORDINARY_LT = "\udc3c"


def _restore(text: str) -> str:
    return text.replace(ORDINARY_LT, "<")


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:24])


@dataclass
class Event:
    kind: str                     # "reasoning" | "content" | "tool_start" | "tool_args" | "tool_call"
    text: str = ""
    call: ToolCall | None = None


def tool_choice_of(tool_choice) -> tuple[str, str | None]:
    """A client's `tool_choice` in either API's shape -> ("auto" | "none" | "required" | "named", name).  OpenAI:
    "auto" / "none" / "required", {"type": "function", "function": {"name": N}} or the flat {"type": "function",
    "name": N}; Anthropic: {"type": "auto" | "any" | "none"} and {"type": "tool", "name": N}.  A value it does not
    know is ("unknown", None)."""
    if tool_choice is None:
        return "auto", None
    if isinstance(tool_choice, str):
        return (tool_choice, None) if tool_choice in ("auto", "none", "required") else ("unknown", None)
    if isinstance(tool_choice, dict):
        kind = tool_choice.get("type")
        if kind in ("auto", "none"):
            return kind, None
        if kind in ("any", "required"):
            return "required", None
        if kind in ("function", "tool"):
            fn = tool_choice.get("function")
            name = (fn.get("name") if isinstance(fn, dict) else None) or tool_choice.get("name")
            if isinstance(name, str) and name:
                return "named", name
    return "unknown", None


def forced_call(tool_choice, tools: list[dict] | None) -> str | None:
    """`tool_choice` -> the text that opens the call the reply must make, or None (the model decides).
    There is no grammar here: the server writes this opening itself, so the model can only go on with a call.
    "required" / Anthropic "any": `<tool_call>`, and the model names one of the tools; a named function (or the only
    tool): `<tool_call>NAME`, and the model writes its arguments.  A value it cannot honour (an unknown shape, a name
    that is not one of the tools, "required" with no tools) is logged and acts as "auto", not a 400: a client's odd
    choice must not stop its request.  "none" is handled by the caller (no tools)."""
    kind, name = tool_choice_of(tool_choice)
    names = {t.get("name") for t in tools or [] if isinstance(t, dict)}
    if kind in ("auto", "none"):
        return None
    if kind == "required" and len(names) == 1:     # one tool to call: name it, the model cannot invent another
        kind, name = "named", next(iter(names))
    if kind == "required" and names:
        return CALL_START
    if kind == "named" and name in names:
        return CALL_START + name
    print(f"[strata] tool_choice {json.dumps(tool_choice)[:200]} is not supported here (or names no tool of the "
          "request): the model decides, as with \"auto\"", flush=True)
    return None


_JSON_FENCE = re.compile(r"```[ \t]*(?:json)?[ \t]*\r?\n\s*", re.I)


def json_tool_call(body: str, schemas: dict) -> ToolCall | None:
    """`{"name": N, "arguments": {...}}` (or "parameters", or the arguments as a JSON string) of a declared tool -> its
    call; anything else -> None.  #2037: the object may sit in a Markdown fence (```json ... ```)."""
    body = body.strip()
    m = _JSON_FENCE.match(body)
    if m:
        body = body[m.end():].rstrip()
        body = body[:-3] if body.endswith("```") else body
    try:
        obj = json.loads(body)
    except ValueError:
        return None
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or obj["name"] not in schemas:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return None
    if args is None:
        args = {}
    return ToolCall(name=obj["name"], arguments=args) if isinstance(args, dict) else None


def _json_follower(after: str) -> str | None:
    """recover: what the text after a `<tool_call>` (whitespace stripped) opens: "json" (a JSON object, bare or in a
    ```json fence), "wait" (not decided yet) or None (anything else)."""
    if after.startswith("{"):
        return "json"
    if after.startswith("```") or (after and "```".startswith(after)):
        m = _JSON_FENCE.match(after)
        if m:
            rest = after[m.end():]
            if rest.startswith("{"):
                return "json"
            return "wait" if not rest else None
        return "wait" if "\n" not in after else None
    return "wait" if not after else None


def _partial(text: str, tags: tuple[str, ...]) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of one of `tags` (a tag still arriving)."""
    best = 0
    for tag in tags:
        for n in range(min(len(tag) - 1, len(text)), best, -1):
            if text.endswith(tag[:n]):
                best = n
                break
    return best


_WS = re.compile(r"\s*")


def value_end(text: str, start: int = 0) -> tuple[int, str]:
    """Where an argument value in `text` ends: the first `</arg_value>` followed (after whitespace) by the next
    `<arg_key>` or the call's `</tool_call>` - the same text inside a value (a file that documents the call format,
    #210) is part of the value.  -> (position, "end"); (position of a candidate whose follower has not arrived,
    "wait"); or (where a later search must start, "open"): no end yet.  `start`: no end lies before it."""
    at = text.find(VALUE_END, start)
    while at >= 0:
        k = _WS.match(text, at + len(VALUE_END)).end()
        after = text[k:k + len(CALL_END)]
        if after.startswith((KEY_START, CALL_END)):
            return at, "end"
        if k + len(after) == len(text) and (KEY_START.startswith(after) or CALL_END.startswith(after)):
            return at, "wait"
        at = text.find(VALUE_END, at + 1)
    return max(start, len(text) - len(VALUE_END) + 1), "open"


def _string_params(tool) -> set[str]:
    """The parameters a tool's schema types as "string": their values are kept as written."""
    params = tool.get("parameters") if isinstance(tool, dict) else None
    props = params.get("properties") if isinstance(params, dict) else None
    if not isinstance(props, dict):
        return set()
    return {k for k, v in props.items() if isinstance(v, dict) and v.get("type") == "string"}


def _escape(text: str) -> str:
    """`text` as the inside of a JSON string: pieces escaped one by one join into the escaped whole."""
    return json.dumps(text)[1:-1]


_NAME = re.compile(r"[\w.:/-]{1,256}")    # a tool name, when there are no declared names to check it against


class CallReader:
    """One tool call in GLM's form, read as its text arrives (the text after `<tool_call>`): the tool's NAME, then
    `<arg_key>K</arg_key><arg_value>V</arg_value>` for each argument (whitespace between the parts is allowed), then
    `</tool_call>`.  `stage` is "name" until the name has ended, then "prose" if that text is not a call (a name that is
    not a declared tool's, or not followed by `<arg_key>` / `</tool_call>`), otherwise the body's stages and "done"
    after `</tool_call>`; the text after it stays in `text`.

    feed() returns the call's arguments as pieces of one JSON object: "{" once the name is known, then for each
    argument `"key":` and its value - a string parameter's text JSON-escaped as it arrives, any other value whole
    once it is complete - and "}" at `</tool_call>`.  Values are kept as written: a parameter the tool's schema types
    as "string" is the text itself, any other is JSON-decoded (the text itself when it is not JSON), as the template
    writes them (`v | tojson if v is not string else v`).  `args` holds the decoded arguments."""

    def __init__(self, schemas: dict, declared: bool = True):
        self.schemas = schemas
        # declared: the name must be one of the request's tools; otherwise any name-like word is a call's name
        self.names = {n for n in schemas if isinstance(n, str)} if declared else None
        self.text = ""               # the text not read yet
        self.seen: list[str] = []    # all the text given, as given
        self.stage = "name"          # name -> between -> key -> value -> str | raw -> between ... -> done (or prose)
        self.name: str | None = None
        self.args: dict = {}
        self.strings: set[str] = set()
        self.key = ""
        self.count = 0               # arguments written so far
        self.part: list[str] = []    # the current value's text read so far
        self.seek = 0                # where the search for the current value's end resumes in self.text

    def _could_be(self, head: str) -> bool:
        """`head` (the text after `<tool_call>`, no `<` in it yet) may still become a call's name."""
        s = head.lstrip()
        core = s.rstrip()
        if self.names is None:
            return not core or _NAME.fullmatch(core) is not None
        if core != s:                # the name has ended (whitespace after it)
            return core in self.names
        return any(n.startswith(core) for n in self.names)

    def _is_name(self, name: str) -> bool:
        return name in self.names if self.names is not None else _NAME.fullmatch(name) is not None

    def _typed(self, key: str, value: str):
        if key in self.strings:
            return value
        try:
            return json.loads(value)
        except (ValueError, RecursionError):
            return value

    def written(self) -> str:
        return "".join(self.seen)

    def feed(self, text: str) -> list[str]:
        self.seen.append(text)
        self.text += text
        out: list[str] = []
        while True:
            t = self.text
            if self.stage == "name":
                lt = t.find("<")
                head = t if lt < 0 else t[:lt]
                if not self._could_be(head):
                    self.stage = "prose"
                    return out
                if lt < 0:
                    return out
                after = t[lt:]
                if after.startswith((KEY_START, CALL_END)):
                    name = head.strip()
                    if not self._is_name(name):
                        self.stage = "prose"
                        return out
                    self.name, self.strings = name, _string_params(self.schemas.get(name))
                    self.text, self.stage = after, "between"
                    out.append("{")
                elif KEY_START.startswith(after) or CALL_END.startswith(after):
                    return out           # a tag still arriving
                else:
                    self.stage = "prose"
                    return out
            elif self.stage == "between":
                # after the name or a value: the next argument or the end.  Both follow whitespace at most (the name
                # is taken only before one of them, and a value ends only before one), so nothing else comes here.
                s = t.lstrip()
                if s.startswith(KEY_START):
                    self.text, self.stage = s[len(KEY_START):], "key"
                elif s.startswith(CALL_END):
                    self.text, self.stage = s[len(CALL_END):], "done"
                    out.append("}")
                    return out
                else:
                    self.text = s
                    return out
            elif self.stage == "key":
                k = t.find(KEY_END)
                if k < 0:
                    return out
                self.key, self.text, self.stage = _restore(t[:k].strip()), t[k + len(KEY_END):], "value"
            elif self.stage == "value":
                s = t.lstrip()
                if s.startswith(VALUE_START):
                    t = s[len(VALUE_START):]
                elif not s or VALUE_START.startswith(s):
                    return out
                # (otherwise the value was written without its <arg_value>: it starts right here)
                out.append(("," if self.count else "") + json.dumps(self.key) + ":")
                self.count += 1
                self.stage = "str" if self.key in self.strings else "raw"
                if self.stage == "str":
                    out.append('"')
                self.text, self.part, self.seek = t, [], 0
            elif self.stage in ("str", "raw"):
                at, how = value_end(t, self.seek)
                if how == "end":
                    piece, self.text = t[:at], t[at + len(VALUE_END):]
                    value = _restore("".join(self.part) + piece)
                    if self.stage == "str":
                        self.args[self.key] = value
                        out.append(_escape(_restore(piece)) + '"')
                    else:
                        self.args[self.key] = self._typed(self.key, value)
                        out.append(json.dumps(self.args[self.key], ensure_ascii=False))
                    self.part, self.stage = [], "between"
                    continue
                # what cannot be the value's end is read now (a string value's goes out as it arrives); a candidate
                # end, or a tag still arriving, waits
                safe = at if how == "wait" else len(t) - _partial(t, (VALUE_END,))
                if safe > 0:
                    self.part.append(t[:safe])
                    self.text = t[safe:]
                    if self.stage == "str":
                        out.append(_escape(_restore(t[:safe])))
                self.seek = 0 if how == "wait" else max(0, at - safe)
                return out
            else:                        # done, prose
                return out

    def close(self) -> list[str]:
        """The JSON pieces that end the arguments of a call whose text was cut off (an empty list before the name is
        known or after the end)."""
        if self.stage in ("name", "prose", "done"):
            return []
        tail = ""
        if self.stage == "str":
            tail = '"'
        elif self.stage == "raw":
            at, how = value_end(self.text, self.seek)
            cut = at if how == "wait" else len(self.text) - _partial(self.text, (VALUE_END,))
            tail = json.dumps(self._typed(self.key, _restore("".join(self.part) + self.text[:cut])),
                              ensure_ascii=False)
        self.stage = "done"
        return [tail + "}"]


def parse_tool_call(body: str, tools: list[dict] | None = None) -> ToolCall:
    """A whole call's text, `NAME<arg_key>K</arg_key><arg_value>V</arg_value>...` (what is between `<tool_call>` and
    `</tool_call>`) -> ToolCall, its values typed by the tool's schema in `tools`.  With tools the name must be one of
    them.  ValueError: the text is not a call."""
    reader = CallReader({t.get("name"): t for t in tools or [] if isinstance(t, dict)}, declared=bool(tools))
    reader.feed(body + CALL_END)
    if reader.stage != "done" or reader.text:
        raise ValueError("malformed tool call: " + body[:80])
    return ToolCall(name=reader.name, arguments=reader.args)


RCALL_MAX = 32768        # the most of a `<tool_call>` inside the reasoning that is held while waiting for its end (#804)


class OutputParser:
    """Incremental parser of the model's text. Feed deltas; get events. A tag split across deltas is held back
    until it is complete, so clients never see `<tool_` or `</thi`.

    GLM's reply is `[<think>]REASONING</think>CONTENT` with tool calls `<tool_call>NAME<arg_key>K</arg_key>
    <arg_value>V</arg_value>...</tool_call>` in the content (the prompt usually opens the `<think>`; `thinking` says
    it did).  Events: "reasoning" and "content" text, and for each call "tool_call" with its name and arguments.
    `state` is "reasoning", "rcall" (a call inside the reasoning, held), "content", "call" (a call being read) or
    "json" (recover: a JSON call).  `buf` is the text held back: a tag still arriving or newlines that may come
    before a call in the reasoning and content states (never a whole `</think>`: it ends the thinking at once), the
    held call in "rcall" and "json", nothing in "call" (the CallReader holds it)."""

    def __init__(self, thinking: bool = True, tools: list[dict] | None = None, stream_tools: bool = False,
                 recover: bool = False):
        self.state = "reasoning" if thinking else "content"
        self.buf = ""
        self.lead = False            # newlines right after </think> or a call are not content
        self.opened = thinking       # the reply's <think> has been written (by the prompt, or by the reply itself)
        self.schemas = {t.get("name"): t for t in tools or [] if isinstance(t, dict)}
        # stream_tools: a tool call is also reported while it is being written - "tool_start" (its name and id) as
        # soon as the name is known, then "tool_args" pieces of its JSON arguments (string parameters character by
        # character; other types whole, once complete) - before the final "tool_call".  Without it, a client sees
        # nothing until the call is complete, which for a large file write can be many minutes.
        self.stream_tools = stream_tools
        # #804/#1058: calls found inside the reasoning wait here as [raw text, ToolCall | None] until the turn shows
        # they were acts: only whitespace (or more calls) after them, then the end of the turn or `</think>`.  The
        # reasoning text before them is tracked (code fence, inline code) to tell an act from a quote.
        self.pending: list[list] = []
        self.rescued = 0             # calls delivered from the reasoning
        self.refused = 0             # declared calls kept as reasoning text (quoted, or the turn was cut)
        self.fence, self.line, self.ticks = "", "", 0
        # recover (opt-in, the config's "tool_call_recovery"): a declared tool's call written as a JSON object
        # ({"name": ..., "arguments": {...}}, bare or in a ```json fence) inside <tool_call> is the call
        self.recover = recover
        self.reader: CallReader | None = None    # the call being read (states "call" and "rcall")
        self.rfed = 0                # how much of an rcall's text its reader has been given
        self.gap = ""                # the newlines before the call's <tool_call>: dropped if it is one
        self.scall: ToolCall | None = None       # the call being read, once its name is known (its id is kept)

    @property
    def line(self) -> str:
        """Materialize the current line only for a decision, newline, or state snapshot."""
        if self._line_blocks is not None:
            self._line_blocks.extend(self._line_parts)
            self._line_parts = ["".join(self._line_blocks)]
            self._line_blocks = None
        elif len(self._line_parts) > 1:
            self._line_parts = ["".join(self._line_parts)]
        return self._line_parts[0] if self._line_parts else ""

    @line.setter
    def line(self, text: str):
        self._line_parts = [text] if text else []
        self._line_blocks = None

    def _track(self, text: str) -> str:
        """Follow the text that has gone out: the open code fence, the current line, and the backticks of the current
        paragraph.  Returns the text, as the client gets it (ORDINARY_LT given back as "<")."""
        # Most token fragments stay on the current line. Fence and paragraph
        # bookkeeping happens only when a newline completes that line.
        if "\n" not in text:
            if text:
                self._line_parts.append(text)
                if len(self._line_parts) == 64:
                    # Release small fragment objects in bounded groups. Completed
                    # blocks are never recopied until the full line is requested.
                    if self._line_blocks is None:
                        self._line_blocks = []
                    self._line_blocks.append("".join(self._line_parts))
                    self._line_parts.clear()
            return _restore(text)
        parts = text.split("\n")
        for k, part in enumerate(parts):
            if k < len(parts) - 1:
                line = self.line + part if self._line_parts or self._line_blocks else part
                self._line_parts.clear()
                self._line_blocks = None
                s = line.lstrip()
                if self.fence:
                    if s.startswith(self.fence * 3):
                        self.fence = ""
                elif s[:3] in ("```", "~~~") and (s[0] * 3) not in s[3:]:
                    self.fence = s[0]
                elif not s:
                    self.ticks = 0
                else:
                    self.ticks += line.count("`")
            else:
                if part:
                    self._line_parts.append(part)
        return _restore(text)

    def _in_code(self) -> bool:
        """The text so far leaves the next character inside a code fence or inline code."""
        if self.fence:
            # a line that closes the fence (its marker, then nothing but marker characters or blanks) ends the code
            # before its newline: GLM writes a call right after the content, "...```<tool_call>"
            s = self.line.lstrip()
            return not s.startswith(self.fence * 3) or bool(s.strip(self.fence + " \t"))
        return (self.ticks + self.line.count("`")) % 2 == 1

    def _in_code_at(self, text: str) -> bool:
        """The text so far followed by `text` (not gone out yet) leaves the next character in code."""
        snap = (self.fence, self.line, self.ticks)
        self._track(text)
        in_code = self._in_code()
        self.fence, self.line, self.ticks = snap
        return in_code

    def _release(self, deliver: bool) -> list[Event]:
        """Settle the calls waiting in self.pending: events for real calls, or all of it back as reasoning text."""
        out = []
        for raw, call in self.pending:
            if call is not None and deliver:
                out.append(Event("tool_call", call=call))
                self.rescued += 1
            else:
                if call is not None:
                    self.refused += 1
                if raw:
                    out.append(Event("reasoning", self._track(raw)))
        self.pending = []
        return out

    def _open_rcall(self):
        """self.buf starts with a `<tool_call>` inside the reasoning: hold it whole until it ends (#804).  Any name is
        read; whether it is a declared tool's is decided at its end."""
        self.state, self.reader, self.rfed = "rcall", CallReader(self.schemas, declared=False), 0

    def feed(self, delta: str) -> list[Event]:
        self.buf += delta
        out: list[Event] = []
        while True:
            if self.state == "rcall":
                # #804: a `<tool_call>` inside the reasoning, with tools declared.  It is held whole (never streamed
                # as a call) until it ends: a declared name is then a tool call, anything else stays reasoning text.
                body = self.buf[len(CALL_START):]
                self.reader.feed(body[self.rfed:])
                self.rfed = len(body)
                r, think = self.reader, body.find(THINK_END)
                end = len(body) - len(r.text) if r.stage == "done" else -1     # just past its </tool_call>
                if end >= 0 and (think < 0 or think >= end):
                    call = ToolCall(name=r.name, arguments=r.args) if r.name in self.schemas else None
                    raw = self.buf[:len(CALL_START) + end]
                    if call is not None or self.pending:
                        self.pending.append([raw, call])    # settled by what follows it (see __init__)
                    else:
                        out.append(Event("reasoning", self._track(raw)))
                    self.buf, self.state, self.reader = body[end:], "reasoning", None
                elif r.stage == "prose":             # not a call's name: the tag is text, the rest is read again
                    out += self._release(False)
                    out.append(Event("reasoning", self._track(CALL_START)))
                    self.buf, self.state, self.reader = body, "reasoning", None
                elif think >= 0:                     # the thinking ended inside it: it never was a call
                    out += self._release(False)
                    out.append(Event("reasoning", self._track(self.buf[:len(CALL_START) + think])))
                    self.buf, self.state, self.reader = body[think:], "reasoning", None
                elif len(body) > RCALL_MAX:          # a call that never ends: stop holding the thinking back
                    out += self._release(False)
                    out.append(Event("reasoning", self._track(self.buf)))
                    self.buf, self.state, self.reader = "", "reasoning", None
                else:
                    return out
            elif self.state == "reasoning":
                if self.pending:
                    # calls wait for what follows: more calls or whitespace keep them, `</think>` makes them acts,
                    # any other text means they were quoted
                    stripped = self.buf.lstrip()
                    if len(stripped) < len(self.buf):
                        self.pending[-1][0] += self.buf[:len(self.buf) - len(stripped)]
                        self.buf = stripped
                    if not self.buf:
                        return out
                    if self.buf.startswith(CALL_START):
                        self._open_rcall()
                    elif self.buf.startswith(THINK_END):
                        out += self._release(True)
                    elif CALL_START.startswith(self.buf) or THINK_END.startswith(self.buf):
                        return out
                    else:
                        out += self._release(False)
                    continue
                i = self.buf.find(THINK_END)
                tool = self.buf.find(CALL_START) if self.schemas else -1
                if tool >= 0 and (i < 0 or tool < i):
                    if tool:
                        out.append(Event("reasoning", self._track(self.buf[:tool])))
                    self.buf = self.buf[tool:]
                    # in a fence, inline code or a "> " blockquote line: a quote, never run.  GLM writes its calls
                    # right after text ("I will read it.<tool_call>read..."), so other text before it on the line is
                    # no sign of a quote
                    if self._in_code() or self.line.lstrip().startswith(">"):
                        out.append(Event("reasoning", self._track(CALL_START)))
                        self.buf = self.buf[len(CALL_START):]
                    else:
                        self._open_rcall()
                    continue
                if i < 0:
                    keep = _partial(self.buf, (THINK_END, CALL_START) if self.schemas else (THINK_END,))
                    if len(self.buf) > keep:
                        out.append(Event("reasoning", self._track(self.buf[:len(self.buf) - keep])))
                        self.buf = self.buf[len(self.buf) - keep:]
                    return out
                # GLM writes the answer right after </think>, with no line break: every </think> ends the thinking
                if i:
                    out.append(Event("reasoning", self._track(self.buf[:i])))
                self.buf = self.buf[i + len(THINK_END):]
                self.state, self.lead, self.opened = "content", True, True
                self.fence, self.line, self.ticks = "", "", 0     # the answer's own text starts here
            elif self.state == "content":
                if self.lead:
                    stripped = self.buf.lstrip("\n")
                    if not stripped:
                        self.buf = ""
                        return out
                    self.buf, self.lead = stripped, False
                if not self.opened:
                    # a reply that opens its own <think> (the prompt did not): its reasoning comes first
                    s = self.buf.lstrip()
                    if s.startswith(THINK_START):
                        self.buf, self.state, self.opened = s[len(THINK_START):], "reasoning", True
                        continue
                    if THINK_START.startswith(s):
                        return out
                    self.opened = True
                if not self.schemas:
                    # no tools declared (or tool_choice "none", which reaches here as none): no call, the tags are text
                    out.append(Event("content", self._track(self.buf)))
                    self.buf = ""
                    return out
                i = self.buf.find(CALL_START)
                if i < 0:
                    # Hold a partial tag AND the newlines before it: if a tool call follows, they are dropped,
                    # so emitting them early would make streamed and whole outputs differ.
                    j = len(self.buf) - _partial(self.buf, (CALL_START,))
                    while j > 0 and self.buf[j - 1] == "\n":
                        j -= 1
                    if j > 0:
                        out.append(Event("content", self._track(self.buf[:j])))
                        self.buf = self.buf[j:]
                    return out
                # #1058: an opener inside a code fence or inline code is text (a quoted example), never a call.  The
                # text before it decides, so streamed and whole outputs agree.  Mid-sentence openers count: GLM
                # writes its call right after the text, with no line break.
                if self._in_code_at(self.buf[:i]):
                    out.append(Event("content", self._track(self.buf[:i + len(CALL_START)])))
                    self.buf = self.buf[i + len(CALL_START):]
                    continue
                j = i
                while j > 0 and self.buf[j - 1] == "\n":
                    j -= 1
                if j > 0:
                    out.append(Event("content", self._track(self.buf[:j])))
                    self.buf, i = self.buf[j:], i - j
                rest = self.buf[i + len(CALL_START):]
                kind = _json_follower(rest.lstrip()) if self.recover and self.schemas else None
                if kind == "wait":
                    return out
                self.gap, self.buf = self.buf[:i], rest
                if kind == "json":
                    self.state = "json"
                else:
                    # the text after the tag decides: a declared tool's name and then <arg_key> or </tool_call> is a
                    # call; anything else is prose that names the tag ("I'll use a <tool_call> block") - content, not
                    # a malformed call
                    self.state, self.reader = "call", CallReader(self.schemas)
            elif self.state == "json":
                # recover: `{"name": ..., "arguments": {...}}` inside the wrapper is the call once its </tool_call> has
                # arrived and it names a declared tool; anything else is the text it is
                i = self.buf.find(CALL_END)
                if i < 0:
                    return out
                call = json_tool_call(_restore(self.buf[:i]), self.schemas)
                if call is None:
                    out.append(Event("content", self._track(self.gap + CALL_START + self.buf[:i + len(CALL_END)])))
                else:
                    out.append(Event("tool_call", call=call))
                self.buf = self.buf[i + len(CALL_END):]
                self.state, self.lead = "content", call is not None
            else:                                    # call
                r = self.reader
                text, self.buf = self.buf, ""
                pieces = r.feed(text)
                if r.stage == "prose":
                    # not a call: the tag is text, and what came after it is read again as the content it is
                    out.append(Event("content", self._track(self.gap + CALL_START)))
                    self.buf, self.state, self.reader = r.text, "content", None
                    continue
                if r.name is not None and self.scall is None:
                    self.scall = ToolCall(name=r.name, arguments={})
                    if self.stream_tools:
                        out.append(Event("tool_start", call=self.scall))
                if self.stream_tools:
                    out += [Event("tool_args", piece, call=self.scall) for piece in pieces]
                if r.stage != "done":
                    return out
                out.append(Event("tool_call", call=ToolCall(name=r.name, arguments=r.args, id=self.scall.id)))
                self.buf, self.reader, self.scall = r.text, None, None
                self.state, self.lead = "content", True

    def finish(self, reason: str | None = None) -> list[Event]:
        """End of generation: flush whatever is held.  A call cut off before its `</tool_call>` gets no "tool_call"
        (#211): one already announced (stream_tools) has its JSON closed so clients can still parse it; one that was
        not is returned as the text it is.  `reason` is how the turn ended: calls waiting from the reasoning become
        real calls only on a natural stop (None = stop); a turn cut by max tokens (or cancelled, or failed) keeps
        them as reasoning text (#1058)."""
        out = []
        if self.pending:
            out += self._release(reason in (None, "stop") and self.state == "reasoning" and not self.buf)
        if self.state == "call":
            if self.stream_tools and self.scall is not None:
                out += [Event("tool_args", piece, call=self.scall) for piece in self.reader.close()]
            else:
                out.append(Event("content", self._track(self.gap + CALL_START + self.reader.written())))
        elif self.state == "json":
            out.append(Event("content", self._track(self.gap + CALL_START + self.buf)))
        elif self.buf:
            # an unfinished call inside the reasoning (#804) is reasoning text, never a call
            out.append(Event("content" if self.state == "content" else "reasoning", self._track(self.buf)))
        self.buf, self.reader, self.scall = "", None, None
        return out
