"""JSON response formats at the HTTP boundary: prompt once, validate before delivery.

The native engine has no grammar decoder. Failed generations are errors, never
silently retried or returned as successful structured output.

`jsonschema` is optional (no new hard dependency of the server): json_object needs only the standard library, and
json_schema is checked against its schema when the package is installed - without it, only that the answer is one
JSON object, and the server says so once.
"""
import functools
import json
import re
import threading
import time

import regex

try:
    from re import _parser as _re_parser
except ImportError:                    # Python 3.10
    import sre_parse as _re_parser

PATTERN_TIME_S = 2.0                   # the most time a schema's patterns may take on one answer, all together
PATTERN_SIZE_MAX = 10000               # the most parts a pattern may have with its repeats written out (_pattern_size)
_REPEATS = {getattr(_re_parser, op) for op in ("MAX_REPEAT", "MIN_REPEAT", "POSSESSIVE_REPEAT")
            if hasattr(_re_parser, op)}


class StructuredOutputError(RuntimeError):
    pass


class _ObjectOnly:
    """The validator for json_object, and for json_schema without jsonschema: one JSON object."""

    def iter_errors(self, value):
        if not isinstance(value, dict):
            yield _ObjectError()


class _ObjectError:
    absolute_path = ()
    message = "the answer is not a JSON object"


_jsonschema = None                     # (validators, SchemaError, Registry, NoSuchResource), False when not installed
_jsonschema_lock = threading.Lock()
_patterns_end = threading.local()      # .end: when this thread's validation stops its patterns (time.monotonic())


def _pattern_size(node) -> int:
    """The parts of a pattern parsed by re, its repeats written out to their minimum count, as regex does when it
    compiles one - before its timeout runs: a{100000000} took 4 GB and 5 s, (?:a{1000}){1000} 270 MB."""
    if isinstance(node, _re_parser.SubPattern):
        return sum(max(av[0], 1) * _pattern_size(av[2]) if op in _REPEATS else 1 + _pattern_size(av)
                   for op, av in node)
    if isinstance(node, (tuple, list)):
        return sum(_pattern_size(x) for x in node)
    return 0


@functools.lru_cache(maxsize=16)       # not regex's own cache: it keeps 500 patterns
def _compiled(pattern):
    if _pattern_size(_re_parser.parse(pattern)) > PATTERN_SIZE_MAX:
        raise ValueError(f"the pattern {pattern[:80]!r} repeats too much to be checked (more than "
                         f"{PATTERN_SIZE_MAX} parts with its repeats written out)")
    return regex.compile(pattern, cache_pattern=False)


_DATA_KEYS = ("const", "enum", "default", "examples")       # values, not schemas: a "pattern" in them is no pattern
_NAMED_SCHEMAS = ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas")


def _check_patterns(node, path="") -> None:
    """Every pattern of a schema (`pattern`, and `patternProperties`' names) is parsed and its size checked when the
    request arrives: a ValueError (a 400 that names where it is) instead of a 502 once the whole answer is written.
    `path`: where `node` is in the schema, as a JSON pointer."""
    def check(pattern, where):
        try:
            _compiled(pattern)
        except (re.error, regex.error) as exc:
            raise ValueError(f"response_format schema at {where}: {pattern[:80]!r} is not a regular expression "
                             f"({exc})") from None
        except (ValueError, RecursionError, OverflowError) as exc:
            raise ValueError(f"response_format schema at {where}: {exc}") from None

    if isinstance(node, list):
        for i, value in enumerate(node):
            _check_patterns(value, f"{path}/{i}")
        return
    if not isinstance(node, dict):
        return
    for key, value in node.items():
        where = f"{path}/{key}"
        if key == "pattern" and isinstance(value, str):
            check(value, where)
        elif key in _NAMED_SCHEMAS and isinstance(value, dict):
            for name, sub in value.items():      # names, not keywords: a property may be called "const"
                if key == "patternProperties":
                    check(name, where)
                _check_patterns(sub, f"{where}/{name}")
        elif key not in _DATA_KEYS:
            _check_patterns(value, where)


class _TimedRe:
    """`re` as jsonschema's keywords see it.  `pattern` and `patternProperties` are the client's regular expressions,
    run on the model's answer: one such as ^(a|a)+$ doubles its time with each character, and re cannot be stopped
    (nor does it let other threads run).  The `regex` module runs them instead, with re's syntax, and stops them
    once the answer's patterns took PATTERN_TIME_S (TimeoutError)."""

    def __getattr__(self, name):
        return getattr(re, name)

    @staticmethod
    def search(pattern, string):
        end = getattr(_patterns_end, "end", None)
        left = PATTERN_TIME_S if end is None else end - time.monotonic()
        if left <= 0:
            raise TimeoutError("regex timed out")
        return _compiled(pattern).search(string, timeout=left)


def jsonschema_modules():
    """jsonschema, imported on first use; None when it is not installed (said once, in the server window)."""
    global _jsonschema
    with _jsonschema_lock:
        if _jsonschema is None:
            try:
                from jsonschema import _keywords, _legacy_keywords, _utils, validators
                from jsonschema.exceptions import SchemaError
                from referencing import Registry
                from referencing.exceptions import NoSuchResource
                for module in (_keywords, _legacy_keywords, _utils):   # where jsonschema 4 runs re.search
                    module.re = _TimedRe()
                _jsonschema = (validators, SchemaError, Registry, NoSuchResource)
            except ImportError:
                _jsonschema = False
                print('[strata] response_format json_schema: the Python package jsonschema is not installed, so '
                      'answers are only checked to be one JSON object (python -m pip install "jsonschema>=4.23,<5")',
                      flush=True)
        return _jsonschema or None


def _only_objects(node, root, refs=()):
    """True when every value `node` accepts is a JSON object, so "return one JSON object" stays true.

    `type: object`, an anyOf/oneOf whose branches all qualify (e.g. a root union of object shapes, which llama.cpp's
    grammar path accepts and apps send), an allOf with a qualifying member, or a local `$ref` to one of these.
    Anything that can also be an array, string, number, boolean or null (including `type: ["object", "null"]`) is
    not, and a `$ref` cycle never qualifies.
    """
    if not isinstance(node, dict):
        return False
    if node.get("type") == "object":
        return True
    ref = node.get("$ref")
    if isinstance(ref, str) and ref.startswith("#") and ref not in refs:
        target = root
        for part in ref[1:].split("/")[1:]:
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or part not in target:
                return False
            target = target[part]
        return _only_objects(target, root, refs + (ref,))
    for key in ("anyOf", "oneOf"):
        branches = node.get(key)
        if isinstance(branches, list) and branches and all(_only_objects(b, root, refs) for b in branches):
            return True
    branches = node.get("allOf")
    return isinstance(branches, list) and any(_only_objects(b, root, refs) for b in branches)


def prepare_format(response_format, messages, with_tools=False):
    if response_format is None:
        return messages, None
    if not isinstance(response_format, dict):
        raise ValueError("response_format must be an object")
    kind = response_format.get("type")
    if kind == "text":
        return messages, None
    if kind == "json_object":
        schema = {"type": "object"}
        modules = None                 # the standard library is enough for "one JSON object"
    elif kind == "json_schema":
        spec = response_format.get("json_schema")
        if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
            raise ValueError("response_format.json_schema needs a schema object")
        if not isinstance(spec.get("name"), str) or not spec["name"]:
            raise ValueError("response_format.json_schema needs a name")
        if "strict" in spec and not isinstance(spec["strict"], bool):
            raise ValueError("response_format.json_schema.strict must be boolean")
        schema = spec["schema"]
        if not _only_objects(schema, schema):
            raise ValueError("response_format schema must accept only JSON objects at its root "
                             "(type object, or anyOf/oneOf of object schemas)")
        modules = jsonschema_modules()
    else:
        raise ValueError("response_format.type must be text, json_object or json_schema")

    def check_refs(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("$ref", "$dynamicRef") and isinstance(value, str) and not value.startswith("#"):
                    raise ValueError("response_format supports only local schema references (#...)")
                check_refs(value)
        elif isinstance(node, list):
            for value in node:
                check_refs(value)
    check_refs(schema)
    if modules is None:
        validator = _ObjectOnly()
    else:
        validators, SchemaError, Registry, NoSuchResource = modules

        def no_remote(uri):
            raise NoSuchResource(ref=uri)

        try:
            cls = validators.validator_for(schema)
            cls.check_schema(schema)
            validator = cls(schema, registry=Registry(retrieve=no_remote))
        except SchemaError as exc:
            raise ValueError(f"invalid response_format schema: {exc.message}") from exc
        _check_patterns(schema)                # the patterns run on the answer: one too big to run is a 400 now
    directive = ("OUTPUT FORMAT REQUIREMENT: Return exactly one JSON object matching the JSON Schema below. "
                 "No Markdown, headings, code fences, commentary, or text outside the JSON. "
                 "Use every required field, correct types, and only allowed fields. "
                 "Put all requested writing inside the appropriate JSON string fields.\nJSON Schema:\n" +
                 json.dumps(schema, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    if with_tools:
        # /v1/responses (#782): the schema is for the final answer; a turn that calls a tool is not an answer
        directive += ("\nThis applies only to your final answer. To use a tool, call it as usual; the JSON object is "
                      "what you write once you are done with the tools.")
    messages = [dict(message) for message in messages]
    if messages and messages[0].get("role") == "system":
        content = messages[0].get("content") or ""
        if isinstance(content, list):
            messages[0]["content"] = content + [{"type": "text", "text": directive}]
        else:
            messages[0]["content"] = content + "\n\n" + directive
    else:
        messages.insert(0, {"role": "system", "content": directive})
    return messages, validator


def _extract_json(text: str) -> str:
    s = (text or "").strip()
    if not s:
        return ""
    if s.startswith("```"):
        first = s.find("\n")
        if first != -1:
            end = s.find("```", first + 1)
            if end != -1:
                s = s[first + 1:end].strip()
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
        else:
            if ch == "\"":
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:i + 1]
    return s[start:]


def validated_json(text, validator, finish):
    def pairs(items):
        obj = {}
        for key, value in items:
            if key in obj:
                raise ValueError(f"duplicate JSON key: {key}")
            obj[key] = value
        return obj

    def constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    if finish != "stop":
        raise StructuredOutputError(f"structured output was incomplete (finish_reason={finish}); increase the output budget")
    try:
        value = json.loads(_extract_json(text), object_pairs_hook=pairs, parse_constant=constant)
        canonical = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (ValueError, TypeError) as exc:
        raise StructuredOutputError(f"model did not return valid JSON: {exc}") from exc
    _patterns_end.end = time.monotonic() + PATTERN_TIME_S
    try:
        error = next(validator.iter_errors(value), None)
    except TimeoutError as exc:
        raise StructuredOutputError(f"the response_format schema's patterns took more than {PATTERN_TIME_S:g} s on "
                                    "the answer and were stopped (a pattern such as ^(a|a)+$ doubles its time with "
                                    "each character)") from exc
    except Exception as exc:
        raise StructuredOutputError(f"could not validate structured output: {exc}") from exc
    finally:
        _patterns_end.end = None
    if error is not None:
        path = "/" + "/".join(str(part) for part in error.absolute_path)
        raise StructuredOutputError(f"model output failed the JSON Schema at {path}: {error.message}")
    return canonical
