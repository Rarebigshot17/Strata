"""serve/test_glm_parser.py - frontend.OutputParser on GLM's reply format: reasoning, content and tool calls
`<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`, fed whole, character by character and in
random pieces, with the event contract the API layers rely on (tool_start / tool_args / tool_call).

    python -m unittest serve.test_glm_parser -v
"""
import json
import random
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serve.frontend import (GAP_MAX, RCALL_MAX, CallReader, OutputParser, forced_call,  # noqa: E402
                            parse_tool_call, value_end)

TOOLS = [{"name": "read_file", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "lines": {"type": "integer"}, "opts": {"type": "object"}}}},
    {"name": "write_file", "parameters": {"type": "object", "properties": {
        "path": {"type": "string"}, "content": {"type": "string"}}}},
    {"name": "ping", "parameters": {"type": "object", "properties": {}}}]

READ = ("<tool_call>read_file<arg_key>path</arg_key><arg_value>/tmp/1.txt</arg_value>"
        "<arg_key>lines</arg_key><arg_value>20</arg_value><arg_key>opts</arg_key>"
        "<arg_value>{\"raw\": true}</arg_value></tool_call>")
READ_ARGS = {"path": "/tmp/1.txt", "lines": 20, "opts": {"raw": True}}


def chunkings(text, seed=1729, count=12):
    """The ways a test feeds `text`: whole, one character at a time, in pairs, and `count` random splits."""
    yield [text]
    yield list(text)
    yield [text[i:i + 2] for i in range(0, len(text), 2)]
    rng = random.Random(seed)
    for _ in range(count):
        pieces, pos = [], 0
        while pos < len(text):
            n = rng.randint(1, 9)
            pieces.append(text[pos:pos + n])
            pos += n
        yield pieces


def feed(pieces, thinking=True, tools=TOOLS, stream_tools=True, recover=False, finish="stop", force=None):
    p = OutputParser(thinking=thinking, tools=tools, stream_tools=stream_tools, recover=recover)
    events = p.feed(force) if force else []
    for piece in pieces:
        events += p.feed(piece)
    return events + p.finish(finish), p


def check_contract(test, events):
    """The ordering rules of the API layers: a call's tool_args only after its tool_start, nothing else between its
    tool_start and its tool_call, its tool_args one JSON object, equal to the final arguments; tool_start carries the
    final name and no arguments, and the same id as its pieces and its tool_call."""
    open_call, args = None, ""
    for e in events:
        if open_call is not None:
            test.assertIn(e.kind, ("tool_args", "tool_call"), "another kind inside a streamed call")
            test.assertEqual(e.call.id, open_call.id)
            if e.kind == "tool_args":
                args += e.text
            else:
                test.assertEqual(e.call.name, open_call.name)
                test.assertEqual(json.loads(args), e.call.arguments)
                open_call, args = None, ""
        elif e.kind == "tool_start":
            test.assertEqual(e.call.arguments, {})
            open_call = e.call
        else:
            test.assertNotEqual(e.kind, "tool_args", "tool_args without its tool_start")
    if open_call is not None:                 # cut off: the JSON is closed all the same (#211)
        test.assertIsInstance(json.loads(args), dict)


def normal(events):
    """Events with consecutive text merged and each streamed call's pieces joined, ids left out."""
    out = []
    for e in events:
        if e.kind in ("reasoning", "content"):
            if not e.text:
                continue
            if out and out[-1][0] == e.kind:
                out[-1] = (e.kind, out[-1][1] + e.text)
            else:
                out.append((e.kind, e.text))
        elif e.kind == "tool_start":
            out.append(("tool_start", e.call.name))
        elif e.kind == "tool_args":
            if out and out[-1][0] == "tool_args":
                out[-1] = ("tool_args", out[-1][1] + e.text)
            else:
                out.append(("tool_args", e.text))
        else:
            out.append(("tool_call", e.call.name, e.call.arguments))
    return out


class Base(unittest.TestCase):
    def parse(self, text, **kw):
        """The normalized events of `text`, the same for every way of feeding it, with the contract checked."""
        results = []
        for pieces in chunkings(text):
            events, _ = feed(pieces, **kw)
            check_contract(self, events)
            results.append(normal(events))
        for r in results[1:]:
            self.assertEqual(r, results[0])
        return results[0]

    def calls(self, text, **kw):
        return [(e[1], e[2]) for e in self.parse(text, **kw) if e[0] == "tool_call"]

    def text(self, text, kind="content", **kw):
        return "".join(e[1] for e in self.parse(text, **kw) if e[0] == kind)


class ReasoningAndContent(Base):
    def test_reasoning_then_answer(self):
        self.assertEqual(self.parse("I should greet.</think>\n\nHello!"),
                         [("reasoning", "I should greet."), ("content", "Hello!")])

    def test_answer_right_after_the_thinking(self):
        # GLM writes the answer right after </think>: a quote or backtick before it changes nothing (#537 was Qwen's)
        for text in ('He said "ok"</think>Answer', "Use `x`</think>Answer", "'a'</think>Answer"):
            with self.subTest(text=text):
                self.assertEqual(self.text(text), "Answer")
                self.assertEqual(self.text(text, "reasoning"), text.split("</think>")[0])

    def test_no_reasoning(self):
        self.assertEqual(self.parse("Just an answer.", thinking=False), [("content", "Just an answer.")])

    def test_reply_opens_its_own_think(self):
        for text in ("<think>hmm</think>Answer", "\n<think>hmm</think>\nAnswer", "  <think>hmm</think>Answer"):
            with self.subTest(text=text):
                self.assertEqual(self.parse(text, thinking=False), [("reasoning", "hmm"), ("content", "Answer")])

    def test_think_only_at_the_start_opens_reasoning(self):
        self.assertEqual(self.parse("Hi <think>x</think>", thinking=False), [("content", "Hi <think>x</think>")])
        self.assertEqual(self.parse("ok</think>Then <think>no</think>"),
                         [("reasoning", "ok"), ("content", "Then <think>no</think>")])

    def test_a_partial_think_at_the_end_is_text(self):
        self.assertEqual(self.parse("<thi", thinking=False), [("content", "<thi")])
        self.assertEqual(self.parse("abc</thi"), [("reasoning", "abc</thi")])

    def test_partial_tag_is_held_then_released(self):
        p = OutputParser(thinking=False, tools=TOOLS)
        self.assertEqual(normal(p.feed("x <tool")), [("content", "x ")])
        self.assertEqual(normal(p.feed("box>")), [("content", "<toolbox>")])
        p = OutputParser(thinking=True)
        self.assertEqual(normal(p.feed("a </th")), [("reasoning", "a ")])
        self.assertEqual(p.buf, "</th")

    def test_leading_newlines_are_not_content(self):
        self.assertEqual(self.text("x</think>\n\nHello\n"), "Hello\n")
        self.assertEqual(self.text("x</think>" + READ + "\n\nDone"), "Done")

    def test_newlines_held_before_a_call_are_bounded(self):
        # the newlines before a call are dropped if one follows, so they are held - at most GAP_MAX of them: a reply
        # of newline tokens ("\n", "\n\n" in turn, which #606's same-token stop does not end) rescanned the whole
        # held run on every token, and 20,000 of them took 25 s
        p = OutputParser(thinking=False, tools=TOOLS, stream_tools=False)
        events, start = p.feed("Answer."), time.perf_counter()
        for k in range(20000):
            events += p.feed("\n" if k % 2 else "\n\n")
            self.assertLessEqual(len(p.buf), GAP_MAX)
        self.assertLess(time.perf_counter() - start, 5.0)
        events += p.feed(READ) + p.finish()
        newlines = 30000 - GAP_MAX                               # the ones before the last GAP_MAX are content
        self.assertEqual(normal(events), [("content", "Answer." + "\n" * newlines),
                                          ("tool_call", "read_file", READ_ARGS)])
        self.assertEqual(self.parse("Answer." + "\n" * 30000 + READ, thinking=False, stream_tools=False),
                         normal(events))
        self.assertEqual(self.text("x</think>Hi\n\n" + READ), "Hi")

    def test_held_text_and_state_for_the_server(self):
        p = OutputParser(thinking=True, tools=TOOLS)
        p.feed("think")
        self.assertEqual((p.state, p.buf), ("reasoning", ""))
        p.feed("</think>")
        self.assertEqual((p.state, p.buf), ("content", ""))       # a clean point: a forced call may open here
        p.feed("Answer\n")
        self.assertEqual((p.state, p.buf), ("content", "\n"))     # newlines are held: a call may follow
        p.feed("<tool_call>read")
        self.assertEqual(p.state, "call")


class ToolCalls(Base):
    def test_tool_call_typed_by_schema(self):
        events = self.parse("Let me look." + READ, thinking=False)
        self.assertEqual(events[0], ("content", "Let me look."))
        self.assertEqual(events[-1], ("tool_call", "read_file", READ_ARGS))
        self.assertEqual(json.loads(events[2][1]), READ_ARGS)       # the streamed JSON

    def test_without_streaming(self):
        events = self.parse("Let me look." + READ, thinking=False, stream_tools=False)
        self.assertEqual(events, [("content", "Let me look."), ("tool_call", "read_file", READ_ARGS)])

    def test_string_param_that_looks_like_json(self):
        text = "<tool_call>read_file<arg_key>path</arg_key><arg_value>123</arg_value></tool_call>"
        self.assertEqual(self.calls(text, thinking=False), [("read_file", {"path": "123"})])

    def test_values_are_verbatim(self):
        text = ("<tool_call>write_file<arg_key>path</arg_key><arg_value> a b </arg_value><arg_key>content</arg_key>"
                "<arg_value>\nline 1\n\tline 2\n\n</arg_value><arg_key>mode</arg_key><arg_value> 5 </arg_value>"
                "<arg_key>raw</arg_key><arg_value>not json</arg_value></tool_call>")
        self.assertEqual(self.calls(text, thinking=False),
                         [("write_file", {"path": " a b ", "content": "\nline 1\n\tline 2\n\n", "mode": 5,
                                          "raw": "not json"})])

    def test_nullable_and_union_string_params(self):
        # str | None (Pydantic, FastMCP: anyOf; zod's .nullable(): a type list) and other unions with a string: a value
        # is decoded only when it is one of the other types the schema allows, else it is the text as written
        tools = [{"name": "w", "parameters": {"type": "object", "properties": {
            "content": {"anyOf": [{"type": "string"}, {"type": "null"}]}, "msg": {"type": ["string", "null"]},
            "n": {"oneOf": [{"type": "string"}, {"type": "integer"}]}, "f": {"type": ["number", "string"]}}}}]
        for key, value, want in (("content", '{"debug": true}', '{"debug": true}'), ("content", "null", None),
                                 ("content", '"q"', '"q"'), ("msg", "123", "123"), ("msg", "true", "true"),
                                 ("msg", "null", None), ("n", "42", 42), ("n", "4.5", "4.5"), ("n", "[1]", "[1]"),
                                 ("f", "4.5", 4.5), ("f", "4", 4), ("f", "x y", "x y")):
            with self.subTest(key=key, value=value):
                text = f"<tool_call>w<arg_key>{key}</arg_key><arg_value>{value}</arg_value></tool_call>"
                self.assertEqual(self.calls(text, thinking=False, tools=tools), [("w", {key: want})])

    def test_non_finite_numbers_are_text(self):
        # json.loads reads NaN, Infinity and 1e999 (inf), which json.dumps writes back as tokens strict JSON parsers
        # (JavaScript's JSON.parse) refuse: such a value is its text
        def strict(text):
            return json.loads(text, parse_constant=lambda c: self.fail(f"{c} in {text}"))
        for value in ("NaN", "Infinity", "-Infinity", "1e999", "-1e999", "[1, NaN]", '{"a": Infinity}'):
            with self.subTest(value=value):
                text = f"<tool_call>read_file<arg_key>lines</arg_key><arg_value>{value}</arg_value></tool_call>"
                for pieces in chunkings(text, count=3):
                    events, _ = feed(pieces, thinking=False)
                    self.assertEqual([e.call.arguments for e in events if e.kind == "tool_call"], [{"lines": value}])
                    self.assertEqual(strict("".join(e.text for e in events if e.kind == "tool_args")),
                                     {"lines": value})
                    self.assertEqual(strict(json.dumps(events[-1].call.arguments)), {"lines": value})
                p = OutputParser(thinking=False, tools=TOOLS, stream_tools=True)
                cut = p.feed(text[:-len("</arg_value></tool_call>")]) + p.finish("length")
                self.assertEqual(strict("".join(e.text for e in cut if e.kind == "tool_args")), {"lines": value})
                # recover's JSON form: not a call
                text = '<tool_call>{"name": "read_file", "arguments": {"lines": %s}}</tool_call>' % value
                self.assertEqual(self.parse(text, thinking=False, recover=True), [("content", text)])

    def test_escapes_and_unicode_stream_as_json(self):
        content = 'He said "hi" \\ path\\to\\x é\U0001F642 \x01\t end'
        text = (f"<tool_call>write_file<arg_key>path</arg_key><arg_value>a</arg_value><arg_key>content</arg_key>"
                f"<arg_value>{content}</arg_value></tool_call>")
        self.assertEqual(self.calls(text, thinking=False), [("write_file", {"path": "a", "content": content})])

    def test_two_calls_after_reasoning(self):
        text = ("plan</think><tool_call>a<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>\n"
                "<tool_call>b\n<arg_key>y</arg_key>\n<arg_value>two</arg_value>\n</tool_call>")
        tools = [{"name": "a"}, {"name": "b"}]
        self.assertEqual(self.parse(text, tools=tools, stream_tools=False),
                         [("reasoning", "plan"), ("tool_call", "a", {"x": 1}), ("tool_call", "b", {"y": "two"})])
        self.assertEqual(self.calls(text, tools=tools), [("a", {"x": 1}), ("b", {"y": "two"})])

    def test_several_calls_with_text_between(self):
        write = ("<tool_call>write_file<arg_key>path</arg_key><arg_value>b</arg_value><arg_key>content</arg_key>"
                 "<arg_value>x</arg_value></tool_call>")
        events = self.parse("go</think>First." + READ + "Then.\n\n" + write + "<tool_call>ping</tool_call>")
        self.assertEqual([e[0] for e in events],
                         ["reasoning", "content", "tool_start", "tool_args", "tool_call", "content", "tool_start",
                          "tool_args", "tool_call", "tool_start", "tool_args", "tool_call"])
        self.assertEqual(events[5], ("content", "Then."))
        self.assertEqual(events[-1], ("tool_call", "ping", {}))
        self.assertEqual(events[-2], ("tool_args", "{}"))

    def test_a_name_with_whitespace_around_it(self):
        text = "<tool_call> read_file\n<arg_key> lines </arg_key>\n<arg_value>3</arg_value>\n</tool_call>"
        self.assertEqual(self.calls(text, thinking=False), [("read_file", {"lines": 3})])

    def test_content_before_a_call_loses_its_newlines(self):
        self.assertEqual(self.parse("Let me look.\n\n" + READ, thinking=False)[0], ("content", "Let me look."))
        self.assertEqual(self.parse("Mid sentence " + READ, thinking=False)[0], ("content", "Mid sentence "))

    def test_an_undeclared_name_is_prose(self):
        for text in ("<tool_call>other<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>",
                     "<tool_call>read_files<arg_key>a</arg_key><arg_value>1</arg_value></tool_call>",
                     "<tool_call>read file</tool_call>", "<tool_call></tool_call>"):
            with self.subTest(text=text):
                self.assertEqual(self.parse("Hi " + text, thinking=False), [("content", "Hi " + text)])

    def test_prose_that_names_the_tag(self):
        text = "I'll use a <tool_call> block, then <tool_call>read_file is the name."
        self.assertEqual(self.parse(text, thinking=False), [("content", text)])

    def test_prose_then_a_real_call(self):
        events = self.parse("The <tool_call> tag:" + READ, thinking=False)
        self.assertEqual(events[:2], [("content", "The <tool_call> tag:"), ("tool_start", "read_file")])
        self.assertEqual(json.loads(events[2][1]), READ_ARGS)
        self.assertEqual(events[3:], [("tool_call", "read_file", READ_ARGS)])

    def test_without_declared_tools_nothing_is_a_call(self):
        # a request with no tools (or tool_choice "none", which the routes send as no tools) gets no call: the chatbot
        # asked to "repeat exactly: <tool_call>ban..." answers with that text
        for tools in (None, []):
            for text in ("Sure: <tool_call>ban<arg_key>user</arg_key><arg_value>everyone</arg_value></tool_call>",
                         "<tool_call>mcp__fs.read-file<arg_key>p</arg_key><arg_value>x</arg_value></tool_call>\n\n",
                         "Text <tool_call>not a name</tool_call> and <tool_call>{\"a\": 1}</tool_call>",
                         "<tool_call>read_file</tool_call>", "x\n\n<tool_ca"):
                with self.subTest(tools=tools, text=text):
                    self.assertEqual(self.parse(text, thinking=False, tools=tools), [("content", text)])
                    self.assertEqual(self.parse("ok</think>" + text, tools=tools, recover=True),
                                     [("reasoning", "ok"), ("content", text)])

    def test_a_value_may_contain_the_closing_tags(self):
        # #210: a file that documents the call format.  </arg_value> ends a value only before <arg_key> or
        # </tool_call>; the closing tags elsewhere in it are its text
        value = "a</arg_value> b</tool_call> c</arg_key>"
        text = (f"<tool_call>write_file<arg_key>content</arg_key><arg_value>{value}</arg_value><arg_key>path"
                f"</arg_key><arg_value>doc.md</arg_value></tool_call>")
        self.assertEqual(self.calls(text, thinking=False), [("write_file", {"content": value, "path": "doc.md"})])
        self.assertEqual(self.calls(text.replace("write_file", "read_file").replace("content", "opts"),
                                    thinking=False), [("read_file", {"opts": value, "path": "doc.md"})])

    def test_an_unfinished_call_is_closed_but_not_a_call(self):
        # #211: cut off before </tool_call>: the streamed JSON is closed, and no tool_call follows
        for cut in ("<tool_call>read_file<arg_key>path</arg_key><arg_value>/tm",
                    "<tool_call>read_file<arg_key>path</arg_key><arg_value>/tmp</arg_value>",
                    "<tool_call>read_file<arg_key>lines</arg_key><arg_value>2",
                    "<tool_call>read_file<arg_key>lines</arg_key><arg_value>20</arg_val",
                    "<tool_call>read_file<arg_key>path</arg_key><arg_value>/tmp</arg_value><arg_key>li",
                    "<tool_call>read_file<arg_key>pa",
                    "<tool_call>read_file<arg_key>path</arg_key><arg_value>a</arg_value></tool_ca"):
            with self.subTest(cut=cut):
                events = self.parse(cut, thinking=False, finish="length")
                self.assertEqual([e[0] for e in events], ["tool_start", "tool_args"])
                self.assertIsInstance(json.loads(events[1][1]), dict)
                self.assertEqual(self.parse(cut, thinking=False, finish="length", stream_tools=False),
                                 [("content", cut)])
        events = self.parse("<tool_call>read_file<arg_key>lines</arg_key><arg_value>2", thinking=False)
        self.assertEqual(json.loads(events[1][1]), {"lines": 2})

    def test_a_name_not_yet_complete_at_the_end_is_text(self):
        for cut in ("<tool_call>", "<tool_call>read", "Hi\n\n<tool_call>read_file", "<tool_call>read_file<arg_"):
            with self.subTest(cut=cut):
                self.assertEqual(self.parse(cut, thinking=False), [("content", cut)])

    def test_a_large_value_streams_as_it_arrives(self):
        content = "".join(f"line {i}: <b>`x`</b> \"q\"\n" for i in range(3000))
        text = (f"<tool_call>write_file<arg_key>path</arg_key><arg_value>big.txt</arg_value><arg_key>content"
                f"</arg_key><arg_value>{content}</arg_value></tool_call>")
        p = OutputParser(thinking=False, tools=TOOLS, stream_tools=True)
        events, pieces = [], 0
        for i in range(0, len(text), 3):
            new = p.feed(text[i:i + 3])
            pieces += sum(e.kind == "tool_args" for e in new)
            events += new
        check_contract(self, events)
        self.assertGreater(pieces, len(content) // 3 - 10)          # piece by piece, not at the end
        self.assertEqual(events[-1].call.arguments, {"path": "big.txt", "content": content})


class CodeFences(Base):
    def test_calls_in_code_are_text(self):
        for body in ("Format:\n```xml\n" + READ + "\n```\nDone.", "Format:\n~~~\n" + READ + "\n~~~\nDone.",
                     "Use `" + READ + "` like this.", "Use `x`, then `\n" + READ + "`", "```\n" + READ):
            with self.subTest(body=body[:20]):
                self.assertEqual(self.parse("ok</think>\n\n" + body), [("reasoning", "ok"), ("content", body)])
                self.assertEqual(self.parse(body, thinking=False), [("content", body)])

    def test_calls_after_code_count(self):
        for body in ("```\ncode\n```\n", "`a` and `b` ", "~~~\nx\n~~~\n\n", "Let me do it. "):
            with self.subTest(body=body):
                self.assertEqual(self.calls("ok</think>" + body + READ), [("read_file", READ_ARGS)])

    def test_a_call_right_after_the_closing_fence(self):
        # GLM's template writes a turn's content and then its call with no line break ("...```<tool_call>"), and the
        # model writes its calls that way: the closing fence's line ends the code
        for body in ("I'll save this:\n```bash\necho hi\n```", "~~~\nx\n~~~", "```\na\n  ````  ", "```\na\n```\n"):
            with self.subTest(body=body):
                self.assertEqual(self.parse("ok</think>" + body + READ, stream_tools=False),
                                 [("reasoning", "ok"), ("content", body.rstrip("\n")),
                                  ("tool_call", "read_file", READ_ARGS)])
                self.assertEqual(self.calls("plan:\n" + body + READ + "</think>"), [("read_file", READ_ARGS)])
        # a fence's opening line, or a line that does not close it, is still code
        for body in ("```bash\necho hi\n```bash", "```\na\n``` x", "~~~\na\n```", "```"):
            with self.subTest(body=body):
                self.assertEqual(self.parse("ok</think>" + body + READ), [("reasoning", "ok"), ("content", body + READ)])

    def test_the_answer_starts_outside_code(self):
        # an open fence in the reasoning does not carry over into the answer
        self.assertEqual(self.calls("```\nnot closed</think>" + READ), [("read_file", READ_ARGS)])


class CallsInReasoning(Base):
    """#804/#1058: a call written inside the thinking (tools declared) is held whole, and is a call only when the turn
    shows it was an act: nothing but whitespace or more calls after it, then </think> or the turn's natural end."""

    def test_a_stranded_call_is_delivered_on_a_natural_stop(self):
        for text in ("I will read it." + READ, READ, "Hmm.\n  " + READ + "\n", "Two." + READ + "\n\n" + READ):
            with self.subTest(text=text[:20]):
                calls = self.calls(text)
                self.assertEqual(calls, [("read_file", READ_ARGS)] * text.count("<tool_call>"))
        self.assertEqual(self.text("Let me think.\n\n" + READ, "reasoning"), "Let me think.\n\n")

    def test_calls_before_the_end_of_the_thinking(self):
        events = self.parse("Plan." + READ + "\n</think>\n\nDone")
        self.assertEqual(events, [("reasoning", "Plan."), ("tool_call", "read_file", READ_ARGS),
                                  ("content", "Done")])

    def test_a_call_followed_by_text_was_quoted(self):
        text = "I could run " + READ + " but first let me check."
        self.assertEqual(self.parse(text), [("reasoning", text)])
        p = OutputParser(thinking=True, tools=TOOLS)
        p.feed(text)
        p.finish("stop")
        self.assertEqual((p.rescued, p.refused), (0, 1))

    def test_a_cut_turn_keeps_the_call_as_text(self):
        for finish in ("length", "cancel", "error", "disconnect"):
            with self.subTest(finish=finish):
                self.assertEqual(self.parse("Now." + READ, finish=finish), [("reasoning", "Now." + READ)])
        p = OutputParser(thinking=True, tools=TOOLS)
        p.feed("go" + READ)
        p.finish("stop")
        self.assertEqual((p.rescued, p.refused), (1, 0))

    def test_quoted_in_code_or_undeclared_stays_reasoning(self):
        for text in ("Example:\n```xml\n" + READ + "\n```\nok", "Use `" + READ + "`",
                     "go " + READ.replace("read_file", "other"), "x <tool_call>not a name"):
            with self.subTest(text=text[:20]):
                self.assertEqual(self.parse(text), [("reasoning", text)])

    def test_no_tools_declared(self):
        for tools in (None, []):
            self.assertEqual(self.parse("before" + READ + "</think>answer", tools=tools),
                             [("reasoning", "before" + READ), ("content", "answer")])

    def test_the_thinking_ending_inside_a_call(self):
        text = "go<tool_call>read_file<arg_key>path</arg_key><arg_value>x</think>answer"
        self.assertEqual(self.parse(text), [("reasoning", text.split("</think>")[0]), ("content", "answer")])

    def test_a_call_waits_and_is_not_streamed_early(self):
        p = OutputParser(thinking=True, tools=TOOLS, stream_tools=True)
        events = p.feed("go\n" + READ + "\n")
        self.assertEqual([e.kind for e in events], ["reasoning"])
        self.assertEqual(p.state, "reasoning")
        self.assertEqual([e.kind for e in p.finish("stop")], ["tool_call"])
        p = OutputParser(thinking=True, tools=TOOLS)
        p.feed("go<tool_call>read_file<arg_key>pa")
        self.assertEqual(p.state, "rcall")

    def test_a_half_opener_after_the_call_was_prose(self):
        self.assertEqual(self.parse("ok\n" + READ + "\n<tool_c"), [("reasoning", "ok\n" + READ + "\n<tool_c")])

    def test_an_unclosed_call_is_held_up_to_a_limit(self):
        head = "go<tool_call>read_file<arg_key>path</arg_key><arg_value>"
        p = OutputParser(thinking=True, tools=TOOLS)
        events = p.feed(head + "x" * 100)
        self.assertEqual(normal(events), [("reasoning", "go")])
        events = p.feed("x" * RCALL_MAX)
        self.assertEqual(normal(events), [("reasoning", head[len("go"):] + "x" * (RCALL_MAX + 100))])
        self.assertEqual(p.state, "reasoning")


class ForcedOpenings(Base):
    def test_openings(self):
        one, two = [{"name": "a"}], [{"name": "a"}, {"name": "b"}]
        self.assertEqual(forced_call("required", one), "<tool_call>a")
        self.assertEqual(forced_call("required", two), "<tool_call>")
        self.assertEqual(forced_call({"type": "any"}, two), "<tool_call>")
        self.assertEqual(forced_call({"type": "function", "function": {"name": "b"}}, two), "<tool_call>b")
        self.assertEqual(forced_call({"type": "tool", "name": "b"}, two), "<tool_call>b")
        self.assertIsNone(forced_call("auto", one))
        self.assertIsNone(forced_call("none", one))
        self.assertIsNone(forced_call({"type": "tool", "name": "c"}, two))
        self.assertIsNone(forced_call("required", None))

    def test_the_model_goes_on_from_the_opening(self):
        rest = "<arg_key>path</arg_key><arg_value>/a</arg_value></tool_call>"
        self.assertEqual(self.calls(rest, thinking=False, force=forced_call({"type": "tool", "name": "read_file"},
                                                                             TOOLS)),
                         [("read_file", {"path": "/a"})])
        self.assertEqual(self.calls("read_file" + rest, thinking=False, force=forced_call("required", TOOLS)),
                         [("read_file", {"path": "/a"})])

    def test_a_name_the_model_extends_is_not_a_call(self):
        self.assertEqual(self.parse("x<arg_key>k</arg_key>", thinking=False, force="<tool_call>read_file"),
                         [("content", "<tool_call>read_filex<arg_key>k</arg_key>")])


class Recover(Base):
    """#2037 and recover mode: a declared tool's call written as a JSON object inside <tool_call> is the call."""
    PAYLOAD = '{"name": "write_file", "arguments": {"content": "hello `world`"}}'

    def test_json_bodies(self):
        for fence in ("", "```json\n", "```\n", "```JSON\r\n"):
            raw = f"<tool_call>\n{fence}{self.PAYLOAD}\n{'```' if fence else ''}\n</tool_call>"
            for st in (False, True):
                with self.subTest(fence=fence, stream_tools=st):
                    self.assertEqual(self.parse("Saving." + raw + "\nDone", thinking=False, recover=True,
                                                stream_tools=st),
                                     [("content", "Saving."), ("tool_call", "write_file", {"content": "hello `world`"}),
                                      ("content", "Done")])

    def test_other_fences_and_prose_stay_text(self):
        raw = "<tool_call>\n```python\nx = 1\n```\n</tool_call>"
        self.assertEqual(self.parse(raw, thinking=False, recover=True), [("content", raw)])
        raw = f"Here:\n```json\n{self.PAYLOAD}\n```\n"
        self.assertEqual(self.parse(raw, thinking=False, recover=True), [("content", raw)])

    def test_off_without_recovery_or_for_an_undeclared_tool(self):
        raw = f"<tool_call>{self.PAYLOAD}</tool_call>"
        self.assertEqual(self.parse(raw, thinking=False), [("content", raw)])
        self.assertEqual(self.parse(raw, thinking=False, recover=True, tools=[{"name": "other"}]),
                         [("content", raw)])

    def test_the_glm_form_still_works(self):
        self.assertEqual(self.calls(READ, thinking=False, recover=True), [("read_file", READ_ARGS)])


class RandomReplies(unittest.TestCase):
    """Random replies built from GLM's tags, names, code markers and text: every way of feeding one gives the same
    events, the contract holds, and the streamed and plain parsers agree on the calls and the text."""
    PARTS = ("Hello", " ", "\n", "\n\n", "`", "```", "~~~", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>",
             "<arg_value>", "</arg_value>", "</think>", "<think>", "read_file", "write_file", "ping", "path", "content",
             "lines", "42", '{"a": [1, 2]}', "x y", "<", ">", "<tool", "</arg_v", "é\U0001F642", '"q"', "\\",
             READ, "<tool_call>ping</tool_call>")

    def test_random_replies(self):
        rng = random.Random(20261018)
        for n in range(300):
            text = "".join(rng.choice(self.PARTS) for _ in range(rng.randint(1, 30)))
            kw = dict(thinking=rng.random() < 0.5, tools=rng.choice((TOOLS, None)), recover=rng.random() < 0.3,
                      finish=rng.choice(("stop", "length")))
            with self.subTest(n=n, text=text, **kw):
                results = []
                for pieces in chunkings(text, seed=n, count=4):
                    events, _ = feed(pieces, stream_tools=True, **kw)
                    check_contract(self, events)
                    results.append(normal(events))
                for r in results[1:]:
                    self.assertEqual(r, results[0])
                plain = normal(feed([text], stream_tools=False, **kw)[0])
                streamed = [e for e in results[0] if e[0] not in ("tool_start", "tool_args")]
                if sum(e[0] == "tool_start" for e in results[0]) == sum(e[0] == "tool_call" for e in plain):
                    self.assertEqual(plain, streamed)     # no call cut off: the same calls and text either way


class Pieces(unittest.TestCase):
    def test_parse_tool_call(self):
        call = parse_tool_call("read_file<arg_key>path</arg_key><arg_value>7</arg_value>", TOOLS)
        self.assertEqual((call.name, call.arguments), ("read_file", {"path": "7"}))
        call = parse_tool_call("anything<arg_key>path</arg_key><arg_value>7</arg_value>")
        self.assertEqual((call.name, call.arguments), ("anything", {"path": 7}))
        for bad in ("other<arg_key>a</arg_key><arg_value>1</arg_value>", "read_file<arg_key>a</arg_key>",
                    "read_file</tool_call>x", ""):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    parse_tool_call(bad, TOOLS)

    def test_value_end(self):
        self.assertEqual(value_end("abc</arg_value></tool_call>"), (3, "end"))
        self.assertEqual(value_end("abc</arg_value>\n<arg_key>"), (3, "end"))
        self.assertEqual(value_end("abc</arg_value> "), (3, "wait"))
        self.assertEqual(value_end("abc</arg_value></tool"), (3, "wait"))
        self.assertEqual(value_end("a</arg_value>b</arg_value></tool_call>"), (14, "end"))
        self.assertEqual(value_end("abc</arg_val")[1], "open")

    def test_reader_pieces(self):
        r = CallReader({t["name"]: t for t in TOOLS})
        pieces = []
        for ch in "write_file<arg_key>content</arg_key><arg_value>a\"b</arg_value></tool_call>tail":
            pieces += r.feed(ch)
        self.assertEqual(pieces, ["{", '"content":', '"', "a", '\\"', "b", '"', "}"])
        self.assertEqual((r.stage, r.text, r.args), ("done", "tail", {"content": 'a"b'}))


if __name__ == "__main__":
    unittest.main()
