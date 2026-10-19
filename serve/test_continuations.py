"""serve/test_continuations.py - a reply made of several engine passes (the thinking budget's wrap-up, a forced call's
opening) goes on from the state the engine holds, also when the engine runs ahead of a slow reader (mock engine; no
model, no GPU).

strata-glm keeps a conversation's state only for a prompt that extends every token in it.  It prints a token, then
either ends the pass there (max_new, an end of turn, an `end=` id) or takes the token into its state, and it sees a
STOP only after the tokens it printed meanwhile.  A pass the server broke off mid-way left more in the engine's state
than the server read, and the next pass's prompt (what the server read, then the wrap-up or the opening) read the
whole thinking again.

    python -m unittest serve.test_continuations -v
"""
from __future__ import annotations

import json
import sys
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import server as S  # noqa: E402
from serve.frontend import CALL_START  # noqa: E402
from serve.glm_fixtures import GlmTokenizer, call_text, glm_template, reply_text  # noqa: E402

THOUGHT = "Let me think step by step about two plus two. " * 4          # 184 reasoning tokens (one per byte)
ANSWER = "The answer is 4."
CALL = call_text("search", q="2+2")
TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {
    "type": "object", "properties": {"q": {"type": "string"}}}}}]
WRAP = S.REASONING_WRAP_UP.split("</think>")[0]


class LiveEngine(S.MockEngine):
    """strata-glm's state, as far as passes go: the prompt and every token it printed but the last, and `lag` tokens
    printed past the last one read before a STOP is seen.  The model writes `script` after the first prompt; after a
    prompt that leaves the script, it writes `after[tail]` for the tail that prompt's text ends with.  Every GEN is
    recorded with its max_new, its end ids and whether its prompt extends the state."""

    def __init__(self, tok, script: str, after: dict[str, str], lag: int):
        super().__init__(tok, "x")
        self.reply = tok.encode(script, parse_special=True)
        self.after, self.lag = after, lag
        self.ends_turn = set(S.end_of_turn_ids(tok))
        self.first, self.live, self.gens = None, [], []

    def generate(self, ids, max_new, sampling, cancel, checkpoint=None, end_ids=None):
        if self.first is None:
            self.first = list(ids)
        self.gens.append({"max_new": max_new, "end_ids": end_ids,
                          "continues": bool(self.live) and ids[:len(self.live)] == self.live})
        made = ids[len(self.first):]
        if self.reply[:len(made)] == made:
            todo = self.reply[len(made):]
        else:
            text = self.tok.decode(made)
            for tail, more in self.after.items():
                if text.endswith(tail):
                    todo = self.tok.encode(more, parse_special=True)
                    break
            else:
                raise AssertionError(f"no reply scripted after {text[-80:]!r}")
        out = []
        for t in todo:                                  # what it prints by itself
            out.append(t)
            if len(out) >= max_new or t in self.ends_turn or t in (end_ids or ()):
                break
        given = 0
        try:
            for t in out:
                if cancel.is_set():
                    break
                given += 1
                yield t
        finally:
            printed = min(len(out), given + self.lag)  # a STOP is seen `lag` tokens on, if it is still going
            self.live = list(ids) + out[:max(0, printed - 1)]


class Continuations(unittest.TestCase):
    def start(self, script, after=None, lag=50):
        self.tok = GlmTokenizer()
        self.engine = LiveEngine(self.tok, script, after or {}, lag)
        httpd = S.serve(S.Service(self.engine, self.tok, glm_template()), port=0)
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        self.base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def chat(self, **extra) -> dict:
        """The whole answer's choice, with its calls as (name, arguments) under "calls"."""
        body = {"model": "m", "messages": [{"role": "user", "content": "2+2?"}], "max_tokens": 400, **extra}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
        choice = d["choices"][0]
        choice["calls"] = [(c["function"]["name"], json.loads(c["function"]["arguments"]))
                           for c in choice["message"].get("tool_calls") or []]
        choice["usage"] = d["usage"]
        return choice

    def ids(self, *tags):
        return sorted(t for tag in tags for t in self.tok.encode(tag, parse_special=True))

    def assertContinued(self):
        """Every pass after the first went on from the state the engine held."""
        self.assertGreater(len(self.engine.gens), 1)
        self.assertTrue(all(g["continues"] for g in self.engine.gens[1:]), self.engine.gens)

    def test_the_budget_wrap_up_goes_on_from_the_engines_state(self):
        # the pass ends by itself at the budget: the engine has nothing in its state that the wrap-up leaves out
        self.start(reply_text(THOUGHT, ANSWER) + "<|user|>", {S.REASONING_WRAP_UP: ANSWER + "<|user|>"})
        c = self.chat(reasoning_budget_tokens=20)
        self.assertEqual((c["message"]["reasoning_content"], c["message"]["content"], c["finish_reason"]),
                         (THOUGHT[:20] + WRAP, ANSWER, "stop"))
        self.assertEqual([g["max_new"] for g in self.engine.gens], [20, 400 - 20 - len(self.tok.encode(
            S.REASONING_WRAP_UP, parse_special=True))])
        self.assertContinued()

    def test_a_pass_cut_at_the_budget_goes_on_where_it_stopped(self):
        # the thinking ends before the budget, and the pass capped at it ends inside the answer: it goes on as it was
        self.start(reply_text(THOUGHT, ANSWER) + "<|user|>")
        c = self.chat(reasoning_budget_tokens=190)
        self.assertEqual((c["message"]["reasoning_content"], c["message"]["content"], c["finish_reason"]),
                         (THOUGHT, ANSWER, "stop"))
        self.assertEqual(c["usage"]["completion_tokens"], len(THOUGHT) + 1 + len(ANSWER) + 1)
        self.assertEqual([g["max_new"] for g in self.engine.gens], [190, 400 - 190])
        self.assertContinued()

    def test_the_wrap_up_waits_for_a_whole_character(self):
        # the budget's token is the first byte of "é": the thinking is wrapped up after the second, one token later
        thought = "a" * 19 + "é" + "b" * 40
        self.start(reply_text(thought, ANSWER) + "<|user|>", {S.REASONING_WRAP_UP: ANSWER + "<|user|>"})
        c = self.chat(reasoning_budget_tokens=20)
        self.assertEqual((c["message"]["reasoning_content"], c["message"]["content"]),
                         ("a" * 19 + "é" + WRAP, ANSWER))
        self.assertEqual([g["max_new"] for g in self.engine.gens][:2], [20, 1])
        self.assertContinued()

    def test_a_forced_call_opens_where_the_engine_stopped(self):
        # tool_choice "required" with thinking: the pass ends at </think>, where the server writes the opening (with
        # the name: there is one tool)
        opening = CALL_START + "search"
        self.start(reply_text(THOUGHT, ANSWER) + "<|user|>",
                   {"</think>" + opening: CALL[len(opening):] + "<|observation|>"})
        c = self.chat(tools=TOOLS, tool_choice="required")
        self.assertEqual((c["finish_reason"], c["calls"]), ("tool_calls", [("search", {"q": "2+2"})]))
        self.assertEqual(c["message"]["reasoning_content"], THOUGHT)
        self.assertEqual(self.engine.gens[0]["end_ids"], self.ids("</think>"))
        self.assertEqual(len(self.engine.gens), 2)
        self.assertContinued()

    def test_its_own_call_at_the_end_of_the_thinking_goes_on(self):
        # tool_choice "required", and the model makes its call in its thinking: the pass ended at </think> for an
        # opening it no longer needs, and goes on to the end of the turn
        self.start(reply_text("I will search." + CALL, "") + "<|observation|>")
        c = self.chat(tools=TOOLS, tool_choice="required")
        self.assertEqual((c["finish_reason"], c["calls"]), ("tool_calls", [("search", {"q": "2+2"})]))
        self.assertEqual(len(self.engine.gens), 2)
        self.assertContinued()

    def test_a_call_in_the_thinking_at_the_budget_stays_a_call(self):
        # the call's last token is the budget's: a wrap-up there would make it a quote (text after a call in the
        # thinking means it was quoted), and the call the model made would never run.  A call's tokens do not count,
        # so the pass inside it ends at its end instead of at the budget.
        self.start("x" * 19 + CALL + "\n<|observation|>", {S.REASONING_WRAP_UP: ANSWER + "<|user|>"})
        c = self.chat(tools=TOOLS, reasoning_budget_tokens=20)
        self.assertEqual((c["finish_reason"], c["calls"]), ("tool_calls", [("search", {"q": "2+2"})]))
        self.assertEqual([(g["max_new"], g["end_ids"]) for g in self.engine.gens],
                         [(20, None), (380, self.ids("</think>", "</tool_call>")), (1, None), (1, None)])
        self.assertContinued()

    def test_a_reader_that_keeps_up_changes_nothing(self):
        for lag in (0, 1):
            with self.subTest(lag=lag):
                self.start(reply_text(THOUGHT, ANSWER) + "<|user|>", {S.REASONING_WRAP_UP: ANSWER + "<|user|>"}, lag)
                c = self.chat(reasoning_budget_tokens=20)
                self.assertEqual(c["message"]["content"], ANSWER)
                self.assertContinued()


if __name__ == "__main__":
    unittest.main()
