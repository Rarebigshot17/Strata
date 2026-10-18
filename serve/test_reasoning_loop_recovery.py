"""CPU-only regression for opt-in reasoning-loop handling (#728, "reasoning_loop_recovery": "stop"): a reply whose
thinking repeats whole passages is ended there as "length".

    python -m unittest serve.test_reasoning_loop_recovery
"""
import threading
import unittest

from serve.glm_fixtures import GlmTokenizer
from serve.server import MockEngine, Service, reasoning_repeat_coverage


class Tokenizer(GlmTokenizer):
    """With token_bytes, so the Detokenizer decodes each token once (the loops below are long)."""
    def token_bytes(self, token):
        return self.SPECIALS[token - 256].encode() if token >= 256 else bytes([token])


class RecordingEngine(MockEngine):
    def __init__(self, tokenizer, scripts):
        super().__init__(tokenizer, scripts, max_context=100000)
        self.calls = []
        self.sampling_calls = []

    def generate(self, ids, max_new, sampling, cancel, checkpoint=None):
        self.calls.append((list(ids), max_new))
        self.sampling_calls.append(dict(sampling or {}))
        yield from super().generate(ids, max_new, sampling, cancel, checkpoint=checkpoint)


class ReasoningLoopRecovery(unittest.TestCase):
    def setUp(self):
        self.tok = Tokenizer()
        # GLM's prompt for one user turn, with the thinking opened by the generation prompt
        self.ids = self.tok.encode('[gMASK]<sop><|user|>Solve this unchanged task<|assistant|><think>',
                                   parse_special=True)
        self.loop = 'Need maybe Agent support agent seven b. ' * 700

    def service(self, scripts, mode="stop"):
        engine = RecordingEngine(self.tok, scripts)
        service = Service(engine, self.tok, None)
        service.reasoning_loop_recovery = mode
        return engine, service

    def test_repetition_and_novel_work(self):
        self.assertEqual(reasoning_repeat_coverage(' '.join(str(i) for i in range(4000))), 0)
        self.assertGreater(reasoning_repeat_coverage(self.loop), .9)

    def test_a_loop_written_as_one_long_word(self):
        """#1753: a 24k-digit string cycling an 11-digit pattern is one word to the split and never one token twice."""
        self.assertEqual(reasoning_repeat_coverage('Counting the digits of 9^104. ' + '58747693387' * 3000), 1)
        counted = 'Digits: ' + ''.join(str(i) for i in range(1000))     # 2,893 chars that never repeat
        self.assertEqual(reasoning_repeat_coverage(counted), 0)
        self.assertEqual(reasoning_repeat_coverage('58747693387' * 100), 0)   # a tail too short to judge

    def test_stop_mode_ends_the_reply_at_the_loop(self):
        engine, service = self.service([self.loop])
        request = {'temperature': .2, 'top_p': .5, 'top_k': 5, 'seed': 2}
        events = list(service.run(self.ids, True, [], 40000, request, threading.Event()))
        self.assertEqual(len(engine.calls), 1)                   # no second pass
        self.assertEqual(events[-1][1]['finish'], 'length')
        self.assertLess(events[-1][1]['completion_tokens'], 20000)
        self.assertEqual(engine.sampling_calls[0], {**engine.sampling_calls[0], **request})   # the client's sampler
        self.assertEqual(request, {'temperature': .2, 'top_p': .5, 'top_k': 5, 'seed': 2})    # not mutated

    def test_stop_mode_ends_a_loop_written_as_one_long_word(self):
        """#1753: the word count never sees the digits, so only the tail's period ends the reply."""
        engine, service = self.service(['9^104 = ' + '58747693387' * 3000])
        events = list(service.run(self.ids, True, [], 40000, {}, threading.Event()))
        self.assertEqual(len(engine.calls), 1)                   # no second pass
        self.assertEqual(events[-1][1]['finish'], 'length')
        self.assertLess(events[-1][1]['completion_tokens'], 5000)

    def test_true_is_stop(self):
        engine, service = self.service([self.loop], mode=True)
        events = list(service.run(self.ids, True, [], 40000, {}, threading.Event()))
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(events[-1][1]['finish'], 'length')
        self.assertLess(events[-1][1]['completion_tokens'], 20000)

    def test_disabled_is_one_pass(self):
        engine, service = self.service([self.loop, 'Unused'], mode=False)
        events = list(service.run(self.ids, True, [], 40000, {}, threading.Event()))
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(events[-1][1]['finish'], 'stop')       # the whole loop, then the end of the turn
        self.assertEqual(events[-1][1]['completion_tokens'], len(self.tok.encode(self.loop)) + 1)
        self.assertEqual(service.history[-1]['reasoning_repeat_coverage'], 0)

    def test_default_single_token_guard_does_not_detect_passage_loop(self):
        engine = RecordingEngine(self.tok, [self.loop])
        service = Service(engine, self.tok, None)
        events = list(service.run(self.ids, True, [], 16384, {}, threading.Event()))
        self.assertEqual(service.repeat_stop_tokens, 256)
        self.assertEqual(events[-1][1]['finish'], 'length')
        self.assertEqual(events[-1][1]['completion_tokens'], 16384)
        self.assertFalse(any(e.kind == 'content' for kind, e in events if kind == 'event'))

    def test_repeated_final_text_is_not_a_reasoning_loop(self):
        engine, service = self.service([self.loop])
        events = list(service.run(self.ids, False, [], 40000, {}, threading.Event()))
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(events[-1][1]['finish'], 'stop')
        content = ''.join(e.text or '' for kind, e in events if kind == 'event' and e.kind == 'content')
        self.assertEqual(content, self.loop)

    def test_a_loop_after_the_thinking_is_not_stopped(self):
        engine, service = self.service(['Short thought.</think>' + self.loop])
        events = list(service.run(self.ids, True, [], 40000, {}, threading.Event()))
        self.assertEqual(events[-1][1]['finish'], 'stop')
        content = ''.join(e.text or '' for kind, e in events if kind == 'event' and e.kind == 'content')
        self.assertEqual(content, self.loop)

    def test_cancel_during_reasoning_ends_as_cancel(self):
        engine, service = self.service([self.loop, 'Unused'])
        cancel = threading.Event()
        events = []
        for event in service.run(self.ids, True, [], 40000, {}, cancel):
            events.append(event)
            if event[0] == 'event':
                cancel.set()
        self.assertEqual(len(engine.calls), 1)
        self.assertEqual(events[-1][1]['finish'], 'cancel')

    def test_status_and_history_record_the_stop(self):
        engine, service = self.service([self.loop, 'Done.</think>Answer.'])
        events = list(service.run(self.ids, True, [], 40000, {}, threading.Event()))
        self.assertEqual(len(engine.calls), 1)                   # the second script is never asked for
        self.assertEqual(events[-1][1]['finish'], 'length')
        self.assertEqual(service.history[-1]['finish'], 'length')
        self.assertGreaterEqual(service.history[-1]['reasoning_repeat_coverage'], .25)
        self.assertEqual(service.history[-1]['output_tokens'], events[-1][1]['completion_tokens'])
        self.assertFalse(service.status['busy'])


if __name__ == '__main__':
    unittest.main()
