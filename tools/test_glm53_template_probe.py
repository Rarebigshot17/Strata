"""tools/glm53_template_probe.py on a small GGUF that holds GLM's special tokens and GLM-5.3-Flash's own template
(serve/chat_template.jinja, the real GGUF's): every request it shows, with the prompt the server renders for it.

    python3 tools/test_glm53_template_probe.py
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from gguf_writer import GGUFWriter  # noqa: E402
from strata_tokenizer import BYTE_TO_UNICODE  # noqa: E402

CONTROL = ["<|endoftext|>", "[MASK]", "[gMASK]", "[sMASK]", "<sop>", "<eop>", "<|system|>", "<|user|>",
           "<|assistant|>", "<|observation|>"]
USER_DEFINED = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>",
                "</arg_value>", "<tool_response>", "</tool_response>"]


def write_gguf(path: Path, template: str):
    """A GGUF with no tensors: a byte-level vocabulary, GLM's special tokens (control and user-defined) and a
    template."""
    tokens = [BYTE_TO_UNICODE[b] for b in range(256)] + CONTROL + USER_DEFINED
    w = GGUFWriter()
    w.add("general.architecture", "glm-test")
    w.add("general.name", "GLM Test")
    w.add("tokenizer.ggml.model", "gpt2")
    w.add("tokenizer.ggml.pre", "glm4")
    w.add("tokenizer.ggml.tokens", tokens)
    w.add("tokenizer.ggml.token_type", [1] * 256 + [3] * len(CONTROL) + [4] * len(USER_DEFINED))
    w.add("tokenizer.ggml.eos_token_id", tokens.index("<|endoftext|>"), "u32")
    w.add("tokenizer.chat_template", template)
    w.write(path)


class Probe(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "glm-00001-of-00004.gguf"
            write_gguf(path, (ROOT / "serve" / "chat_template.jinja").read_text(encoding="utf-8"))
            r = subprocess.run([sys.executable, str(ROOT / "tools" / "glm53_template_probe.py"), str(path)],
                               capture_output=True, text=True, timeout=120)
        cls.code, cls.out, cls.err = r.returncode, r.stdout, r.stderr
        sections = cls.out.split("\n-- ")[1:]
        cls.requests = {s.split("\n", 1)[0]: s.split("\n", 1)[1] for s in sections}

    def shown(self, title: str) -> tuple[str, str]:
        """-> (the request's summary line, the prompt it shows) for the request titled `title`."""
        self.assertIn(title, self.requests, list(self.requests))
        summary, _, prompt = self.requests[title].partition("\n")
        return summary, prompt.strip("\n")

    def test_it_runs(self):
        self.assertEqual(self.code, 0, self.err)
        self.assertNotIn("FAILED", self.out)

    def test_the_levels(self):
        for title, level, thinking in (("openai: thinking left to the model", "Max", True),
                                       ("openai: reasoning_effort none (thinking off)", "Max", False),
                                       ("openai: reasoning_effort low", "Low", True),
                                       ("openai: reasoning_effort medium", "High", True),
                                       ("openai: reasoning_effort high", "High", True),
                                       ("openai: reasoning_effort max", "Max", True),
                                       ("anthropic: thinking enabled with budget_tokens", "High", True),
                                       ("anthropic: thinking disabled", "Max", False)):
            with self.subTest(title=title):
                summary, prompt = self.shown(title)
                self.assertIn(f"thinking {thinking};", summary)
                self.assertTrue(prompt.startswith(f"[gMASK]<sop><|system|>Reasoning Effort: {level}<|"), prompt)
                self.assertTrue(prompt.endswith("<|assistant|><think>" if thinking else "<|assistant|><think></think>"),
                                prompt)

    def test_a_forced_call_with_thinking_off(self):
        summary, prompt = self.shown("openai: a forced call (tool_choice required) with thinking off")
        self.assertIn("thinking False;", summary)
        self.assertTrue(prompt.endswith("<|user|>Show me notes.txt<|assistant|><think></think><tool_call>read_file"),
                        prompt)


if __name__ == "__main__":
    unittest.main()
