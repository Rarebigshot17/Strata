"""The tokenizer's piece cache (the ids are the ones the merge loop gives, cache cold, warm or switched off), its
pre-tokenizer types as llama.cpp has them, special tokens' bytes and the GGUF constructor.  The same cases were checked
against llama.cpp's llama-tokenize on GGUFs of the synthetic GLM model's vocabulary (tools/glm53_tokcheck.py checks a
real one).

    python -m unittest tools.test_strata_tokenizer
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_tokenizer as ST  # noqa: E402


def toy() -> ST.Tokenizer:
    base = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
    merges = ["t h", "th e", "i n", "a n", "an d", "Ġ t", "Ġt h", "Ġth e", "e r", "o n"]
    merges = [m.replace("Ġ", ST.BYTE_TO_UNICODE[32]) for m in merges]
    tokens = list(base)
    for m in merges:
        a, b = m.split(" ")
        tokens.append(a + b)
    return ST.Tokenizer(tokens, merges)


class PieceCache(unittest.TestCase):
    def test_same_ids_cold_warm_and_off(self):
        rng = random.Random(7)
        words = ["the", "and", "in", "on", "there", "other", "x" * 70, "你好", "def", "return", "\n\n", "  "]
        text = "".join(rng.choice(words) + rng.choice([" ", "", "\n", "\t"]) for _ in range(3000))
        a = toy()
        a.PIECE_CACHE_MAX = 0
        want = a.encode(text)
        b = toy()
        cold = b.encode(text)
        warm = b.encode(text)
        self.assertEqual(cold, want)
        self.assertEqual(warm, want)
        self.assertGreater(len(b._piece_ids), 0)
        self.assertEqual(b.decode(warm), text)

    def test_the_cache_is_bounded(self):
        t = toy()
        t.PIECE_CACHE_MAX = 5
        t.encode(" ".join("w%d" % i for i in range(100)))
        self.assertLessEqual(len(t._piece_ids), 5)


class ByteCache(unittest.TestCase):
    def test_decode_before_encode_cold_and_warm(self):
        t = toy()
        text = "café 日本語 🙂\r\n\x00"
        ids = list(text.encode("utf-8"))  # toy's first 256 token IDs are the raw bytes
        self.assertEqual(t.decode(ids, errors="strict"), text)
        self.assertEqual(t.decode(ids, errors="strict"), text)
        self.assertEqual(t.decode([]), "")
        for i in range(256):
            self.assertEqual(t.token_bytes(i), bytes([i]))

    def test_invalid_ids_before_and_after_cache_use(self):
        t = toy()
        for _ in range(2):
            for i in (-1, len(t.tokens), len(t.tokens) + 100):
                with self.assertRaises(IndexError):
                    t.token_bytes(i)
            self.assertEqual(t.token_bytes(65), b"A")

    def test_instances_with_different_vocabularies_stay_independent(self):
        tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
        a, b = ST.Tokenizer(tokens, []), ST.Tokenizer(tokens[::-1], [])
        text = "the café 日本語 🙂\n"
        for t in (a, b, a, b):
            expected = [t.ids[ST.BYTE_TO_UNICODE[byte]] for byte in text.encode("utf-8")]
            self.assertEqual(t.encode(text), expected)
            self.assertEqual(t.decode(expected, errors="strict"), text)
        self.assertEqual(a.token_bytes(0), b"\x00")
        self.assertEqual(b.token_bytes(0), b"\xff")

    def test_shared_tokenizer_concurrent_first_decode(self):
        import threading
        from concurrent.futures import ThreadPoolExecutor
        t = toy()
        texts = ["café", "日本語", "🙂\r\n", "\x00plain"]
        barrier = threading.Barrier(len(texts))
        def decode(text):
            ids = list(text.encode("utf-8"))
            barrier.wait(timeout=5)
            for _ in range(100):
                if t.decode(ids, errors="strict") != text:
                    raise AssertionError("concurrent decode changed the text")
            return text
        with ThreadPoolExecutor(len(texts)) as pool:
            self.assertEqual(list(pool.map(decode, texts)), texts)


def merge_vocab(pre: str, types: bool = True) -> ST.Tokenizer:
    """Bytes, then three merges in an order where BPE splits "xqz" as x + qz although "xqz" is a token, and the
    GLM special tokens: CONTROL (type 3) and USER_DEFINED (type 4), one of them with a space and an e-acute."""
    tokens = [ST.BYTE_TO_UNICODE[b] for b in range(256)]
    merges = ["q z", "x q", "xq z"]
    tokens += [m.replace(" ", "") for m in merges]
    specials = ["<|user|>", "<|assistant|>", "<think>", "<|bloc é|>"]
    tokens += specials
    return ST.Tokenizer(tokens, merges, [1] * (len(tokens) - 4) + [3, 3, 4, 4] if types else None, pre)


class PreTokenizers(unittest.TestCase):
    TEXT = "it's a,b!! 1234567\tx  y"

    def test_glm_patterns(self):
        t = merge_vocab("glm5")
        self.assertEqual(t.pieces(self.TEXT), ["it", "'s", " a", ",b", "!!", " ", "123", "456", "7", "\tx", " ", " y"])
        self.assertEqual(t.pre_patterns, (ST.GLM4_PATTERN,))
        self.assertEqual(merge_vocab("chatglm-bpe").pieces(self.TEXT), t.pieces(self.TEXT))

    def test_ignore_merges_only_for_glm4_and_glm5(self):
        """llama.cpp sets ignore_merges for glm4 and glm5, not for chatglm-bpe (same pattern) or default."""
        for pre, whole in (("glm4", True), ("glm5", True), ("chatglm-bpe", False), ("default", False),
                           ("qwen35", False)):
            t = merge_vocab(pre)
            self.assertEqual(t.ignore_merges, whole, pre)
            want = [t.ids["xqz"]] if whole else [t.ids["x"], t.ids["qz"]]
            self.assertEqual(t.encode("xqz"), want, pre)

    def test_default_applies_four_patterns_one_after_another(self):
        """What llama.cpp uses without tokenizer.ggml.pre: punctuation runs first (so "'s" never forms), then GPT-2's
        pattern, digit runs and groups of three; what a pattern does not match stays a piece ("\t")."""
        t = merge_vocab("default")
        self.assertEqual(t.pieces(self.TEXT),
                         ["it", "'", "s", " a", ",", "b", "!!", " ", "123", "456", "7", "\t", "x", " ", " y"])
        for text in (self.TEXT, "", "  ", "end\n\n", "€1,234.56 «q» 😀", "x" * 100):
            self.assertEqual("".join(t.pieces(text)), text)
            self.assertEqual(t.decode(t.encode(text)), text)

    def test_an_unknown_pre_is_refused(self):
        with self.assertRaises(ValueError) as c:
            merge_vocab("llama-bpe")
        self.assertIn("unknown pre-tokenizer", str(c.exception))


class SpecialTokenBytes(unittest.TestCase):
    def test_types_3_and_4_are_their_text_in_utf8(self):
        t = merge_vocab("glm5")
        i = t.ids["<|bloc é|>"]                  # a space and an e-acute: neither is a byte-mapped character
        self.assertEqual(t.token_bytes(i), "<|bloc é|>".encode("utf-8"))
        self.assertEqual(t.token_bytes(t.ids["<|user|>"]), b"<|user|>")
        ids = t.encode("a<|bloc é|>b<|user|>", parse_special=True)
        self.assertEqual(ids, [97, i, 98, t.ids["<|user|>"]])
        self.assertEqual(t.decode(ids, errors="strict"), "a<|bloc é|>b<|user|>")

    def test_a_normal_token_outside_the_byte_table_still_raises(self):
        t = merge_vocab("glm5", types=False)       # no token types: "<|bloc é|>" is an ordinary token here
        with self.assertRaises(KeyError):
            t.token_bytes(t.ids["<|bloc é|>"])


class FromGguf(unittest.TestCase):
    def write(self, path, pre):
        from gguf_writer import GGUFWriter
        t = merge_vocab("glm5")
        w = GGUFWriter()
        w.add("general.architecture", "glm-test")
        w.add("tokenizer.ggml.model", "gpt2")
        if pre is not None:
            w.add("tokenizer.ggml.pre", pre)
        w.add("tokenizer.ggml.tokens", t.tokens)
        w.add("tokenizer.ggml.merges", ["q z", "x q", "xq z"])
        w.add("tokenizer.ggml.token_type", t.token_types)
        w.add("tokenizer.ggml.eos_token_id", t.ids["<|user|>"], "u32")
        w.add("tokenizer.chat_template", "{{ messages }}")
        w.write(path)

    def test_pre_from_the_gguf_and_default_without_it(self):
        with tempfile.TemporaryDirectory() as d:
            for pre, want in (("glm5", "glm5"), ("chatglm-bpe", "chatglm-bpe"), (None, "default")):
                path = Path(d) / f"{want}.gguf"
                self.write(path, pre)
                t = ST.Tokenizer.from_gguf(path)
                self.assertEqual(t.pre, want)
                self.assertEqual(t.special_ids, {"tokenizer.ggml.eos_token_id": t.ids["<|user|>"]})
                self.assertEqual(t.encode("<|user|>xqz", parse_special=True)[0], t.ids["<|user|>"])

    def test_extract_writes_the_patterns_of_its_pre(self):
        with tempfile.TemporaryDirectory() as d:
            for pre, patterns in (("glm5", [ST.GLM4_PATTERN]), (None, list(ST.DEFAULT_PATTERNS))):
                path = Path(d) / "m.gguf"
                self.write(path, pre)
                cfg = ST.extract(path, Path(d) / "out")
                written = json.loads((Path(d) / "out" / "tokenizer" / "tokenizer.json").read_text(encoding="utf-8"))
                self.assertEqual(written, cfg)
                self.assertEqual(cfg["pre_patterns"], patterns)
                self.assertEqual(cfg["ignore_merges"], pre == "glm5")
                self.assertIn("LLAMA_VOCAB_PRE_TYPE_" + ("CHATGLM4" if pre else "DEFAULT"), cfg["pre_pattern_source"])
                self.assertEqual((Path(d) / "out" / "tokenizer" / "chat_template.jinja").read_text(encoding="utf-8"),
                                 "{{ messages }}")


if __name__ == "__main__":
    unittest.main()
