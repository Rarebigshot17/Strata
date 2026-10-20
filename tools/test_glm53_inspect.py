"""tools/glm53_inspect.py without a real GGUF: the values the port assumes (the indexer's pool size read from the file,
the indexer keys the real file lacks and what is used then), the MTP layer's tensors counted apart from the layers
Strata runs, and the memory section with this machine's RAM beside the plan's 192 GB.  A small GGUF header written
with tools/gguf_writer.py stands in for the model; the reader never needs tensor data.

    python -m unittest tools.test_glm53_inspect
"""
from __future__ import annotations

import contextlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import glm53_inspect as IN  # noqa: E402
import gguf_writer as GW  # noqa: E402
from gguf_reader import TensorInfo  # noqa: E402

A = "glm5-next"
TYPE_IDS = {"Q2_K": 10, "Q3_K": 11, "IQ3_S": 21, "IQ2_S": 22}


def tensor(name: str, type_name: str, shape=(256, 2, 2)) -> TensorInfo:
    return TensorInfo(name, list(shape), TYPE_IDS.get(type_name, 0), type_name, 0)


def header(meta: dict) -> dict:
    """The keys of the real file the checks below look at, under the architecture's prefix."""
    return {f"{A}.{k}": v for k, v in meta.items()}


def real_file_meta(**over) -> dict:
    """What the inspector printed for the real 3-bit file on the owner's PC: 46 blocks with one MTP layer, 4-token
    pools, and neither attention.indexer.types nor attention.indexer.kpool_select_tail."""
    m = {"block_count": 46, "nextn_predict_layers": 1, "attention.indexer.kpool": 4, "attention.indexer.top_k": 2048,
         "attention.indexer.key_length": 128, "attention.kv_lora_rank": 512,
         "attention.head_count_kv": [1 if i % 4 == 3 else 0 for i in range(46)]}
    m.update(over)
    return header(m)


def write_model(path: Path, mtp_type: str = "Q2_K", trunk_type: str = "IQ2_S") -> Path:
    """Three blocks, the last the MTP layer (nextn_predict_layers 1): an MLA layer, a KDA layer, the MTP layer."""
    w = GW.GGUFWriter()
    w.add("general.architecture", A)
    for k, v in (("block_count", 3), ("nextn_predict_layers", 1), ("attention.indexer.kpool", 4),
                 ("attention.indexer.top_k", 2048), ("attention.indexer.key_length", 128),
                 ("attention.kv_lora_rank", 512)):
        w.add(f"{A}.{k}", v, "u32")
    w.add(f"{A}.attention.head_count_kv", [1, 0, 1], "array:i32")
    w.add_f32("blk.0.attn_kv_a_mqa.weight", np.zeros((4, 8)))
    w.add_f32("blk.1.ssm_a", np.zeros(8))
    w.add_f32("blk.2.nextn.eh_proj.weight", np.zeros((4, 8)))
    for name, ty in (("blk.0.ffn_gate_exps.weight", trunk_type), ("blk.1.ffn_down_exps.weight", "IQ3_S"),
                     ("blk.2.ffn_gate_exps.weight", mtp_type), ("blk.2.ffn_down_exps.weight", "Q3_K")):
        w.tensors.append(GW._Tensor(name, [256, 2, 2], ty, bytes(tensor(name, ty).expected_bytes())))
    with mock.patch.dict(GW.TYPE_IDS, TYPE_IDS):
        return w.write(path)


def run(path: Path, ram: float = 91.0) -> str:
    out = io.StringIO()
    with mock.patch.object(sys, "argv", ["glm53_inspect.py", str(path)]), \
            mock.patch.object(IN, "ram_gib", return_value=ram, create=True), contextlib.redirect_stdout(out):
        code = IN.main()
    assert code == 0, out.getvalue()
    return out.getvalue()


def section(text: str, n: int) -> str:
    """The report's section n, from its "== n." heading to the next heading."""
    return re.search(rf"^== {n}\..*?(?=^== |\Z)", text, re.M | re.S).group(0)


def line_of(text: str, needle: str) -> str:
    lines = [ln for ln in text.splitlines() if needle in ln]
    assert len(lines) == 1, (needle, lines)
    return lines[0]


class TheFile(unittest.TestCase):
    """The whole report on a small header shaped like the real file."""

    def test_report(self):
        with tempfile.TemporaryDirectory() as d:
            out = run(write_model(Path(d) / "m.gguf"))
        # the pool size is the file's, which the engine reads: not a difference from the port
        values = section(out, 2)
        kpool = line_of(values, f"{A}.attention.indexer.kpool ")
        self.assertTrue(kpool.lstrip().startswith("INFO"), kpool)
        self.assertIn("= 4", kpool)
        self.assertIn("the engine uses the file's value", kpool)
        # the two keys the real file lacks: what the engine and llama.cpp use then
        tail = line_of(values, "kpool_select_tail")
        self.assertTrue(tail.lstrip().startswith("MISSING"), tail)
        self.assertIn("the engine and llama.cpp use true", tail)
        types = line_of(values, "attention.indexer.types")
        self.assertIn("every layer runs its own (full) indexer", types)
        # Q2_K and Q3_K only in the MTP layer: counted apart, and no type flagged
        self.assertIn("MTP layer (blk.2), not loaded", out)
        self.assertNotIn("Strata kernels: NO", out)
        self.assertIn("unsupported expert types: none", out)
        self.assertIn("RAM left of 192 GB", out)
        self.assertIn("this machine: 91 GiB of RAM", out)
        # one MLA layer runs (blk.2 is the MTP layer): its latent and its pooled keys
        self.assertIn("VRAM per token of context: 1152 bytes", out)

    def test_a_trunk_layer_with_a_type_the_kernels_lack_is_flagged(self):
        with tempfile.TemporaryDirectory() as d:
            out = run(write_model(Path(d) / "m.gguf", trunk_type="Q2_K"))
        flagged = [ln for ln in out.splitlines() if "Strata kernels: NO" in ln]
        self.assertEqual(len(flagged), 1, out)
        self.assertIn("Q2_K", flagged[0])
        self.assertIn("unsupported expert types: Q2_K", out)


class AssumedValues(unittest.TestCase):
    def test_kpool_is_reported_as_the_engine_uses_it(self):
        for kpool in (4, 16):
            lines, differs = IN.check_values(real_file_meta(**{"attention.indexer.kpool": kpool}), f"{A}.")
            line = next(ln for ln in lines if ".kpool " in ln)
            self.assertTrue(line.lstrip().startswith("INFO"), line)
            self.assertIn(f"= {kpool}", line)
            self.assertEqual(differs, 0, lines)

    def test_missing_keys_say_what_is_used(self):
        lines, _ = IN.check_values(real_file_meta(), f"{A}.")
        tail = next(ln for ln in lines if "kpool_select_tail" in ln)
        self.assertIn("MISSING", tail)
        self.assertIn("the engine and llama.cpp use true", tail)
        types = next(ln for ln in lines if "indexer.types" in ln)
        self.assertIn("MISSING", types)
        self.assertIn("every layer runs its own (full) indexer", types)

    def test_a_present_key_is_checked_as_before(self):
        lines, differs = IN.check_values(real_file_meta(**{"attention.indexer.kpool_select_tail": False}), f"{A}.")
        tail = next(ln for ln in lines if "kpool_select_tail" in ln)
        self.assertTrue(tail.lstrip().startswith("DIFFERS"), tail)
        self.assertEqual(differs, 1)
        lines, differs = IN.check_values(real_file_meta(**{"attention.indexer.kpool_select_tail": True}), f"{A}.")
        self.assertTrue(next(ln for ln in lines if "kpool_select_tail" in ln).lstrip().startswith("OK"))
        self.assertEqual(differs, 0)

    def test_a_missing_key_without_a_default_stays_plain(self):
        lines, _ = IN.check_values(header({"block_count": 46}), f"{A}.")
        line = next(ln for ln in lines if ".expert_count " in ln)
        self.assertTrue(line.rstrip().endswith("(routed experts)"), line)


class QuantTypes(unittest.TestCase):
    def test_the_mtp_layer_is_counted_apart(self):
        tensors = [tensor("blk.0.ffn_gate_exps.weight", "IQ2_S"), tensor("blk.44.ffn_down_exps.weight", "IQ3_S"),
                   tensor("blk.44.attn_q_a.weight", "Q3_K"), tensor("blk.45.ffn_gate_exps.weight", "Q2_K"),
                   tensor("blk.45.nextn.eh_proj.weight", "Q3_K"), tensor("output.weight", "Q3_K")]
        sizes, unknown = IN.quant_sizes(tensors, n_trunk=45)
        self.assertEqual(unknown, 0)
        self.assertEqual(set(sizes["experts"]), {"IQ2_S", "IQ3_S"})
        self.assertEqual(set(sizes["rest"]), {"Q3_K"})
        self.assertEqual(sizes["rest"]["Q3_K"], 2 * 4 * 110)
        self.assertEqual(set(sizes["mtp"]), {"Q2_K", "Q3_K"})
        self.assertEqual(IN.unsupported_expert_types(sizes["experts"]), [])
        sizes, _ = IN.quant_sizes(tensors + [tensor("blk.3.ffn_up_exps.weight", "Q2_K")], n_trunk=45)
        self.assertEqual(IN.unsupported_expert_types(sizes["experts"]), ["Q2_K"])

    def test_trunk_layers_from_the_header(self):
        by_layer = {i: [tensor(f"blk.{i}.ffn_gate_exps.weight", "IQ2_S")] for i in range(46)}
        self.assertEqual(IN.trunk_layers(real_file_meta(), f"{A}.", by_layer), 45)
        # without the keys: the layers before the first one with nextn tensors
        by_layer[45].append(tensor("blk.45.nextn.eh_proj.weight", "Q3_K"))
        self.assertEqual(IN.trunk_layers({}, f"{A}.", by_layer), 45)
        self.assertEqual(IN.trunk_layers({}, f"{A}.", {0: [], 1: []}), 2)


class Memory(unittest.TestCase):
    def test_this_machine_beside_the_plan(self):
        text = "\n".join(IN.memory_lines(117e9, 9e9, 3.6e9, 91.0))
        self.assertIn("RAM left of 192 GB after experts and ~10 GB for Linux and services: 65.0 GB", text)
        self.assertIn("the MTP layer's 3.6 GB is not loaded", text)
        self.assertIn("this machine: 91 GiB of RAM (MemTotal)", text)
        self.assertIn("the experts do not fit", text)
        text = "\n".join(IN.memory_lines(117e9, 9e9, 0, 188.0))
        self.assertIn("about 79 GiB left beside the experts: they fit", text)
        text = "\n".join(IN.memory_lines(117e9, 9e9, 0, 125.0))
        self.assertIn("about 16 GiB left beside the experts: tight", text)
        text = "\n".join(IN.memory_lines(117e9, 9e9, 0, 0.0))
        self.assertIn("this machine: MemTotal not readable", text)

    def test_vram_per_token_from_the_file(self):
        per_token, how = IN.vram_per_token(real_file_meta(), f"{A}.", 45)
        self.assertEqual(per_token, 11 * 512 * 2 + 11 * 128 * 4 // 4)
        self.assertEqual(per_token, IN.VRAM_BYTES_PER_TOKEN)      # setup.py's figure is the real file's
        self.assertIn("11 MLA layers", how)
        # a layer with a shared indexer keeps no pooled keys of its own
        types = [1 if i in (3, 7) else 0 for i in range(46)]
        per_token, _ = IN.vram_per_token(real_file_meta(**{"attention.indexer.types": types}), f"{A}.", 45)
        self.assertEqual(per_token, 11 * 512 * 2 + 2 * 128 * 4 // 4)


if __name__ == "__main__":
    unittest.main()
