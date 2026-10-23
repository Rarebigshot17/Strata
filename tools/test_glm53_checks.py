"""tools/glm53_selftest.sh and tools/glm53_gpu_check.sh on cards another program fills, without a GPU or llama.cpp: the
scripts run from a temporary folder beside the real glm53_compare.py, with stubs for llama.cpp's tools, the engines,
nvidia-smi, glm53_tiny_variant.py and glm53_park_check.py. A stub that needs a GPU dies the way napster-ai's run did
(2026-10-20) when the cards are full and not hidden from it. Also: the engine's Config default for a key the released
GGUF lacks is read_config's.

    python -m unittest tools.test_glm53_checks
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
try:
    import numpy  # noqa: F401  (glm53_compare.py needs it)
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

# logits in the layout glm53_compare.py reads: 40 rows of 8, the same from every engine. FAKE_TIE=1 puts row 5's
# top two tokens 0.0005 apart, in the other order in llama.cpp's (argument "ref"): a tie glm53_compare.py accepts.
# FAKE_SPREAD=1 gives the rows the scale of the GPU check's small models (the largest logit 0.32) and the engines
# other than llama.cpp the CUDA rounding measured on Q4_0 (napster-ai, 2026-10-20): the worst row off by 3.23% of its
# largest logit, and three rows of 40 whose top two tokens, 0.012 apart (more than the tie margin), swap places.
# FAKE_SPREAD=kl gives them, at the same scale, every logit 4% of the row's largest up or down instead: within the
# relative limit of quantized weights, but a mean KL far above the rounding's, as a wrong scale or a dropped term gives
WRITE_LOGITS = """import os, struct, sys
spread = os.environ.get("FAKE_SPREAD") == "1"
bug = os.environ.get("FAKE_SPREAD") == "kl"
ref = sys.argv[2:] == ["ref"]
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<ii", 8, 40))
    for r in range(40):
        row = [float((r * 7 + i * 3) % 11) for i in range(8)]
        if r == 5 and os.environ.get("FAKE_TIE") == "1":
            top = sorted(row)
            row[row.index(top[-2])] = top[-1] + (0.0005 if ref else -0.0005)
        if spread:
            row = [0.032 * v for v in row]
            hi = max(row)
            if r in (3, 17, 29):
                a, b = row.index(hi), row.index(sorted(row)[-2])
                row[b] = hi - 0.012
                if not ref:
                    row[a] -= 0.0075
                    row[b] += 0.0075
            if r == 11 and not ref:
                row[row.index(min(row))] += 0.0323 * hi
        if bug:
            row = [0.032 * v for v in row]
            if not ref:
                hi = max(row)
                row = [v + (0.04 if i % 2 else -0.04) * hi for i, v in enumerate(row)]
        f.write(struct.pack("<8f", *row))
"""

# every stub logs its name, CUDA_VISIBLE_DEVICES and CUDA_DEVICE_ORDER, then its arguments
STUB_HEAD = """#!/bin/sh
echo "$(basename "$0") CVD=<${CUDA_VISIBLE_DEVICES-unset}> ORDER=<${CUDA_DEVICE_ORDER-unset}> $*" >> "$FAKE/calls.log"
out=""; prev=""
for a in "$@"; do [ "$prev" = "--out" ] && out="$a"; prev="$a"; done
gpu_full() { [ "$FAKE_GPU_FULL" = 1 ] && [ "${CUDA_VISIBLE_DEVICES-unset}" != "" ]; }
"""

STUBS = {
    "lb/bin/test-llama-archs": 'prev=""; for a in "$@"; do [ "$prev" = "-o" ] && echo gguf > "$a/glm5-next-moe.gguf"; '
                               'prev="$a"; done\n',
    "lb/bin/llama-quantize": 'cp "$2" "$3"\n',
    "lb/bin/llama-strata-ref": 'if gpu_full || [ "$FAKE_REF_FAILS" = 1 ]; then\n'
                               '    echo "CUDA error: out of memory" >&2; echo "  in ggml_cuda_set_device" >&2\n'
                               '    exit 1\n'
                               'fi\npython3 "$FAKE/write_logits.py" "$out" ref\n',
    "sb/strata-glm-ref": 'python3 "$FAKE/write_logits.py" "$out"\n',
    "sb/strata-glm": 'if gpu_full; then echo "strata-glm: cudaSetDevice: out of memory" >&2; exit 1; fi\n'
                     'case " $* " in\n'
                     '  *" --pin-experts "*) echo "experts in page-locked RAM: 2 of 2 MoE layers, 0.1 GB" >&2\n'
                     '     echo "check-prefill: top-1 agreement 5/5: PASS";;\n'
                     '  *" --prefill "*) echo "check-prefill: top-1 agreement 5/5: PASS";;\n'
                     '  *) python3 "$FAKE/write_logits.py" "$out"\n'
                     '     echo "GPU 0: layers 0-1, 1.21 GB weights and buffers"\n'
                     # FAKE_LAYERS: that many positions where a layer's residual parts from the CPU's
                     '     i=0; while [ "$i" -lt "${FAKE_LAYERS:-0}" ]; do\n'
                     '         echo "position $i: layer 0 residual differs by 0.0172 (relative) [MLA]"; i=$((i+1))\n'
                     '     done\n'
                     '     echo "check-cpu: top-1 agreement 40/40: PASS";;\n'
                     'esac\n',
    # FAKE_SMI is what nvidia-smi prints; FAKE_SMI_FAILS makes it fail as it does without a driver
    "bin/nvidia-smi": 'if [ "$FAKE_SMI_FAILS" = 1 ]; then echo "NVIDIA-SMI has failed: no driver"; exit 9; fi\n'
                      'printf "%b" "$FAKE_SMI"\n',
}
PY_STUBS = {
    "tools/glm53_tiny_variant.py": "import os, shutil, sys\nshutil.copy(sys.argv[1], sys.argv[2])\n"
                                   "with open(os.environ['FAKE'] + '/calls.log', 'a') as f:\n"
                                   "    f.write('tiny_variant ' + ' '.join(sys.argv[1:]) + '\\n')\n",
    "tools/glm53_park_check.py": "print('parking: 6 turns, all PASS')\n",
}

MODELS = ["glm5-next-moe", "v1", "v2", "v1-bare", "q4_0"]
GPU_MODELS = ["glm5-next-moe", "v1", "v1-bare", "q4_0", "q8_0"]


@unittest.skipIf(BASH is None, "needs bash")
@unittest.skipUnless(HAVE_NUMPY, "glm53_compare.py needs numpy")
class Checks(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        (self.tmp / "tools").mkdir()
        for name in ("glm53_selftest.sh", "glm53_gpu_check.sh", "glm53_compare.py"):
            shutil.copy(ROOT / "tools" / name, self.tmp / "tools" / name)
        (self.tmp / "write_logits.py").write_text(WRITE_LOGITS, encoding="utf-8")
        for rel, body in STUBS.items():
            p = self.tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(STUB_HEAD + body, encoding="utf-8")
            p.chmod(0o755)
        for rel, body in PY_STUBS.items():
            (self.tmp / rel).write_text(body, encoding="utf-8")
        (self.tmp / "calls.log").write_text("")

    def run_script(self, name, **env_extra):
        # nothing from the caller's shell that the scripts read: SEED=N would fix test-llama-archs' seed
        env = {k: v for k, v in os.environ.items() if not k.startswith(("CUDA_", "FAKE")) and k != "SEED"}
        env.update(FAKE=str(self.tmp), PATH=f"{self.tmp / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}")
        env.update(env_extra)
        r = subprocess.run([BASH, str(self.tmp / "tools" / name), str(self.tmp / "lb"), str(self.tmp / "sb")], env=env,
                           capture_output=True, text=True, timeout=120)
        return r.returncode, r.stdout + r.stderr

    def calls(self, prog):
        return [ln for ln in (self.tmp / "calls.log").read_text().splitlines() if ln.startswith(prog + " ")]

    # ---- glm53_selftest.sh

    def test_selftest_runs_with_the_gpus_hidden_when_they_are_full(self):
        rc, out = self.run_script("glm53_selftest.sh", FAKE_GPU_FULL="1")
        self.assertEqual(rc, 0, out)
        for m in MODELS:
            self.assertIn(f"PASS {m}: rows 40   mean KL 0.00000", out)
        for prog in ("test-llama-archs", "llama-quantize", "llama-strata-ref", "strata-glm-ref"):
            self.assertTrue(self.calls(prog), prog)
            for line in self.calls(prog):
                self.assertIn("CVD=<>", line)

    def test_selftest_runs_llama_cpp_with_an_f32_cache(self):
        # Strata keeps the indexer's keys in fp32; llama.cpp keeps them in its cache type
        rc, out = self.run_script("glm53_selftest.sh")
        self.assertEqual(rc, 0, out)
        calls = self.calls("llama-strata-ref")
        self.assertEqual(len(calls), len(MODELS))
        for line in calls:
            self.assertIn(" -ctk f32 -ctv f32 ", line)
            self.assertIn(" -fa off ", line)

    def test_selftest_counts_two_nearly_equal_tokens_as_a_tie(self):
        rc, out = self.run_script("glm53_selftest.sh", FAKE_TIE="1")
        self.assertEqual(rc, 0, out)
        for m in MODELS:
            self.assertIn(f"PASS {m}: rows 40   mean KL 0.00000", out)
        self.assertEqual(out.count("top-1 agreement 97.50% (100.00% with 1 tie)"), len(MODELS), out)

    def test_selftest_takes_the_seed_to_run_again(self):
        self.run_script("glm53_selftest.sh", SEED="17")
        self.assertIn(" -s 17 ", self.calls("test-llama-archs")[0])
        (self.tmp / "calls.log").write_text("")
        self.run_script("glm53_selftest.sh")
        self.assertNotIn(" -s ", self.calls("test-llama-archs")[0])

    def test_selftest_writes_v1_bare_without_the_two_keys(self):
        self.run_script("glm53_selftest.sh")
        bare = [ln for ln in self.calls("tiny_variant") if "v1-bare.gguf" in ln]
        self.assertEqual(len(bare), 1)
        self.assertIn(" -glm5-next.attention.indexer.kpool_select_tail", bare[0])
        self.assertIn(" -glm5-next.attention.indexer.types", bare[0])

    # ---- glm53_gpu_check.sh

    def test_gpu_check_passes_on_free_cards(self):
        rc, out = self.run_script("glm53_gpu_check.sh", FAKE_SMI="0, 23900\\n1, 11800\\n2, 9800\\n")
        self.assertEqual(rc, 0, out)
        self.assertEqual(out.count("vs llama.cpp: rows 40"), len(GPU_MODELS), out)
        # each model's runs: decode, the prompt path, it 6 rows at a time, both with the latent cache in RAM, and
        # the prompt path with the experts page-locked, on one card and on every card
        for line in ("prefill: check-prefill: top-1 agreement 5/5: PASS",
                     "prefill, 6 rows at a time: check-prefill: top-1 agreement 5/5: PASS",
                     "latent cache in RAM: check-cpu: top-1 agreement 40/40: PASS",
                     "prefill, latent cache in RAM: check-prefill: top-1 agreement 5/5: PASS",
                     "prefill, experts page-locked: experts in page-locked RAM: 2 of 2 MoE layers, 0.1 GB; "
                     "check-prefill: top-1 agreement 5/5: PASS",
                     "prefill, experts on every card: check-prefill: top-1 agreement 5/5: PASS"):
            found = re.findall("^" + re.escape(line) + "$", out, re.M)
            self.assertEqual(len(found), len(GPU_MODELS), line + "\n" + out)
        self.assertEqual(out.count("check-prefill:"), 5 * len(GPU_MODELS), out)
        self.assertIn("parking: 6 turns, all PASS", out)
        self.assertNotIn("Traceback", out)

    def test_gpu_check_shows_each_models_result_after_many_layer_lines(self):
        # strata-glm prints a line for every position where a layer parts from the CPU; the check-cpu line after
        # them is the one that says PASS or FAIL, and must not be cut off with them
        rc, out = self.run_script("glm53_gpu_check.sh", FAKE_SMI="0, 23900\\n", FAKE_LAYERS="20")
        self.assertEqual(rc, 0, out)
        decode = re.findall("^check-cpu: top-1 agreement 40/40: PASS$", out, re.M)
        self.assertEqual(len(decode), len(GPU_MODELS), out)
        self.assertEqual(out.count("residual differs"), 4 * len(GPU_MODELS), out)
        self.assertEqual(out.count("(16 more positions where a layer differs)"), len(GPU_MODELS), out)

    def test_gpu_check_stops_with_one_line_per_full_card(self):
        rc, out = self.run_script("glm53_gpu_check.sh", FAKE_GPU_FULL="1", FAKE_SMI="0, 812\\n1, 11800\\n2, 301\\n")
        self.assertNotEqual(rc, 0)
        lines = out.strip().splitlines()
        self.assertEqual(len(lines), 2, out)
        self.assertIn("GPU 0 has 812 MiB free", lines[0])
        self.assertIn("GPU 2 has 301 MiB free", lines[1])
        for ln in lines:
            self.assertIn("stop the program that holds it", ln)
        self.assertEqual(self.calls("strata-glm"), [])
        self.assertEqual(self.calls("llama-strata-ref"), [])

    def test_gpu_check_reads_only_the_cards_cuda_visible_devices_names(self):
        rc, out = self.run_script("glm53_gpu_check.sh", CUDA_VISIBLE_DEVICES="1,2", FAKE_SMI="1, 11800\\n2, 9800\\n")
        self.assertEqual(rc, 0, out)
        self.assertIn("-i 1,2 ", self.calls("nvidia-smi")[0])
        # CUDA numbers the cards as nvidia-smi does, so both mean the same cards
        for line in self.calls("strata-glm"):
            self.assertIn("CVD=<1,2> ORDER=<PCI_BUS_ID>", line)
        # llama.cpp's reference runs on the CPU (test_gpu_check_runs_llama_cpp_on_the_cpu_with_an_f32_cache)
        for line in self.calls("llama-strata-ref"):
            self.assertIn("CVD=<>", line)

    def test_gpu_check_runs_llama_cpp_on_the_cpu_with_an_f32_cache(self):
        rc, out = self.run_script("glm53_gpu_check.sh", FAKE_SMI="0, 23900\\n", FAKE_TIE="1")
        self.assertEqual(rc, 0, out)
        calls = self.calls("llama-strata-ref")
        self.assertEqual(len(calls), len(GPU_MODELS))
        for line in calls:
            self.assertIn("CVD=<>", line)
            self.assertIn(" -ctk f32 -ctv f32 ", line)
            self.assertIn(" -fa off ", line)
        for line in self.calls("strata-glm"):
            self.assertIn("CVD=<unset>", line)
        self.assertEqual(out.count("(100.00% with 1 tie)"), len(GPU_MODELS), out)

    def vs_llama(self, out):
        """{model: its "vs llama.cpp" line} from the GPU check's output."""
        lines, model = {}, None
        for ln in out.splitlines():
            if ln.startswith("== "):
                model = ln[3:].strip()
            elif ln.startswith("vs llama.cpp:") or ln.startswith("FAIL vs llama.cpp"):
                lines[model] = ln
        return lines

    def test_gpu_check_takes_the_cuda_rounding_on_quantized_weights_only(self):
        # top-1 37/40 without ties fails glm53_compare.py's defaults (98%); the GPU check holds the comparison to
        # strata-glm's own rule (check.hpp): KL and the worst relative diff, by the weights' type
        rc, out = self.run_script("glm53_gpu_check.sh", FAKE_SMI="0, 23900\\n", FAKE_SPREAD="1")
        lines = self.vs_llama(out)
        self.assertEqual(sorted(lines), sorted(GPU_MODELS), out)
        for m in ("q4_0", "q8_0"):
            self.assertIn("top-1 agreement 92.50%", lines[m])
            self.assertNotIn("tie", lines[m])
            self.assertTrue(lines[m].rstrip().endswith("PASS"), lines[m])
        # float weights differ by 0.0001 on the GPU: the same spread there is a bug
        for m in ("glm5-next-moe", "v1", "v1-bare"):
            self.assertIn("FAIL", lines[m])
            self.assertIn("worst relative diff <= 0.005", lines[m])
        self.assertNotEqual(rc, 0)
        self.assertNotIn("Traceback", out)

    def test_gpu_check_fails_a_kl_far_above_the_cuda_rounding(self):
        # the relative diff of a wrong scale or a dropped term can stay within the quantized weights' 0.05; the mean
        # KL limit (1e-5, glm53_compare.py's default is 0.01) is what catches it
        rc, out = self.run_script("glm53_gpu_check.sh", FAKE_SMI="0, 23900\\n", FAKE_SPREAD="kl")
        lines = self.vs_llama(out)
        self.assertEqual(sorted(lines), sorted(GPU_MODELS), out)
        for m in ("q4_0", "q8_0"):
            self.assertIn("worst relative diff 0.04 ", lines[m])
            self.assertIn("FAIL (needs mean KL <= 1e-05 and worst relative diff <= 0.05)", lines[m])
        self.assertNotEqual(rc, 0)
        self.assertNotIn("Traceback", out)

    def test_gpu_check_with_no_visible_card_says_so(self):
        rc, out = self.run_script("glm53_gpu_check.sh", CUDA_VISIBLE_DEVICES="", FAKE_SMI="0, 23900\\n")
        self.assertNotEqual(rc, 0)
        self.assertIn("CUDA_VISIBLE_DEVICES", out)
        self.assertEqual(len(out.strip().splitlines()), 1, out)
        self.assertEqual(self.calls("strata-glm"), [])

    def test_gpu_check_goes_on_when_nvidia_smi_cannot_read_the_cards(self):
        rc, out = self.run_script("glm53_gpu_check.sh", FAKE_SMI_FAILS="1")
        self.assertEqual(rc, 0, out)
        self.assertIn("nvidia-smi", out.splitlines()[0])
        self.assertEqual(out.count("vs llama.cpp: rows 40"), len(GPU_MODELS), out)

    def test_gpu_check_a_missing_reference_is_a_fail_line_not_a_traceback(self):
        rc, out = self.run_script("glm53_gpu_check.sh", FAKE_REF_FAILS="1", FAKE_SMI="0, 23900\\n")
        self.assertNotEqual(rc, 0)
        self.assertNotIn("Traceback", out)
        self.assertEqual(out.count("FAIL vs llama.cpp: llama-strata-ref wrote no logits"), len(GPU_MODELS), out)
        self.assertIn("CUDA error: out of memory", out)


class ConfigDefaults(unittest.TestCase):
    """A Config made without read_config (a test, a tool) must not disagree with what read_config does for the
    released GGUF, which lacks attention.indexer.kpool_select_tail."""

    def test_the_struct_default_of_indexer_select_tail_is_read_configs(self):
        import re
        hpp = (ROOT / "include/strata/glm/model.hpp").read_text(encoding="utf-8")
        cpp = (ROOT / "src/glm/model.cpp").read_text(encoding="utf-8")
        struct = re.search(r"\bbool indexer_select_tail = (true|false);", hpp)
        read = re.search(r'num_or\("attention\.indexer\.kpool_select_tail", ([01])\)', cpp)
        self.assertIsNotNone(struct, "model.hpp: bool indexer_select_tail = ...;")
        self.assertIsNotNone(read, "model.cpp: num_or(\"attention.indexer.kpool_select_tail\", ...)")
        self.assertEqual(struct.group(1) == "true", read.group(1) == "1")


if __name__ == "__main__":
    unittest.main()
