"""tools/glm53_selftest.sh and tools/glm53_gpu_check.sh on cards another program fills, without a GPU or llama.cpp: the
scripts run from a temporary folder beside the real glm53_compare.py, with stubs for llama.cpp's tools, the engines,
nvidia-smi, glm53_tiny_variant.py and glm53_park_check.py. A stub that needs a GPU dies the way napster-ai's run did
(2026-10-20) when the cards are full and not hidden from it.

    python -m unittest tools.test_glm53_checks
"""
from __future__ import annotations

import os
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

# logits in the layout glm53_compare.py reads: 40 rows of 8, the same from every engine
WRITE_LOGITS = """import struct, sys
with open(sys.argv[1], "wb") as f:
    f.write(struct.pack("<ii", 8, 40))
    for r in range(40):
        f.write(struct.pack("<8f", *[float((r * 7 + i * 3) % 11) for i in range(8)]))
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
                               'fi\npython3 "$FAKE/write_logits.py" "$out"\n',
    "sb/strata-glm-ref": 'python3 "$FAKE/write_logits.py" "$out"\n',
    "sb/strata-glm": 'if gpu_full; then echo "strata-glm: cudaSetDevice: out of memory" >&2; exit 1; fi\n'
                     'case " $* " in\n'
                     '  *" --prefill "*) echo "check-prefill: top-1 agreement 5/5: PASS";;\n'
                     '  *) python3 "$FAKE/write_logits.py" "$out"\n'
                     '     echo "GPU 0: layers 0-1, 1.21 GB weights and buffers"\n'
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
        env = {k: v for k, v in os.environ.items() if not k.startswith(("CUDA_", "FAKE"))}
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
        self.assertEqual(out.count("check-prefill:"), len(GPU_MODELS), out)
        self.assertIn("parking: 6 turns, all PASS", out)
        self.assertNotIn("Traceback", out)

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
        for line in self.calls("strata-glm") + self.calls("llama-strata-ref"):
            self.assertIn("CVD=<1,2> ORDER=<PCI_BUS_ID>", line)

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


if __name__ == "__main__":
    unittest.main()
