"""Tests for setup.py as the GLM-5.3-Flash installer: the size chosen by RAM, the repository's listing and the shards
taken from it, the GPUs' compute capabilities and the CMake command line, the run config, the start script, the
server's command line and the port.  No network, no GPU, no compiler: nvidia-smi, the Hugging Face API, the downloads
and the build are mocked, and setup runs in a temporary Strata folder.

    python -m unittest tools.test_setup_glm
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402

# the owner's PC: an RTX 3090 Ti, a 3080 Ti and a 3080 (compute capability 8.6) and 192 GB of RAM (188 GiB in
# /proc/meminfo)
OWNER_SMI = ("0, NVIDIA GeForce RTX 3090 Ti, 24564, 8.6, 580.95.05\n"
             "1, NVIDIA GeForce RTX 3080 Ti, 12288, 8.6, 580.95.05\n"
             "2, NVIDIA GeForce RTX 3080, 10240, 8.6, 580.95.05\n")
OWNER_RAM = 188.0
GB = 10**9
REAL_GPUS = setup.gpus          # the tests below put owner_gpus in its place


def owner_gpus():
    with mock.patch.object(setup, "out", return_value=OWNER_SMI):
        return REAL_GPUS()


def entry(path, size=GB, sha="ab" * 32):
    """One file of the Hub's tree listing, as /api/models/<repo>/tree/main?recursive=true answers it."""
    e = {"type": "file", "oid": "0" * 40, "size": 134 if sha else size, "path": path}
    if sha:
        e["lfs"] = {"oid": sha, "size": size, "pointerSize": 134}
    return e


def shards(stem, n, folder="", size=GB):
    return [entry(f"{folder}{stem}-{i:05d}-of-{n:05d}.gguf", size, f"{i:02x}" * 32) for i in range(1, n + 1)]


# a repository laid out the way unsloth's are: one folder per quantization, the shards inside
LISTING = ([{"type": "directory", "oid": "1" * 40, "size": 0, "path": "UD-IQ3_XXS"},
            entry("README.md", 5000, None), entry(".gitattributes", 2000, None),
            entry("mmproj-F16.gguf", 2 * GB)]
           + shards("UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS", 3, size=40 * GB)
           + shards("IQ4_XS/GLM-5.3-Flash-IQ4_XS", 4, size=42 * GB)
           + shards("UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS", 4)
           + shards("UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL", 4))


class Response:
    """What urlopen returns, enough for hf_tree: read(), headers.get(), a context manager."""

    def __init__(self, body, link=None):
        self.body, self.headers = json.dumps(body).encode(), {"Link": link} if link else {}

    def read(self):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def hub(listing, seen=None):
    """A fake urlopen that answers the tree listing (and records the URLs it was asked for)."""
    def urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else req
        if seen is not None:
            seen.append(url)
        if "/api/models/" not in url:
            raise AssertionError(f"unexpected request {url}")
        return Response(listing)
    return urlopen


def quiet(fn, *args, **kw):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kw), buf.getvalue()


def install(argv, ram=OWNER_RAM, smi=OWNER_SMI, listing=LISTING, configs=(), env=None, free=10000.0):
    """setup.main() with `argv` in a temporary Strata folder, everything outside it mocked: nvidia-smi says `smi`, the
    PC has `ram` GiB and `free` GB of disk, the Hub lists `listing`, a download writes an empty file, the engine build
    writes build/strata-glm and the start is recorded.  Returns (exit code, output, {config name: config}, {script
    name: text}, a record of the downloads, builds, starts and Hub requests)."""
    with tempfile.TemporaryDirectory() as d:
        base = Path(d)
        root = base / "Strata"
        (root / "serve").mkdir(parents=True)
        for name, cfg in configs:
            (root / name).write_text(json.dumps(cfg), encoding="utf-8")
        rec = {"downloads": [], "builds": [], "starts": [], "urls": []}

        def download(url, dst, what=None):
            rec["downloads"].append((url, dst.relative_to(base).as_posix()))
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(b"")
            setup.mark(dst)

        def build_engine(archs, yes, llama, force=False):
            rec["builds"].append(sorted(archs))
            (root / "build").mkdir(exist_ok=True)
            (root / "build" / "strata-glm").write_bytes(b"")
            return root / "build" / "strata-glm"

        def start(cfg_path, port, gpu=None, keep=None):
            rec["starts"].append((cfg_path.name, port))
            return 0

        patches = [mock.patch.object(setup, "ROOT", root), mock.patch.object(setup, "BUILD_DIR", root / "build"),
                   mock.patch.object(setup, "out", return_value=smi),
                   mock.patch.object(setup, "ram_gb", return_value=ram),
                   mock.patch.object(setup, "cpu_info", return_value=("Test CPU", True)),
                   mock.patch.object(setup, "free_gb", return_value=free),
                   mock.patch.object(setup, "settings_path", return_value=base / "settings.json"),
                   mock.patch.object(setup, "other_installs", return_value=[]),
                   mock.patch.object(setup.urllib.request, "urlopen", hub(listing, rec["urls"])),
                   mock.patch.object(setup, "download", download),
                   mock.patch.object(setup, "verify_file", lambda s, size, sha: None),
                   mock.patch.object(setup, "check_shards", lambda s: None),
                   mock.patch.object(setup, "python_packages", lambda: None),
                   mock.patch.object(setup, "get_llama_cpp", return_value=root / "third_party" / "llama.cpp"),
                   mock.patch.object(setup, "build_engine", build_engine),
                   mock.patch.object(setup, "update_installed_engine", lambda force=False: None),
                   mock.patch.object(setup, "start", start),
                   mock.patch.object(setup.time, "sleep", lambda s: None),
                   mock.patch.object(sys, "argv", ["setup.py", *argv]),
                   mock.patch.object(sys, "stdin", io.StringIO("")),        # a question ends at EOF
                   mock.patch.dict(os.environ, env or {})]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            if not (env or {}).get("HF_ENDPOINT"):
                os.environ.pop("HF_ENDPOINT", None)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    code = setup.main()
                except SystemExit as e:
                    code = e.code
        cfgs = {p.name: json.loads(p.read_text(encoding="utf-8")) for p in root.glob("strata-*.json")}
        scripts = {p.name: p.read_text(encoding="utf-8") for p in root.glob("run-*.sh")}
        rec["root"], rec["base"] = str(root), str(base)
        return code, buf.getvalue(), cfgs, scripts, rec


class ModelByRam(unittest.TestCase):
    def test_192_gb_recommends_the_3_bit_and_allows_the_4_bit_tightly(self):
        self.assertEqual(setup.recommended_model(OWNER_RAM), "UD-IQ3_XXS")
        self.assertEqual(setup.model_fit("UD-IQ3_XXS", OWNER_RAM), "fits")
        self.assertEqual(setup.model_fit("IQ4_XS", OWNER_RAM), "tight")
        self.assertEqual(setup.park_mib("UD-IQ3_XXS", OWNER_RAM), 8192)
        self.assertEqual(setup.park_mib("IQ4_XS", OWNER_RAM), 4096)
        self.assertEqual(setup.park_mib("IQ4_XS", OWNER_RAM, 2048), 2048)      # --park-mib as given

    def test_more_ram_recommends_the_4_bit(self):
        self.assertEqual(setup.recommended_model(251.0), "IQ4_XS")            # 256 GB
        self.assertEqual(setup.park_mib("IQ4_XS", 251.0), 8192)

    def test_too_little_ram_for_either(self):
        self.assertEqual(setup.model_fit("UD-IQ3_XXS", 125.0), "no")          # 128 GB
        self.assertEqual(setup.recommended_model(125.0), "UD-IQ3_XXS")        # the smallest: nothing fits
        self.assertIn("do not fit", setup.fit_text("IQ4_XS", 125.0))

    def test_yes_installs_the_recommended_size(self):
        code, out, cfgs, _, rec = install(["--yes", "--no-start"])
        self.assertEqual(code, 0, out)
        self.assertEqual(list(cfgs), ["strata-glm-ud-iq3_xxs.json"])
        self.assertIn("(recommended)", out)
        args = cfgs["strata-glm-ud-iq3_xxs.json"]["args"]
        self.assertEqual(args[args.index("--park-mib") + 1], "8192")

    def test_the_4_bit_on_192_gb_warns_and_parks_less(self):
        code, out, cfgs, _, _ = install(["--model", "IQ4_XS", "--yes", "--no-start"])
        self.assertEqual(code, 0, out)
        self.assertIn("leaves only ~26 GB of RAM beside its experts", out)
        self.assertIn("--park-mib 4096", out)
        args = cfgs["strata-glm-iq4_xs.json"]["args"]
        self.assertEqual(args[args.index("--park-mib") + 1], "4096")

    def test_a_size_that_does_not_fit_needs_consent(self):
        code, out, cfgs, _, _ = install(["--yes", "--no-start"], ram=125.0)   # --yes alone: no consent
        self.assertEqual(code, 1)
        self.assertIn("do not fit", out)
        self.assertEqual(cfgs, {})
        code, out, cfgs, _, _ = install(["--model", "UD-IQ3_XXS", "--yes", "--no-start"], ram=125.0)
        self.assertEqual(code, 0, out)                                        # --model is the consent
        self.assertIn("installing UD-IQ3_XXS with 125 GB of RAM, as you chose", out)


class Listing(unittest.TestCase):
    def test_hf_tree_reads_sizes_and_hashes_and_follows_the_pages(self):
        pages = {"first": Response(LISTING[:6], '<https://huggingface.co/api/models/x/tree/main?cursor=2>; rel="next"'),
                 "next": Response(LISTING[6:])}
        seen = []

        def urlopen(req, timeout=None):
            seen.append(req.full_url)
            return pages["next" if "cursor=2" in req.full_url else "first"]
        with mock.patch.object(setup.urllib.request, "urlopen", urlopen), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HF_ENDPOINT", None)
            files = setup.hf_tree()
        self.assertEqual(seen[0], "https://huggingface.co/api/models/unsloth/GLM-5.3-Flash-GGUF/tree/main"
                                  "?recursive=true")
        self.assertEqual(len(seen), 2)
        self.assertNotIn("UD-IQ3_XXS", [f["path"] for f in files])           # a folder is no file
        self.assertEqual(len(files), len(LISTING) - 1)
        first = next(f for f in files if f["path"].endswith("UD-IQ3_XXS-00001-of-00003.gguf"))
        self.assertEqual(first, {"path": "UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00003.gguf",
                                 "size": 40 * GB, "sha256": "01" * 32})     # the LFS size, not the pointer's
        readme = next(f for f in files if f["path"] == "README.md")
        self.assertEqual((readme["size"], readme["sha256"]), (5000, None))

    def test_hf_endpoint_is_honoured(self):
        seen = []
        with mock.patch.object(setup.urllib.request, "urlopen", hub(LISTING, seen)), \
                mock.patch.dict(os.environ, {"HF_ENDPOINT": "https://hf-mirror.example/"}):
            setup.hf_tree()
            url = setup.hf_file_url("UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00003.gguf")
        self.assertTrue(seen[0].startswith("https://hf-mirror.example/api/models/unsloth/GLM-5.3-Flash-GGUF/tree/"))
        self.assertEqual(url, "https://hf-mirror.example/unsloth/GLM-5.3-Flash-GGUF/resolve/main/"
                              "UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00003.gguf")

    def files(self, listing=LISTING):
        return [{"path": e["path"], "size": (e.get("lfs") or e)["size"], "sha256": (e.get("lfs") or {}).get("oid")}
                for e in listing if e["type"] == "file"]

    def test_the_shards_of_the_size_in_their_folder(self):
        quant, got = setup.pick_shards(self.files(), "UD-IQ3_XXS")
        self.assertEqual(quant, "UD-IQ3_XXS")
        self.assertEqual([f["path"] for f in got],
                         [f"UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS-0000{i}-of-00003.gguf" for i in (1, 2, 3)])

    def test_iq4_xs_is_not_ud_iq4_xs(self):
        quant, got = setup.pick_shards(self.files(), "IQ4_XS")
        self.assertEqual(quant, "IQ4_XS")
        self.assertEqual(len(got), 4)
        self.assertTrue(all(f["path"].startswith("IQ4_XS/") for f in got))
        self.assertFalse(setup.quant_in("UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00004.gguf", "IQ4_XS"))
        self.assertFalse(setup.quant_in("GLM-5.3-Flash-IQ4_XS_L.gguf", "IQ4_XS"))
        self.assertTrue(setup.quant_in("GLM-5.3-Flash-iq4_xs-00001-of-00002.gguf", "IQ4_XS"))

    def test_the_4_bit_falls_back_to_ud_q4_k_xl(self):
        files = [f for f in self.files() if not f["path"].startswith("IQ4_XS/")]
        quant, got = setup.pick_shards(files, "IQ4_XS")
        self.assertEqual(quant, "UD-Q4_K_XL")
        self.assertEqual(len(got), 4)
        self.assertIsNone(setup.pick_shards([f for f in files if "Q4_K_XL" not in f["path"]], "IQ4_XS"))

    def test_files_at_the_top_and_an_unsplit_file(self):
        files = self.files(shards("GLM-5.3-Flash-UD-IQ3_XXS", 2) + [entry("GLM-5.3-Flash-IQ4_XS.gguf")])
        self.assertEqual([f["path"] for f in setup.pick_shards(files, "UD-IQ3_XXS")[1]],
                         ["GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00002.gguf", "GLM-5.3-Flash-UD-IQ3_XXS-00002-of-00002.gguf"])
        self.assertEqual([f["path"] for f in setup.pick_shards(files, "IQ4_XS")[1]], ["GLM-5.3-Flash-IQ4_XS.gguf"])

    def test_an_incomplete_set_and_mmproj_are_left_out(self):
        files = self.files(shards("UD-IQ3_XXS/GLM-5.3-Flash-UD-IQ3_XXS", 3)[:2]
                           + [entry("UD-IQ3_XXS/mmproj-UD-IQ3_XXS.gguf")])
        self.assertIsNone(setup.pick_shards(files, "UD-IQ3_XXS"))

    def test_the_download_takes_every_shard_from_the_listed_path(self):
        code, out, cfgs, _, rec = install(["--yes", "--no-start"])
        self.assertEqual(code, 0, out)
        self.assertEqual([d for d in rec["downloads"]],
                         [("https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/main/UD-IQ3_XXS/"
                           f"GLM-5.3-Flash-UD-IQ3_XXS-0000{i}-of-00003.gguf",
                           f"Strata-GLM-data/models/glm-ud-iq3_xxs/GLM-5.3-Flash-UD-IQ3_XXS-0000{i}-of-00003.gguf")
                          for i in (1, 2, 3)])
        self.assertIn("UD-IQ3_XXS: 3 files, 120 GB", out)

    def test_the_fallback_is_said_and_downloaded(self):
        listing = [e for e in LISTING if not e["path"].startswith("IQ4_XS/")]
        code, out, cfgs, _, rec = install(["--model", "IQ4_XS", "--yes", "--no-start"], ram=251.0, listing=listing)
        self.assertEqual(code, 0, out)
        self.assertIn("the repository has no IQ4_XS files: the 4-bit model is its UD-Q4_K_XL", out)
        self.assertTrue(all("/glm-ud-q4_k_xl/" in dst for _url, dst in rec["downloads"]))
        self.assertIn("strata-glm-iq4_xs.json", cfgs)                         # the config is named by the size

    def test_a_repository_without_the_size_stops(self):
        code, out, cfgs, _, rec = install(["--yes", "--no-start"], listing=[entry("README.md", 10, None)])
        self.assertEqual(code, 1)
        self.assertIn("has no UD-IQ3_XXS files", out)
        self.assertIn("--gguf-dir", out)
        self.assertEqual(rec["builds"], [])                                   # stopped before the long steps

    def test_not_enough_disk_stops_before_the_download(self):
        code, out, _, _, rec = install(["--yes", "--no-start"], free=50.0)
        self.assertEqual(code, 1)
        self.assertIn("not enough free disk space", out)
        self.assertIn("need ~121 GB", out)
        self.assertEqual((rec["downloads"], rec["builds"]), ([], []))


class Gpus(unittest.TestCase):
    def test_nvidia_smi_is_parsed(self):
        found = owner_gpus()
        self.assertEqual([g["index"] for g in found], [0, 1, 2])
        self.assertEqual([g["arch"] for g in found], ["86", "86", "86"])
        self.assertAlmostEqual(found[0]["vram_gb"], 24.0, places=0)
        self.assertEqual(setup.driver_major(found[0]), 580)
        self.assertEqual(setup.cc(found[0]), "8.6")

    def test_compute_capabilities_become_cmake_architectures(self):
        self.assertEqual(setup.cuda_archs(setup.card_archs(owner_gpus())), "86")
        mixed = [{"arch": "89"}, {"arch": "86"}, {"arch": "120"}, {"arch": ""}]
        self.assertEqual(setup.card_archs(mixed), [86, 89, 120])
        self.assertEqual(setup.cuda_archs(setup.card_archs(mixed)), "86;89;120")

    def test_an_old_or_unknown_card_cannot_be_used(self):
        with mock.patch.object(setup, "out", return_value="0, Tesla P40, 24576, 6.1, 580.1\n1, Odd, 8192, [N/A], 580\n"):
            p40, odd = setup.gpus()
        self.assertEqual(p40["arch"], "61")
        self.assertIn("7.5 or newer", setup.gpu_problem(p40))
        self.assertEqual(odd["arch"], "")
        self.assertIn("compute capability", setup.gpu_problem(odd))
        self.assertIsNone(setup.gpu_problem(owner_gpus()[0]))

    def test_a_usable_card_beside_an_old_one(self):
        smi = OWNER_SMI.splitlines()[0] + "\n1, Tesla P40, 24576, 6.1, 580.95.05\n"
        code, out, cfgs, _, rec = install(["--yes", "--no-start"], smi=smi)
        self.assertEqual(code, 0, out)
        self.assertEqual(rec["builds"], [[86]])                               # compiled for the usable card only
        self.assertEqual(cfgs["strata-glm-ud-iq3_xxs.json"]["gpu"], 0)       # and the config names it

    def test_an_old_driver_stops(self):
        code, out, _, _, _ = install(["--yes", "--no-start"], smi=OWNER_SMI.replace("580.95.05", "570.172.08"))
        self.assertEqual(code, 1)
        self.assertIn("the NVIDIA driver is too old (570.172.08; 580 or newer", out)

    def test_no_gpu_stops(self):
        code, out, _, _, _ = install(["--yes", "--no-start"], smi="")
        self.assertEqual(code, 1)
        self.assertIn("no NVIDIA GPU found", out)


class Build(unittest.TestCase):
    real_source_hash = staticmethod(setup.source_hash)

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for p in (mock.patch.object(setup, "BUILD_DIR", self.root / "build"),
                  mock.patch.object(setup, "source_hash", lambda parts=None: "hash-1"),
                  mock.patch.object(setup, "source_version", lambda: "0.2.0")):
            p.start()
            self.addCleanup(p.stop)
        nvcc = self.root / "cuda" / "bin" / "nvcc"
        nvcc.parent.mkdir(parents=True)
        nvcc.write_bytes(b"")
        (self.root / "cuda" / "lib64").mkdir()
        self.nvcc = nvcc

    def build(self, archs, force=False):
        cmds = []

        def run(cmd, cwd=None, env=None, check=True, quiet=False):
            cmds.append([str(c) for c in cmd])
            if "--build" in cmd:
                (self.root / "build").mkdir(exist_ok=True)
                (self.root / "build" / "strata-glm").write_bytes(b"")
            return mock.Mock(returncode=0)
        with mock.patch.object(setup, "run", run), \
                mock.patch.object(setup, "install_build_tools", return_value=str(self.nvcc)), \
                mock.patch.object(setup, "find_tool", lambda name: f"/venv/bin/{name}"), \
                mock.patch.dict(os.environ, {}, clear=False):
            for k in ("CC", "CXX", "CUDAHOSTCXX"):
                os.environ.pop(k, None)
            exe, out = quiet(setup.build_engine, archs, True, Path("/src/third_party/llama.cpp"), force)
        return exe, cmds, out

    def test_the_cmake_command_line(self):
        exe, cmds, out = self.build([86, 86, 86])
        self.assertEqual(exe, self.root / "build" / "strata-glm")
        conf, build = cmds
        self.assertEqual(conf[:7], ["/venv/bin/cmake", "-G", "Ninja", "-DCMAKE_MAKE_PROGRAM=/venv/bin/ninja", "-S",
                                    str(setup.ROOT), "-B"])
        self.assertEqual(conf[7], str(self.root / "build"))
        for d in ("-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_ENABLE_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=86",
                  "-DSTRATA_GGML_DIR=/src/third_party/llama.cpp", f"-DCMAKE_CUDA_COMPILER={self.nvcc}",
                  f"-DCUDAToolkit_ROOT={self.root / 'cuda'}", "-DSTRATA_BUILD_TESTS=OFF"):
            self.assertIn(d, conf)
        self.assertFalse([c for c in conf if "HIP" in c or "VISION" in c])
        self.assertEqual(build[:5], ["/venv/bin/cmake", "--build", str(self.root / "build"), "--target", "strata-glm"])
        self.assertEqual(build[5], "strata-glm-ref")
        self.assertIn("sm_86", out)

    def test_the_stamp_makes_the_next_run_skip_and_a_source_change_compile(self):
        self.build([86])
        meta = json.loads((self.root / "build" / "BUILD.json").read_text())
        self.assertEqual((meta["src"], meta["archs"], meta["source"]), ("hash-1", [86], "local"))
        self.assertEqual(meta["lib_dirs"], [str(self.root / "cuda" / "lib64")])
        self.assertTrue(setup.engine_current([86]))
        _, cmds, out = self.build([86])
        self.assertEqual(cmds, [])
        self.assertIn("already built", out)
        self.assertFalse(setup.engine_current([86, 89]))                     # a new card's generation
        with mock.patch.object(setup, "source_hash", lambda parts=None: "hash-2"):
            self.assertFalse(setup.engine_current([86]))                     # a git pull changed the source
        _, cmds, _ = self.build([89])                                         # keeps the generations it had
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=86;89", cmds[0])
        _, cmds, _ = self.build([86, 89], force=True)                         # --build
        self.assertEqual(len(cmds), 2)

    def test_the_source_hash_covers_the_engine_sources(self):
        self.assertEqual(setup.ENGINE_SOURCES, ("CMakeLists.txt", "src", "include", "third_party"))
        # the real function (setUp replaced it): a source edit changes it, llama.cpp's own files and CRLF do not
        real = Build.real_source_hash
        with tempfile.TemporaryDirectory() as d, mock.patch.object(setup, "ROOT", Path(d)):
            root = Path(d)
            (root / "src" / "glm").mkdir(parents=True)
            (root / "src" / "glm" / "engine.cu").write_text("a\n")
            (root / "third_party" / "llama.cpp").mkdir(parents=True)
            (root / "third_party" / "llama.cpp" / "big.c").write_text("x")
            (root / "CMakeLists.txt").write_text("project(strata VERSION 0.2.0)")
            first = real()
            (root / "third_party" / "llama.cpp" / "big.c").write_text("y")
            self.assertEqual(real(), first)
            (root / "src" / "glm" / "engine.cu").write_bytes(b"a\r\n")
            self.assertEqual(real(), first)
            (root / "src" / "glm" / "engine.cu").write_text("b\n")
            self.assertNotEqual(real(), first)

    def test_a_failed_update_keeps_the_installed_engine(self):
        self.build([86])
        with mock.patch.object(setup, "source_hash", lambda parts=None: "hash-2"), \
                mock.patch.object(setup, "gpus", owner_gpus), \
                mock.patch.object(setup, "get_llama_cpp", return_value=Path("/src/third_party/llama.cpp")), \
                mock.patch.object(setup, "build_engine", side_effect=SystemExit(1)) as build:
            _, out = quiet(setup.update_installed_engine)
        build.assert_called_once()
        self.assertIn("starting the installed one", out)
        with mock.patch.object(setup, "gpus", owner_gpus), \
                mock.patch.object(setup, "build_engine", side_effect=AssertionError("compiled")):
            quiet(setup.update_installed_engine)                              # current: nothing to do


class Config(unittest.TestCase):
    def test_the_config_and_its_args(self):
        code, out, cfgs, scripts, rec = install(["--yes", "--no-start", "--context", "131072"])
        self.assertEqual(code, 0, out)
        cfg = cfgs["strata-glm-ud-iq3_xxs.json"]
        root, data = Path(rec["root"]), Path(rec["base"]) / "Strata-GLM-data"
        self.assertEqual(set(cfg), {"exe", "args", "cwd", "log", "lib_dirs", "model_name", "port"})
        self.assertEqual(cfg["exe"], str(root / "build" / "strata-glm"))
        first = data / "models" / "glm-ud-iq3_xxs" / "GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00003.gguf"
        self.assertEqual(cfg["args"], ["-m", str(first), "--ctx", "131072", "--park-mib", "8192",
                                       "--expert-profile", str(data / "glm-ud-iq3_xxs.profile")])
        self.assertEqual(cfg["cwd"], str(root))
        self.assertEqual(cfg["log"], str(root / "strata-glm-ud-iq3_xxs.log"))
        self.assertEqual(cfg["model_name"], "glm-5.3-flash")
        self.assertEqual(cfg["port"], 8101)
        self.assertNotIn("tokenizer", cfg)                                    # the server reads it from the GGUF

    def test_every_engine_option_is_one_strata_glm_takes(self):
        code, out, cfgs, _, _ = install(["--yes", "--no-start"])
        args = cfgs["strata-glm-ud-iq3_xxs.json"]["args"]
        src = (ROOT / "src" / "glm" / "glm_main.cpp").read_text(encoding="utf-8")
        for flag in (a for a in args if a.startswith("-")):
            self.assertIn(f'"{flag}"', src, flag)

    def test_network_key_gpus_and_browser(self):
        code, out, cfgs, scripts, _ = install(["--yes", "--no-start", "--host", "0.0.0.0", "--api-key", "k1",
                                               "--gpus", "0,2", "--no-browser", "--port", "9000"])
        self.assertEqual(code, 0, out)
        cfg = cfgs["strata-glm-ud-iq3_xxs.json"]
        self.assertEqual((cfg["host"], cfg["api_key"], cfg["gpu"], cfg["open_browser"], cfg["port"]),
                         ("0.0.0.0", "k1", [0, 2], False, 9000))
        self.assertNotIn("--open", scripts["run-glm-ud-iq3_xxs.sh"])
        self.assertIn("http://127.0.0.1:9000/", out)

    def test_every_gpu_is_used_without_naming_them(self):
        code, out, cfgs, _, rec = install(["--yes", "--no-start"])
        self.assertNotIn("gpu", cfgs["strata-glm-ud-iq3_xxs.json"])
        self.assertEqual(rec["builds"], [[86]])

    def test_context_menu_and_its_vram(self):
        code, out, cfgs, _, _ = install(["--no-start", "--model", "UD-IQ3_XXS"])   # questions: stdin is empty
        self.assertEqual(code, 1)                                                  # EOF stops at the question
        self.assertIn("1.5 GB of VRAM", out)
        self.assertIn("11.8 GB of VRAM", out)
        self.assertIn("1M tokens", out)
        self.assertIn("128K tokens", out)
        self.assertAlmostEqual(setup.mla_gb(131072), 1.48, places=2)
        self.assertAlmostEqual(setup.mla_gb(1048576), 11.81, places=2)
        self.assertEqual(setup.recommended_context(46.0), 131072)
        self.assertEqual(setup.recommended_context(12.0), 32768)

    def test_a_long_context_on_little_vram_is_kept_with_a_warning(self):
        smi = "0, NVIDIA GeForce RTX 3060, 12288, 8.6, 580.95.05\n"
        code, out, cfgs, _, _ = install(["--yes", "--no-start", "--context", "1048576"], smi=smi)
        self.assertEqual(code, 0, out)
        self.assertIn("few experts stay cached in VRAM", out)
        args = cfgs["strata-glm-ud-iq3_xxs.json"]["args"]
        self.assertEqual(args[args.index("--ctx") + 1], "1048576")

    def test_choices_from_config_read_it_back(self):
        code, out, cfgs, _, rec = install(["--yes", "--no-start", "--context", "262144", "--park-mib", "6144",
                                           "--port", "8102"])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "strata-glm-ud-iq3_xxs.json"
            p.write_text(json.dumps(cfgs["strata-glm-ud-iq3_xxs.json"]), encoding="utf-8")
            ch = setup.choices_from_config(p)
        self.assertEqual((ch["model"], ch["context"], ch["park_mib"], ch["port"]), ("UD-IQ3_XXS", 262144, 6144, 8102))

    def test_a_qwen_config_is_left_alone(self):
        qwen = {"exe": "engine/strata", "args": ["--native", "packs/iq3_s", "--max-context", "65536"]}
        code, out, cfgs, _, rec = install(["--yes", "--no-start"], configs=[("strata-iq3_s.json", qwen)])
        self.assertEqual(code, 0, out)
        self.assertIn("strata-iq3_s.json: a config of the earlier Qwen version", out)
        self.assertEqual(cfgs["strata-iq3_s.json"], qwen)
        self.assertIn("strata-glm-ud-iq3_xxs.json", cfgs)


class RunScriptAndStart(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        exe = self.root / "build" / "strata-glm"
        exe.parent.mkdir()
        exe.write_bytes(b"")
        first = self.root / "models" / "GLM-5.3-Flash-UD-IQ3_XXS-00001-of-00003.gguf"
        first.parent.mkdir()
        first.write_bytes(b"")
        self.cfg = {"exe": str(exe), "args": ["-m", str(first), "--ctx", "131072", "--park-mib", "8192"],
                    "cwd": str(self.root), "model_name": "glm-5.3-flash", "port": 8101}
        p = mock.patch.object(setup, "ROOT", self.root)
        p.start()
        self.addCleanup(p.stop)

    def test_the_run_script(self):
        cfg_path = self.root / "strata-glm-ud-iq3_xxs.json"
        script = setup.write_run_script("glm-ud-iq3_xxs", cfg_path, 8101)
        self.assertEqual(script.name, "run-glm-ud-iq3_xxs.sh")
        self.assertTrue(os.access(script, os.X_OK))
        text = script.read_text(encoding="utf-8")
        self.assertTrue(text.startswith("#!/bin/sh\n"))
        self.assertIn(f"exec {sys.executable} {self.root / 'serve' / 'server.py'} --engine strata --config "
                      f"{cfg_path} --port 8101 --open", text)
        self.assertNotIn("--open", setup.write_run_script("glm-ud-iq3_xxs", cfg_path, 8101, False).read_text())

    def start(self, port=None, gpu=None, keep=None, cfg=None):
        p = self.root / "strata-glm-ud-iq3_xxs.json"
        p.write_text(json.dumps({**self.cfg, **(cfg or {})}), encoding="utf-8")
        call = mock.Mock(return_value=0)
        with mock.patch.object(setup, "gpus", owner_gpus), mock.patch.object(setup.subprocess, "call", call), \
                mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("STRATA_EXECV", None)
            code, out = quiet(setup.start, p, port, gpu, keep)
        self.assertEqual(code, 0)
        return call.call_args[0][0], out, json.loads(p.read_text(encoding="utf-8"))

    def test_the_start_command_line(self):
        cmd, out, _ = self.start()
        self.assertEqual(cmd, [sys.executable, str(self.root / "serve" / "server.py"), "--engine", "strata",
                               "--config", str(self.root / "strata-glm-ud-iq3_xxs.json"), "--port", "8101", "--open"])
        self.assertIn("--ctx 131072 --park-mib 8192; server 127.0.0.1:8101", out)

    def test_port_gpu_and_browser_on_a_start(self):
        cmd, _, _ = self.start(port=9000, gpu=1)
        self.assertEqual(cmd[-5:], ["--gpu", "1", "--port", "9000", "--open"])
        cmd, _, _ = self.start(cfg={"port": 8102, "open_browser": False})
        self.assertEqual(cmd[-2:], ["--port", "8102"])
        cmd, out, saved = self.start(keep={"open_browser": False, "park_mib": 4096, "host": "0.0.0.0"})
        self.assertNotIn("--open", cmd)
        self.assertEqual((saved["open_browser"], saved["host"]), (False, "0.0.0.0"))
        self.assertEqual(saved["args"][saved["args"].index("--park-mib") + 1], "4096")
        self.assertIn("saved for this model:", out)
        cmd, _, saved = self.start(gpu=[0, 2])                                # --gpus: saved, not on the line
        self.assertNotIn("--gpu", cmd)
        self.assertEqual(saved["gpu"], [0, 2])

    def test_a_config_without_a_port_starts_on_8101(self):
        cfg = dict(self.cfg)
        del cfg["port"]
        self.cfg = cfg
        cmd, _, _ = self.start()
        self.assertEqual(cmd[cmd.index("--port") + 1], "8101")

    def test_missing_files_stop_with_a_repair_hint(self):
        self.cfg["exe"] = str(self.root / "build" / "gone")
        with self.assertRaises(SystemExit):
            self.start()


class Ports(unittest.TestCase):
    def test_8101_everywhere(self):
        self.assertEqual(setup.PORT, 8101)
        self.assertIn('default=8101', (ROOT / "chat.py").read_text(encoding="utf-8"))
        docker = (ROOT / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("EXPOSE 8101", docker)
        self.assertIn("${PORT:-8101}", docker)
        self.assertIn('PORT="${PORT:-8101}"', (ROOT / "docker-entrypoint.sh").read_text(encoding="utf-8"))

    def test_an_installed_model_starts_on_its_port_and_a_new_one_on_8101(self):
        code, out, cfgs, scripts, rec = install(["--yes"])
        self.assertEqual(code, 0, out)
        self.assertEqual(rec["starts"], [("strata-glm-ud-iq3_xxs.json", 8101)])
        self.assertIn("--port 8101", scripts["run-glm-ud-iq3_xxs.sh"])


class Check(unittest.TestCase):
    def test_check_writes_nothing(self):
        code, out, cfgs, _, rec = install(["--check"])
        self.assertEqual(code, 0, out)
        self.assertIn("This PC can run GLM-5.3-Flash (UD-IQ3_XXS recommended)", out)
        self.assertIn("IQ4_XS     4-bit: ~162 GB of experts in RAM - fits, with little RAM to spare", out)
        self.assertEqual((cfgs, rec["downloads"], rec["builds"], rec["urls"]), ({}, [], [], []))
        self.assertFalse((Path(rec["base"]) / "Strata-GLM-data").exists())
        self.assertFalse((Path(rec["base"]) / "settings.json").exists())

    def test_check_on_too_little_ram(self):
        code, out, _, _, _ = install(["--check"], ram=60.0)
        self.assertEqual(code, 1)
        self.assertIn("neither size fits 60 GB of RAM", out)


if __name__ == "__main__":
    unittest.main()
