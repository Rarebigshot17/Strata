#!/usr/bin/env python3
"""Strata one-click setup and start: GLM-5.3-Flash on Linux with NVIDIA GPUs and plenty of RAM.

    ./setup.sh      - installs Python if needed and runs this file

The first time it asks two questions - which size (3-bit or 4-bit) and how much context - then installs everything
and starts the model on http://127.0.0.1:8101 (OpenAI- and Anthropic-compatible API and a chat page).  Every later
start skips straight to running the model: nothing that is already downloaded, built or installed is done again.

What the first run does (each step is skipped when it is already done):

  1. checks your PC: NVIDIA GPUs (compute capability 7.5 or newer) and driver, RAM, CPU, free disk space
  2. asks the questions, and lists the model's files on Hugging Face (unsloth/GLM-5.3-Flash-GGUF)
  3. installs the Python packages it needs into .venv
  4. compiles the engine (strata-glm) for your GPUs; the CUDA toolkit is installed first when it is missing (asks)
  5. downloads the model's GGUF files (resumable; sizes and SHA-256 checked)
  6. writes strata-glm-<size>.json and run-glm-<size>.sh and starts the model

Options: --model UD-IQ3_XXS|IQ4_XS, --context 131072, --port 8101, --yes (recommended answers, no questions),
--setup (set up the other size / change settings instead of starting), --no-start, --no-browser, --host 0.0.0.0
--api-key KEY (reach it from other devices on your network), --gpu N / --gpus 0,2, --park-mib N, --data-dir DIR,
--models-dir DIR, --gguf-dir DIR (use GGUF files you already have), --build (compile the engine again), --check
(only check this PC), --update.

Setup recommends, it never forces: the recommended answers are the defaults (--yes, or Enter), and a bigger choice
than it recommends - the 4-bit model on a PC whose RAM it nearly fills, a long context - is kept, with what it
risks.  With --yes, an explicit flag (--model) is the consent to a risk setup would otherwise stop at; --yes alone
is not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = "unsloth/GLM-5.3-Flash-GGUF"     # the model's GGUF files; their names are listed from it at run time
HF_DEFAULT = "https://huggingface.co"
LLAMA_CPP_COMMIT = "3cf03257f219afbe7334045ff7c6a06ac68c627d"   # ggml for the build (CMakeLists.txt pins the same)
LLAMA_CPP_ZIP = f"https://github.com/ggml-org/llama.cpp/archive/{LLAMA_CPP_COMMIT}.zip"
MIN_DRIVER = 580                       # the driver a CUDA 13 build runs with
MIN_ARCH = 75                          # the oldest compute capability strata-glm is compiled for (RTX 20)
PORT = 8101                            # clear of what often runs beside it (3000, 8000, 8080)
MODEL_NAME = "glm-5.3-flash"
PY_PACKAGES = ["numpy", "jinja2", "regex", "pyyaml", "tqdm", "requests", "cmake", "ninja", "pillow", "psutil"]
REQUIREMENTS = ROOT / "requirements.txt"   # the same packages and their dependencies, pinned (#214)
EXE = "strata-glm"                     # the engine serve/server.py drives
REF_EXE = "strata-glm-ref"             # the CPU reference the GLM checks compare against (tools/glm53_*)
TARGETS = (EXE, REF_EXE)
BUILD_DIR = ROOT / "build"

# The two sizes, by the quantization of their routed experts (docs/GLM53_PORT.md, "Memory on napster-ai").  Every
# expert stays in RAM; the GPUs cache the most used ones.  "fallback": the files taken when the repository has none
# of the size's own quantization.  The sizes are decimal GB (10^9 bytes), as the Hub and the disks count them; the
# RAM is counted in GiB (2^30 bytes), as Linux reports it, so the experts are compared with it in GiB (experts_gib).
MODELS = {
    "UD-IQ3_XXS": {"bits": "3-bit", "about": "the recommended size for a PC with about 192 GB of RAM",
                   "download_gb": 120, "experts_gb": 117},
    "IQ4_XS": {"bits": "4-bit", "about": "better answers, but its experts need about 42 GiB more RAM",
               "download_gb": 170, "experts_gb": 162, "fallback": ("UD-Q4_K_XL",)},
}
GIB = 2**30
RAM_LEFT_GIB = 40        # RAM beside the experts from which a size fits well: Linux, the server, parked conversations
RAM_LEFT_MIN_GIB = 12    # below this the OS pages the experts from the disk: very slow, or no start at all
PARK_MIB = 8192          # the engine's RAM for parked conversations and checkpoints (its default)
PARK_MIB_TIGHT = 4096    # with less than RAM_LEFT_GIB beside the experts (the 4-bit model on 192 GB)
CONTEXTS = [32768, 131072, 262144, 524288, 1048576]
MAX_CONTEXT = 1048576    # GLM-5.3-Flash's trained context
MLA_BYTES_PER_TOKEN = 11 * 512 * 2     # the attention cache in VRAM: a 512-wide f16 latent per token in 11 MLA layers


# ------------------------------------------------------------------------------------------------ output
def say(msg=""):
    print(msg, flush=True)


def step(n, title):
    say()
    say(f"=== Step {n}: {title} ===")


def ok(msg):
    say(f"  [ok] {msg}")


def warn(msg):
    say(f"  [!]  {msg}")


def fail(msg, hint=None):
    say(f"\n  [X]  {msg}")
    if hint:
        say(f"       {hint}")
    say("\nSetup stopped. Fix the item above and run it again - everything already done is kept and skipped.")
    sys.exit(1)


def flush_typed_ahead() -> None:
    """#841: keys pressed while setup was busy (a download, a build) are not answers to the next question: Enter pressed
    to "wake" a slow step answered "Go on anyway?" with its default.  Dropped before a question is asked, on a terminal
    only (a pipe or a test keeps its input)."""
    try:
        if not sys.stdin.isatty():
            return
        import termios
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except (ImportError, OSError, ValueError, AttributeError):
        pass


def ask(question, choices, default, yes):
    if yes:
        return default
    flush_typed_ahead()
    while True:
        try:
            a = input(f"{question} [{default}]: ").strip()
        except EOFError:
            fail("input ended before a setup answer was received",
                 "run setup in a terminal, or pass --yes to accept the recommended answers")
        if not a:
            return default
        if a.lower() in [c.lower() for c in choices]:
            return next(c for c in choices if c.lower() == a.lower())
        say(f"  please answer one of: {', '.join(choices)}")


def confirm_risk(msg, explicit, yes, stop, hint=None, question="  Go on anyway?", default="n") -> None:
    """The owner's rule: setup recommends, it never forces.  A choice setup expects to fail or run badly is said
    plainly (msg), then asked (default n), or with --yes taken as consent when it was asked for explicitly (a flag such
    as --model): --yes alone keeps the stop (stop, hint).  Returns when it goes on; the caller says what it does."""
    warn(msg)
    if yes and explicit:
        return
    if ask(question, ["y", "n"], default, yes) != "y":
        fail(stop, hint)


def run(cmd, cwd=None, env=None, check=True, quiet=False):
    say("  > " + " ".join(str(c) for c in cmd))
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, env=env,
                       stdout=subprocess.PIPE if quiet else None, stderr=subprocess.STDOUT if quiet else None,
                       text=True)
    if check and r.returncode != 0:
        if quiet and r.stdout:
            say(r.stdout[-4000:])
        fail(f"command failed (exit {r.returncode}): {Path(str(cmd[0])).name}")
    return r


def out(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def done(path: Path) -> bool:
    """A step's finish mark: <path>.done exists (written only after the step completed)."""
    return path.with_name(path.name + ".done").exists()


def mark(path: Path, text=""):
    """Write the finish mark.  A folder that cannot be written (--gguf-dir on a read-only share, #570) only costs the
    mark: setup says so and goes on (the step is repeated on the next run), it does not stop."""
    try:
        path.with_name(path.name + ".done").write_text(text or time.strftime("%Y-%m-%d %H:%M"), encoding="utf-8")
    except OSError as e:
        warn(f"the finish mark of {path.name} cannot be written ({e}): the next run does this step again")


# ------------------------------------------------------------------------------------------------ the PC
def ram_gib():
    """The RAM Linux reports (MemTotal, the total of `free -g`), in GiB: 188 on a PC with 192 GB installed."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                return int(line.split()[1]) * 1024 / GIB
    except OSError:
        pass
    return 0.0


def cpu_info():
    """(name, avx2) from /proc/cpuinfo."""
    name, avx2 = "unknown CPU", False
    try:
        txt = open("/proc/cpuinfo").read()
        flags = re.search(r"^flags\s*:\s*(.*)$", txt, re.M)
        avx2 = bool(flags) and "avx2" in flags.group(1).split()
        m = re.search(r"^model name\s*:\s*(.*)$", txt, re.M)
        name = m.group(1) if m else name
    except OSError:
        pass
    return name, avx2


def gpus():
    """Every NVIDIA GPU, numbered as nvidia-smi numbers them (by PCI bus, the order the engine is told to use).  "arch"
    is the compute capability without its dot ("86" for 8.6), "" when nvidia-smi does not say it."""
    s = out(["nvidia-smi", "--query-gpu=index,name,memory.total,compute_cap,driver_version",
             "--format=csv,noheader,nounits"])
    found = []
    for line in s.strip().splitlines():
        try:
            idx, name, mem, cc_, drv = [x.strip() for x in line.split(",")]
            found.append({"index": int(idx), "name": name, "vram_gb": float(mem) / 1024.0,
                          "arch": cc_.replace(".", "") if re.fullmatch(r"\d+\.\d+", cc_) else "", "driver": drv})
        except ValueError:
            continue
    return found


def cc(g) -> str:
    return f"{g['arch'][:-1]}.{g['arch'][-1]}" if g.get("arch") else "?"


def driver_major(g) -> int:
    m = re.match(r"(\d+)", str(g.get("driver", "")))
    return int(m.group(1)) if m else 0


def gpu_problem(g):
    """Why the engine cannot use this card, in plain words (None: it can)."""
    if not g.get("arch"):
        return "nvidia-smi does not say its compute capability (is the driver too old?)"
    if int(g["arch"]) < MIN_ARCH:
        return (f"not supported - compute capability {cc(g)}; strata-glm needs 7.5 or newer (the RTX 20 series and "
                "newer)")
    return None


def gpu_name(g) -> str:
    return f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB)"


def gpu_table(found) -> None:
    say("  Your NVIDIA GPUs:")
    for g in found:
        p = gpu_problem(g)
        say(f"    GPU {g['index']}: {g['name']}, {g['vram_gb']:.0f} GB VRAM, compute capability {cc(g)} - "
            + ("can be used" if p is None else p))


def card_archs(cards) -> list[int]:
    """The compute capabilities of these cards, each once (86 for an RTX 30 card)."""
    return sorted({int(g["arch"]) for g in cards if g.get("arch")})


def cuda_archs(archs) -> str:
    """CMAKE_CUDA_ARCHITECTURES for these compute capabilities: "86", or "86;89" for two generations."""
    return ";".join(str(x) for x in sorted({int(x) for x in archs}))


def check_gpus(sel, found) -> None:
    """Stops with a plain message when a chosen card is missing or cannot run the engine, and says which can."""
    for i in sel:
        g = next((x for x in found if x["index"] == i), None)
        p = "not found on this PC" if g is None else gpu_problem(g)
        if p is None:
            continue
        say()
        gpu_table(found)
        usable = [x["index"] for x in found if gpu_problem(x) is None]
        fail(f"GPU {i}{'' if g is None else ' (' + g['name'] + ')'} cannot be used: {p}",
             ("use " + (f"--gpus {','.join(map(str, usable))}" if len(usable) > 1 else f"--gpu {usable[0]}"))
             if usable else "Strata needs an NVIDIA GPU of compute capability 7.5 or newer (RTX 20 series or newer)")


def parse_gpus(text, found) -> list:
    """--gpus: "0,2" (as nvidia-smi numbers the cards) or "all" (every card that can run the engine)."""
    if str(text).strip().lower() == "all":
        sel = [g["index"] for g in found if gpu_problem(g) is None]
        if not sel:
            gpu_table(found)
            fail("--gpus all: none of this PC's GPUs can run the engine",
                 "Strata needs an NVIDIA GPU of compute capability 7.5 or newer (RTX 20 series or newer)")
        return sel
    try:
        sel = [int(x) for x in str(text).split(",") if x.strip()]
    except ValueError:
        fail(f"--gpus takes GPU numbers as nvidia-smi numbers them, e.g. --gpus 0,2 (or --gpus all), not {text!r}")
    if not sel or len(set(sel)) != len(sel):
        fail("--gpus takes different GPU numbers, e.g. --gpus 0,2")
    check_gpus(sel, found)
    return sel


def choose_gpus(a, found) -> tuple[list, bool]:
    """The cards this install runs on, and whether the config names them: --gpus / --gpu as given, else every card
    that can run the engine (it shares the model's layers across all the cards it sees).  The config names them when
    they were given, or when a card it cannot use has to be left out (CUDA_VISIBLE_DEVICES hides it then)."""
    if a.gpus:
        return parse_gpus(a.gpus, found), True
    if a.gpu is not None:
        check_gpus([a.gpu], found)
        return [a.gpu], True
    usable = [g["index"] for g in found if gpu_problem(g) is None]
    if not usable:
        gpu_table(found)
        fail("none of your GPUs can run the engine",
             "Strata needs an NVIDIA GPU of compute capability 7.5 or newer (RTX 20 series or newer)")
    return usable, len(usable) < len(found)


def config_gpu(sel: list):
    """The run config's "gpu": one card's number, or the list of several."""
    return sel[0] if len(sel) == 1 else list(sel)


def free_gb(path):
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 1e9


# ------------------------------------------------------------------------------------------------ the model's size
def experts_gib(model: str) -> float:
    """The size's experts in GiB, the unit of the RAM: 109 at 3-bit, 151 at 4-bit."""
    return MODELS[model]["experts_gb"] * 1e9 / GIB


def ram_left(model: str, ram: float) -> float:
    """The RAM (GiB) left beside the size's experts on a PC with `ram` GiB."""
    return ram - experts_gib(model)


def model_fit(model: str, ram: float) -> str:
    """"fits" (RAM_LEFT_GIB or more beside the experts), "tight" (less, but more than RAM_LEFT_MIN_GIB), or "no"."""
    left = ram_left(model, ram)
    return "fits" if left >= RAM_LEFT_GIB else "tight" if left >= RAM_LEFT_MIN_GIB else "no"


def recommended_model(ram: float) -> str:
    """The best size that fits well (192 GB installed, 188 GiB reported: the 3-bit; the 4-bit leaves only ~37 GiB
    there); the smallest when none does."""
    return next((m for m in reversed(list(MODELS)) if model_fit(m, ram) == "fits"), next(iter(MODELS)))


def park_mib(model: str, ram: float, asked: int | None = None) -> int:
    """--park-mib: as given, else the engine's default, or PARK_MIB_TIGHT where little RAM is left beside the
    experts (parked conversations share it with Linux and the server)."""
    if asked is not None:
        return asked
    return PARK_MIB if model_fit(model, ram) == "fits" else PARK_MIB_TIGHT


def fit_text(model: str, ram: float) -> str:
    left = ram_left(model, ram)
    return {"fits": f"~{left:.0f} GiB of RAM left beside them",
            "tight": f"only ~{left:.0f} GiB of RAM left beside them",
            "no": f"they do not fit this PC's {ram:.0f} GiB of RAM"}[model_fit(model, ram)]


def mla_gb(ctx: int) -> float:
    """The attention cache's VRAM (GB) for a context of `ctx` tokens: 1.5 GB at 131072, 11.8 GB at 1M."""
    return ctx * MLA_BYTES_PER_TOKEN / 1e9


def recommended_context(vram_gb: float) -> int:
    """128K where the GPUs hold its 1.5 GB attention cache with room to spare for the expert cache, else 32K."""
    return 131072 if vram_gb >= 16 else 32768


def context_label(ctx: int) -> str:
    return "1M" if ctx == MAX_CONTEXT else f"{ctx // 1024}K"


# ------------------------------------------------------------------------------------------------ Hugging Face
def hf_endpoint() -> str:
    """#495: the Hugging Face host - HF_ENDPOINT as huggingface_hub reads it (a mirror, e.g. https://hf-mirror.com),
    else huggingface.co."""
    return (os.environ.get("HF_ENDPOINT") or "").strip().rstrip("/") or HF_DEFAULT


def hf_file_url(path: str, repo: str = REPO) -> str:
    """The download URL of a file of the repository's current revision."""
    return f"{hf_endpoint()}/{repo}/resolve/main/{urllib.parse.quote(path)}"


def hf_tree(repo: str = REPO) -> list[dict]:
    """Every file of the repository's current revision: [{"path", "size", "sha256"}] (sha256 None for a file kept
    without LFS).  The Hub pages a long listing; each page's Link header names the next one."""
    url = f"{hf_endpoint()}/api/models/{repo}/tree/main?recursive=true"
    files = []
    while url:
        req = urllib.request.Request(url, headers={"User-Agent": "strata-setup", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            page = json.loads(r.read())
            link = r.headers.get("Link") or ""
        for f in page:
            if f.get("type") != "file":
                continue
            lfs = f.get("lfs") or {}
            files.append({"path": f["path"], "size": int(lfs.get("size") or f.get("size") or 0),
                          "sha256": (lfs.get("oid") or "").lower() or None})
        nxt = re.search(r'<([^>]+)>\s*;\s*rel="next"', link)
        url = urllib.parse.urljoin(url, nxt.group(1)) if nxt else None
    return files


def repo_files() -> list[dict]:
    """hf_tree(REPO), tried three times; setup stops with a plain message when the Hub does not answer."""
    for attempt in range(3):
        try:
            return hf_tree(REPO)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404) or attempt == 2:
                fail(f"cannot list the model's files ({REPO} on {hf_endpoint()}: {e})",
                     "check the internet connection, or HF_ENDPOINT if you set one, and run it again")
        except (OSError, ValueError) as e:
            if attempt == 2:
                fail(f"cannot list the model's files ({REPO} on {hf_endpoint()}: {e})",
                     "check your internet connection and run it again (HF_ENDPOINT=<mirror> lists and downloads "
                     "from a Hugging Face mirror)")
        time.sleep(5)
    return []


SHARD_NAME = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def quant_in(path: str, quant: str) -> bool:
    """Whether a file's path names this quantization, in a folder or in the file name: as a word of its own, so IQ4_XS
    is not UD-IQ4_XS or IQ4_XS_L."""
    not_ud = "" if quant.upper().startswith("UD-") else r"(?<!UD-)"
    return re.search(rf"(?<![A-Za-z0-9_]){not_ud}{re.escape(quant)}(?![A-Za-z0-9_])", path, re.I) is not None


def shard_sets(files: list[dict], quant: str) -> list[list[dict]]:
    """The complete GGUF files of one quantization in a listing, each the list of its shards in order: every
    <stem>-0000i-of-0000N.gguf of a stem (in any folder), or a file that is not split."""
    groups = {}
    for f in files:
        path = f["path"]
        name = path.rsplit("/", 1)[-1]
        if not name.lower().endswith(".gguf") or name.lower().startswith("mmproj") or not quant_in(path, quant):
            continue
        m = SHARD_NAME.search(path)
        key = (path[:m.start()], int(m.group(2))) if m else (path, 1)
        groups.setdefault(key, {})[int(m.group(1)) if m else 1] = f
    return [[got[i] for i in range(1, n + 1)] for (_stem, n), got in sorted(groups.items())
            if sorted(got) == list(range(1, n + 1))]


def pick_shards(files: list[dict], model: str):
    """(the quantization, its shards) setup downloads for a size: the size's own files, else its fallback's (the 4-bit:
    UD-Q4_K_XL when the repository has no IQ4_XS).  None when the listing has neither."""
    for quant in (model, *MODELS[model].get("fallback", ())):
        sets = shard_sets(files, quant)
        if sets:
            if len(sets) > 1:
                warn(f"the repository has {len(sets)} complete {quant} files: taking "
                     + sets[0][0]["path"] + " (the others: " + ", ".join(s[0]["path"] for s in sets[1:]) + ")")
            return quant, sets[0]
    return None


def model_tag(name: str) -> str:
    """The name of a size's config, start script and expert profile, and of a quantization's model folder."""
    return "glm-" + name.lower()


# ------------------------------------------------------------------------------------------------ downloads
def drop_archive(z: Path) -> None:
    """A source archive that is unpacked or damaged goes with its .done mark: a damaged one kept them, and every later
    run reused it ("already downloaded")."""
    z.unlink(missing_ok=True)
    z.with_name(z.name + ".done").unlink(missing_ok=True)


def download(url, dst: Path, what=None):
    """Resumable HTTP(S) download with a progress line; `file://` and plain paths are copied (tests, mirrors).
    A finished file gets a <name>.done mark, so a later run skips it without asking the server."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and done(dst):
        ok(f"{what or dst.name} already downloaded")
        return
    if not url.startswith(("http://", "https://")):
        src = Path(url[7:] if url.startswith("file://") else url)
        if not src.exists():
            fail(f"not found: {src}")
        shutil.copyfile(src, dst)
        mark(dst)
        ok(f"{what or dst.name} copied")
        return
    hint = "check your internet connection and run it again"
    part = dst.with_name(dst.name + ".part")
    total = 0
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "strata-setup"})
            total = int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length", 0))
            break
        except urllib.error.HTTPError as e:            # the server answers, but not with the file
            if e.code in (401, 403, 404) or attempt == 4:
                model_file = url.startswith(hf_endpoint() + "/")
                fail(f"{url.split('/')[2]} does not give {what or dst.name} ({e})",
                     hint if e.code not in (401, 403, 404) else
                     "run setup again: it lists the model's files again (HF_ENDPOINT, if you set one, must be a "
                     "Hugging Face mirror)" if model_file else
                     "check that this PC may reach it (a proxy or firewall can refuse it) and run it again")
            time.sleep(5)
        except OSError as e:
            if attempt == 4:
                fail(f"cannot reach {url.split('/')[2]} ({e})", hint)
            time.sleep(5)
    if dst.exists() and total and dst.stat().st_size == total:    # finished by an earlier run (no mark yet)
        mark(dst)
        ok(f"{what or dst.name} already downloaded")
        return
    have = part.stat().st_size if part.exists() else 0
    for attempt in range(30):
        if total and have >= total:                    # stopped after the last byte, before the rename: nothing to
            break                                      # ask for (a range past the end is a 416, retried 30 times)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "strata-setup", "Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:                     # the server ignored the range: start over
                    f.seek(0)
                    f.truncate()
                    have = 0
                last = 0.0
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    f.write(b)
                    have += len(b)
                    if time.time() - last > 2:
                        last = time.time()
                        size = f"{have / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * have / total:.0f}%)" if total \
                            else f"{have / 1e6:7.1f} MB"
                        print(f"\r  {what or dst.name}: {size}   ", end="", flush=True)
            print()
            if not total or have >= total:
                break
        except OSError as e:
            print()
            warn(f"download interrupted ({e}); retrying in 10 s ...")
            time.sleep(10)
    got = part.stat().st_size if part.exists() else 0
    if not part.exists() or (total and got != total):  # no .part: every try failed before the first byte
        fail(f"could not finish downloading {dst.name}: {got:,} bytes on disk"
             + (f", the server says {total:,}" if total else ""),
             "check your internet connection and run it again (the download resumes where it stopped)")
    part.replace(dst)
    mark(dst)
    ok(f"{what or dst.name} downloaded")


def verify_file(s: Path, size: int, sha: str | None) -> None:
    """A downloaded shard against the repository's listing: its size, and its SHA-256 when the listing has one.  The
    hash is kept in the finish mark, so the minutes of hashing 120-170 GB happen once.  A wrong file is deleted, so
    the next run downloads it again."""
    m = s.with_name(s.name + ".done")
    if sha and m.exists() and f"sha256 {sha}" in m.read_text(encoding="utf-8", errors="replace"):
        return
    have = s.stat().st_size if s.exists() else -1
    if size and have != size:
        s.unlink(missing_ok=True)
        m.unlink(missing_ok=True)
        fail(f"{s.name} is {have:,} bytes, not the {size:,} the repository lists: deleted",
             "run setup again to download it again")
    if not sha:
        return
    say(f"  checking {s.name} (SHA-256, {have / 1e9:.1f} GB) ...")
    h = hashlib.sha256()
    with open(s, "rb") as f:
        while True:
            b = f.read(16 << 20)
            if not b:
                break
            h.update(b)
    if h.hexdigest() != sha:
        s.unlink(missing_ok=True)
        m.unlink(missing_ok=True)
        fail(f"{s.name} has the wrong SHA-256 ({h.hexdigest()}, the repository lists {sha}): deleted",
             "run setup again to download it again")
    mark(s, f"sha256 {sha}")


def shard_paths(first: Path) -> list[Path]:
    """Every shard a first shard's name counts (<stem>-00001-of-0000N.gguf), in order; a file that is not split alone."""
    m = SHARD_NAME.search(first.name)
    if not m:
        return [first]
    stem, n = first.name[:m.start()], int(m.group(2))
    return [first.with_name("%s-%05d-of-%05d.gguf" % (stem, i, n)) for i in range(1, n + 1)]


def finished_model(folder: Path) -> list[Path] | None:
    """The shards of a model folder whose download finished (each with its finish mark), or None."""
    if not folder.is_dir():
        return None
    for first in sorted(folder.glob("*-00001-of-*.gguf")):
        shards = shard_paths(first)
        if all(s.exists() and done(s) for s in shards):
            return shards
    return None


def find_model(model: str, bases: list[Path]):
    """A size already downloaded into one of these models folders (this data folder's, or one on another drive):
    (its quantization, its shards), else None."""
    for quant in (model, *MODELS[model].get("fallback", ())):
        for base in dict.fromkeys(bases):
            got = finished_model(base / model_tag(quant))
            if got:
                return quant, got
    return None


def gguf_dir_model(folder: Path, model: str):
    """--gguf-dir: (the quantization, its shards) of GGUF files already on this PC - the first shard under the folder
    (in sub-folders too, as `hf download --local-dir` leaves them) whose path names the size's quantization or its
    fallback.  check_shards says later when a shard is missing or short."""
    for quant in (model, *MODELS[model].get("fallback", ())):
        firsts = sorted(p for p in folder.rglob("*.gguf")
                        if not p.name.lower().startswith("mmproj") and quant_in(p.relative_to(folder).as_posix(), quant)
                        and (SHARD_NAME.search(p.name) is None or SHARD_NAME.search(p.name).group(1) == "00001"))
        if firsts:
            return quant, shard_paths(firsts[0])
    names = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*.gguf"))[:8]
    fail(f"{folder} has no {model} GGUF file" + (f" (it has: {', '.join(names)})" if names else ""),
         "--gguf-dir takes the folder with the model's shards (<name>-00001-of-0000N.gguf ...), as "
         f"`hf download {REPO} --include \"*{model}*\"` leaves them")


def check_shards(shards):
    """Every shard present and whole, or setup stops naming the file and the numbers.  Whole means as long as
    its own tensor directory says (the header is read, the data is not): a truncated copy (--gguf-dir, a .part
    renamed by hand) otherwise passes as a model file and the engine fails much later, at the first tensor that runs
    past the end."""
    sys.path.insert(0, str(ROOT / "tools"))
    from gguf_reader import GGUFFile
    for s in shards:
        if not s.exists():
            fail(f"missing {s}")
        try:
            g = GGUFFile(s)
        except (ValueError, KeyError, struct.error) as e:
            fail(f"{s.name} is not a whole GGUF shard ({e})", "delete it and run setup again")
        need = g.data_start + max((t.offset + (t.expected_bytes() or 0) for t in g.tensors), default=0)
        have = s.stat().st_size
        if have < need:
            fail(f"{s.name} is short: {have:,} of {need:,} bytes ({need - have:,} missing)",
                 "delete it and run setup again (or copy the whole file into --gguf-dir)")


def get_llama_cpp():
    """llama.cpp at the pinned commit (ggml for the build: STRATA_GGML_DIR; gguf-py for the tools), as a zip: no git."""
    llama = ROOT / "third_party" / "llama.cpp"
    if (llama / "ggml" / "CMakeLists.txt").exists() and (llama / "gguf-py").is_dir():
        return llama
    z = ROOT / "third_party" / f"llama.cpp-{LLAMA_CPP_COMMIT[:7]}.zip"
    download(LLAMA_CPP_ZIP, z, "llama.cpp source")
    tmp = ROOT / "third_party" / "_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        with zipfile.ZipFile(z) as f:
            # llama.cpp's own web UI (tools/ui) is not used
            f.extractall(tmp, [m for m in f.namelist() if "/tools/ui/" not in m])
    except zipfile.BadZipFile:
        # #1797: do not let the .done mark make every later run reuse a damaged source archive.
        try:
            drop_archive(z)
        except OSError as e:
            warn(f"cannot remove damaged {z.name} ({e}): delete it and its .done mark before running setup again")
        raise
    top = next(tmp.iterdir())
    shutil.rmtree(llama, ignore_errors=True)
    shutil.move(str(top), str(llama))                  # to `llama` itself: its parent would keep the zip's folder name
    shutil.rmtree(tmp, ignore_errors=True)
    drop_archive(z)
    return llama


# ------------------------------------------------------------------------------------------------ Python packages
def req_name(line: str) -> str:
    """The distribution name of a requirement line ("numpy==2.5.3; python_version >= '3.12'" -> "numpy")."""
    return re.split(r"[\s<>=!~;\[]", line.strip(), maxsplit=1)[0].lower().replace("_", "-")


def requirement_lines(path: Path | None = None) -> list[str]:
    """requirements.txt's requirements, comments and blank lines left out."""
    lines = []
    for raw in (path or REQUIREMENTS).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


def _installed(name: str) -> bool:
    try:
        import importlib.metadata as md
        md.distribution(name)
        return True
    except Exception:
        return False


def pip_install(packages, what):
    """pip install into .venv, skipped when the same list was installed before.  An install from before the pinned
    requirements (#214) recorded bare names: those packages are kept as they are (nothing is reinstalled), and the
    pinned dependencies it already has count as installed."""
    stamp = Path(sys.prefix) / ".strata-pip.json"
    have = json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else []
    bare = {p.lower() for p in have if req_name(p) == p.lower()}
    need = [p for p in packages if p not in have and req_name(p) not in bare
            and not (bare and "==" in p and _installed(req_name(p)))]
    if not need:
        ok(f"{what} already installed")
        return
    say(f"  Installing {what} ...")
    run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *need])
    stamp.write_text(json.dumps(sorted(set(have) | set(need)), indent=0))
    ok(f"{what} installed")


def python_packages():
    pip_install(requirement_lines() if REQUIREMENTS.exists() else PY_PACKAGES, ", ".join(PY_PACKAGES))


# ------------------------------------------------------------------------------------------------ the engine
def find_nvcc(below=None):
    """The newest CUDA toolkit's nvcc and its (major, minor); with `below`, the newest older than that version.
    #601: STRATA_NVCC=<path to nvcc> is the only one considered (a newer toolkit beside it that cannot build on this
    PC - CUDA 12.9 with glibc 2.43 - is not taken instead)."""
    pick = os.environ.get("STRATA_NVCC")
    if pick:
        if not Path(pick).exists():
            warn(f"STRATA_NVCC={pick}: no such file; looking for a CUDA toolkit as usual")
        else:
            v = re.search(r"release (\d+)\.(\d+)", out([pick, "--version"]))
            ver = (int(v.group(1)), int(v.group(2))) if v else None
            if ver and below is not None and ver >= below:
                warn(f"STRATA_NVCC={pick} is CUDA {ver[0]}.{ver[1]}; this build needs one older than "
                     f"{below[0]}.{below[1]}")
                return (None, None)
            return (pick, ver) if ver else (None, None)
    cands = [shutil.which("nvcc")]
    for var in ("CUDA_PATH", "CUDA_HOME"):
        if os.environ.get(var):
            cands.append(str(Path(os.environ[var]) / "bin" / "nvcc"))
    cands += [str(p / "bin" / "nvcc") for p in sorted(Path("/usr/local").glob("cuda*"), reverse=True)]
    cands += [str(p / "bin" / "nvcc") for p in sorted(Path("/opt").glob("cuda*"), reverse=True)]   # Arch (#46)
    best = (None, None)
    for c in dict.fromkeys(cands):                     # every toolkit found; the newest wins
        if c and Path(c).exists():
            v = re.search(r"release (\d+)\.(\d+)", out([c, "--version"]))
            ver = (int(v.group(1)), int(v.group(2))) if v else None
            if ver and (below is None or ver < below) and (best[1] is None or ver > best[1]):
                best = (c, ver)
    return best


def find_tool(name):
    """A tool on PATH, or the one pip installed next to this Python (cmake, ninja).  A PATH copy that does not run
    (a pip launcher whose Python is gone) is skipped for the one next to this Python."""
    p = shutil.which(name)
    if p:
        try:
            if subprocess.run([p, "--version"], capture_output=True, timeout=30).returncode == 0:
                return p
        except (OSError, subprocess.SubprocessError):
            pass
    for d in (Path(sys.executable).parent, Path.home() / ".local" / "bin"):
        c = d / name
        if c.exists():
            return str(c)
    return None


CUDA_SM120_SUSPECT = (13, 2)   # #892 #968: nvcc 13.2.0 / 13.2.1 for sm_120 made garbage answers that the same source
                               # compiled with 13.0 answers correctly (llama.cpp hit it too)
CUDA_SM120_FIXED_BUILD = 86    # CUDA 13.2.2 (nvcc build 13.2.86) fixes it (ggml-org/llama.cpp#28581)


def nvcc_build(nvcc):
    """The build number of an nvcc ("V13.2.86" -> 86); None when it cannot be read."""
    m = re.search(r"\bV\d+\.\d+\.(\d+)", out([nvcc, "--version"]) if nvcc else "")
    return int(m.group(1)) if m else None


def sm120_nvcc(nvcc, cuda_v, archs):
    """#892 / #968: CUDA 13.2.0 / 13.2.1's nvcc for an RTX 50 card (sm_120).  Returns (nvcc, cuda_v): an older 13.x
    toolkit when one is installed and the user did not name a compiler (STRATA_NVCC), else the one found, with a
    warning that says what to do.  CUDA 13.2.2 (build 86 or newer) is fine."""
    if not nvcc or cuda_v != CUDA_SM120_SUSPECT or max(archs) < 120:
        return nvcc, cuda_v
    build = nvcc_build(nvcc)
    if build is not None and build >= CUDA_SM120_FIXED_BUILD:
        return nvcc, cuda_v
    if not os.environ.get("STRATA_NVCC"):
        alt, alt_v = find_nvcc(below=CUDA_SM120_SUSPECT)
        if alt and alt_v and alt_v >= (13, 0):
            ok(f"CUDA {alt_v[0]}.{alt_v[1]} is used for the RTX 50 card (sm_120): CUDA 13.2.0 / 13.2.1's compiler made "
               "wrong answers there (#892, #968); STRATA_NVCC=<nvcc> picks another")
            return alt, alt_v
    warn("CUDA 13.2.0 and 13.2.1's compiler (nvcc) made garbage answers for RTX 50 cards (sm_120) in two reports "
         "(#892, #968): the same source compiled with CUDA 13.0, 13.1 or 13.2.2 is right. If the engine answers with "
         "nonsense, update to CUDA 13.2.2 or install CUDA 13.0 next to it and set STRATA_NVCC to its nvcc")
    return nvcc, cuda_v


def os_release() -> dict:
    """/etc/os-release as a dict (ID, VERSION_ID, ...: quotes removed); {} when it cannot be read."""
    try:
        lines = Path("/etc/os-release").read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    return {k.strip(): v.strip().strip('"') for k, v in (line.split("=", 1) for line in lines if "=" in line)}


def install_build_tools(archs, yes):
    """The C++ compiler and the CUDA toolkit, installed for the user when they are missing (asks once): build-essential
    and CUDA 13.0 through apt on Ubuntu 22.04 / 24.04.  An installed toolkit of 12.0 or newer is used as it is (13.0
    for an RTX 50 card, sm_120).  Returns nvcc."""
    need = (13, 0) if max(archs) >= 120 else (12, 0)
    nvcc, cuda_v = find_nvcc()
    have_cc = shutil.which("g++") is not None
    missing = []
    if not have_cc:
        missing.append("the C++ compiler (build-essential)")
    if nvcc is None or cuda_v < need:
        missing.append("the NVIDIA CUDA Toolkit 13.0")
    if not missing:
        nvcc, cuda_v = sm120_nvcc(nvcc, cuda_v, archs)
        ok(f"build tools present (CUDA {cuda_v[0]}.{cuda_v[1]})")
        return nvcc
    say("  The engine is compiled for your GPUs, which needs: " + " and ".join(missing) + ".")
    say("  They can be installed now (about 8-10 GB, 15-40 minutes, sudo will ask for your password).")
    if ask("  Install them now?", ["y", "n"], "y", yes) != "y":
        fail("the build tools are needed", "install them yourself and run it again")
    if shutil.which("apt-get") is None:
        fail("missing: " + " and ".join(missing) + " (the automatic install is only done on Ubuntu/Debian)",
             "install them with your distribution's packages (Arch: pacman -S base-devel cuda; nvcc is found on "
             "PATH, in /usr/local/cuda* and in /opt/cuda*), then run it again")
    if not have_cc:
        run(["sudo", "apt-get", "install", "-y", "build-essential"])
    if nvcc is None or cuda_v < need:
        osr = os_release()
        ver = osr.get("VERSION_ID", "").replace(".", "")
        if osr.get("ID") != "ubuntu" or ver not in ("2204", "2404"):
            fail("the CUDA Toolkit can be installed automatically on Ubuntu 22.04 / 24.04 only",
                 "install it from https://developer.nvidia.com/cuda-downloads and run it again")
        deb = Path("/tmp/cuda-keyring.deb")
        download(f"https://developer.download.nvidia.com/compute/cuda/repos/ubuntu{ver}/x86_64/cuda-keyring_1.1-1_all.deb",
                 deb, "CUDA repository key")
        run(["sudo", "dpkg", "-i", str(deb)])
        run(["sudo", "apt-get", "update"])
        run(["sudo", "apt-get", "install", "-y", "cuda-toolkit-13-0"])
    nvcc, cuda_v = find_nvcc()
    if shutil.which("g++") is None:
        fail("the C++ compiler did not install", "install build-essential by hand and run it again")
    if nvcc is None or cuda_v < need:
        fail("the CUDA Toolkit did not install", "install it from https://developer.nvidia.com/cuda-downloads, then run it again")
    nvcc, cuda_v = sm120_nvcc(nvcc, cuda_v, archs)
    ok(f"build tools installed (CUDA {cuda_v[0]}.{cuda_v[1]})")
    return nvcc


def cache_generator(bdir: Path) -> str | None:
    """The CMake generator a build folder was configured with (CMakeCache.txt), None when it has none yet."""
    try:
        m = re.search(r"^CMAKE_GENERATOR:INTERNAL=(.*)$", (bdir / "CMakeCache.txt").read_text(errors="replace"), re.M)
    except OSError:
        return None
    return m.group(1).strip() if m else None


def cmake_build(src, bdir, targets, defs):
    """Configure and build `targets` with Ninja (a folder configured by hand with another generator - `cmake -B build`
    gives Makefiles - keeps it: CMake refuses to switch).  A failed build is tried once more: CUDA's ptxas now and then
    fails to parse a PTX file it just wrote, and the same command then gets past it (issue #45)."""
    cmake, ninja = find_tool("cmake"), find_tool("ninja")
    if cmake is None or ninja is None:
        fail("cmake / ninja not found after installing them", "run: .venv/bin/python -m pip install cmake ninja")
    gen = cache_generator(Path(bdir))
    use = ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"] if gen in (None, "Ninja") else []
    conf = [cmake, *use, "-S", str(src), "-B", str(bdir), "-DCMAKE_BUILD_TYPE=Release", *defs]
    build = [cmake, "--build", str(bdir), "--target", *targets, "-j", str(max(2, (os.cpu_count() or 4) // 2))]
    run(conf)
    if run(build, check=False).returncode != 0:
        say("  (the build stopped - trying it once more)")
        run(build)


# The files the engine is compiled from: when a `git pull` changes them, it is compiled again (issue #31).
# third_party/llama.cpp is llama.cpp at LLAMA_CPP_COMMIT, which the hash names instead of reading it.
ENGINE_SOURCES = ("CMakeLists.txt", "src", "include", "third_party")
NOT_SOURCES = ("third_party/llama.cpp", "third_party/_unpack")


def source_hash(parts=ENGINE_SOURCES) -> str:
    """A fingerprint of the files a compiled engine is built from, kept in build/BUILD.json."""
    h = hashlib.sha256(LLAMA_CPP_COMMIT.encode())
    for part in parts:
        base = ROOT / part
        if base.is_file():
            files = [base]
        else:
            files = []
            for d, dirs, names in os.walk(base):
                rel = Path(d).relative_to(ROOT).as_posix()
                dirs[:] = sorted(x for x in dirs if f"{rel}/{x}" not in NOT_SOURCES)
                files += [Path(d) / n for n in names if not n.endswith((".zip", ".done", ".part"))]
        for f in sorted(files):
            h.update(f.relative_to(ROOT).as_posix().encode() + b"\0" + f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:16]


def host_compiler_defs() -> list:
    """CC / CXX / CUDAHOSTCXX pick the host compilers of a build - as -D definitions, so an existing build folder
    takes them too (the environment reaches CMake only on a folder's first configure, #1645).  CUDAHOSTCXX is nvcc's
    host compiler (-ccbin); it defaults to CXX, as CMake's does."""
    c = os.environ.get("CC", "").strip()
    cxx = os.environ.get("CXX", "").strip()
    hostcxx = os.environ.get("CUDAHOSTCXX", "").strip() or cxx
    return ([f"-DCMAKE_C_COMPILER={c}"] if c else []) + ([f"-DCMAKE_CXX_COMPILER={cxx}"] if cxx else []) + \
           ([f"-DCMAKE_CUDA_HOST_COMPILER={hostcxx}"] if hostcxx else [])


def toolkit_root(nvcc) -> Path:
    return Path(nvcc).resolve().parent.parent


def toolkit_root_defs(nvcc) -> list:
    """CUDAToolkit_ROOT for the toolkit whose nvcc builds the engine.  Without it CMake can take cudart and cuBLAS from
    another toolkit (the distribution's CUDA 12.0 in /usr beside a CUDA 13.0 in /usr/local).  The distribution's own
    nvcc (/usr/bin) keeps CMake's search as before."""
    root = toolkit_root(nvcc)
    return [] if root == Path("/usr") else [f"-DCUDAToolkit_ROOT={root}"]


def build_defs(archs, nvcc, llama) -> list[str]:
    """The CMake definitions of the engine's build: CUDA for these compute capabilities, ggml from llama.cpp at the
    pinned commit, no unit tests."""
    return ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs(archs)}",
            f"-DCMAKE_CUDA_COMPILER={nvcc}", *toolkit_root_defs(nvcc), f"-DSTRATA_GGML_DIR={llama}",
            *host_compiler_defs()]


def source_version() -> str:
    """The engine version the source tree builds (CMakeLists.txt's project version)."""
    m = re.search(r"project\(strata VERSION ([\d.]+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"))
    return m.group(1) if m else "0"


def installed_build() -> dict:
    """build/BUILD.json: what the engine in build/ was compiled from and for ({} when there is none)."""
    try:
        meta = json.loads((BUILD_DIR / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return meta if isinstance(meta, dict) else {}


def engine_current(archs, meta: dict | None = None) -> bool:
    """The engine in build/ is of this source and has code for every one of these compute capabilities."""
    meta = installed_build() if meta is None else meta
    return (BUILD_DIR / EXE).exists() and meta.get("src") == source_hash() and \
        {int(x) for x in archs} <= {int(x) for x in meta.get("archs", [])}


def write_build_stamp(archs, nvcc) -> dict:
    """build/BUILD.json after a build: the source's hash and version, the compute capabilities, and the toolkit's
    library folders (the run config's lib_dirs: the engine loads cuBLAS and the CUDA runtime from there)."""
    root = toolkit_root(nvcc)
    libs = [str(d) for d in (root / "lib64", root / "targets" / "x86_64-linux" / "lib") if d.is_dir()][:1]
    meta = {"source": "local", "version": source_version(), "archs": sorted({int(x) for x in archs}),
            "nvcc": str(nvcc), "lib_dirs": libs, "src": source_hash()}
    (BUILD_DIR / "BUILD.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


def build_engine(archs, yes, llama, force=False) -> Path:
    """Compile strata-glm, and strata-glm-ref (the CPU reference the GLM checks compare against), into build/ for these
    compute capabilities - unless the build there is of this source and has code for all of them (`force`: compile
    anyway).  A build whose source changed since (a `git pull`) compiles only what changed, a few minutes; it keeps
    the compute capabilities it had."""
    meta = installed_build()
    if not force and engine_current(archs, meta):
        ok("engine already built for this PC")
        return BUILD_DIR / EXE
    archs = sorted({int(x) for x in archs} | {int(x) for x in meta.get("archs", [])})
    nvcc = install_build_tools(archs, yes)
    say("  Compiling the engine for " + ", ".join(f"sm_{x}" for x in archs) + (
        " (a few minutes: only what changed) ..." if (BUILD_DIR / EXE).exists() else " (10-20 minutes, once) ..."))
    cmake_build(ROOT, BUILD_DIR, TARGETS, build_defs(archs, nvcc, llama))
    if not (BUILD_DIR / EXE).exists():
        fail(f"the build finished without {BUILD_DIR / EXE}", f"delete {BUILD_DIR} and run setup again")
    write_build_stamp(archs, nvcc)
    ok(f"engine compiled: {BUILD_DIR / EXE}")
    return BUILD_DIR / EXE


def update_installed_engine(force=False) -> None:
    """Before a start or with --update: the engine is compiled again when its source changed (a `git pull`), when it
    is missing, or when the PC has a GPU it has no code for.  A failed compile keeps the installed engine: the model
    starts as before."""
    meta = installed_build()
    usable = [g for g in gpus() if gpu_problem(g) is None]
    archs = card_archs(usable) or [int(x) for x in meta.get("archs", [])]
    if not force and engine_current(archs, meta):
        return
    if not archs:
        warn("no NVIDIA GPU found (nvidia-smi did not answer): the engine is not compiled again")
        return
    if (BUILD_DIR / EXE).exists() and not force:
        say("  The engine's source changed, or a GPU is new: compiling it again ...")
    try:
        build_engine(archs, False, get_llama_cpp(), force=True)
    except (Exception, SystemExit) as e:               # a failed compile must not stop the model from starting
        warn(f"could not compile the updated engine{'' if isinstance(e, SystemExit) else f' ({e})'}: starting the "
             "installed one")


# ------------------------------------------------------------------------------------------------ the data folder
# The model files (120-170 GB) live in a data folder NEXT TO the Strata folder (`Strata-GLM-data`), not inside it:
# updating Strata by unzipping a new copy then finds them instead of downloading them again.  Where it is, and which
# Strata folders this user ran, is kept in a small per-user file, so every Strata folder on the PC finds the same
# files.  Its name and the data folder's are this version's own: an older Strata of the Qwen model keeps its own.
DATA_ITEMS = ("models",)


def settings_path() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "strata-glm" / "settings.json"


def load_settings() -> dict:
    try:
        s = json.loads(settings_path().read_text(encoding="utf-8"))
        return s if isinstance(s, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(s: dict) -> None:
    try:
        settings_path().parent.mkdir(parents=True, exist_ok=True)
        settings_path().write_text(json.dumps(s, indent=1), encoding="utf-8")
    except OSError as e:
        warn(f"could not save {settings_path()} ({e})")


def has_data(folder: Path) -> bool:
    for d in DATA_ITEMS:
        try:
            if (folder / d).is_dir() and any((folder / d).iterdir()):
                return True
        except OSError:
            pass
    return False


def glm_checkout(d: Path) -> bool:
    """A Strata folder of this GLM-only version (one of the Qwen version has src/core as well)."""
    return (d / "setup.py").is_file() and (d / "src" / "glm").is_dir() and not (d / "src" / "core").is_dir()


def other_installs(settings: dict) -> list:
    """GLM Strata folders besides this one that may hold model files or configs: the ones this user ran before, and
    Strata* folders next to this one (a zip unpacked again lands in e.g. `Strata-glm53-flash (1)/Strata-glm53-flash`)."""
    cands = [Path(p) for p in settings.get("installs", [])]
    for base in dict.fromkeys((ROOT.parent, ROOT.parent.parent)):
        try:
            for d in base.iterdir():
                if d.is_dir() and d.name.lower().startswith("strata"):
                    cands.append(d)
                    cands += [c for c in d.iterdir() if c.is_dir() and c.name.lower().startswith("strata")]
        except OSError:
            pass
    found = []
    for d in cands:
        try:
            d = d.resolve()
            if d != ROOT and d not in found and glm_checkout(d):
                found.append(d)
        except OSError:
            pass
    return found


def same_drive(a: Path, b: Path) -> bool:
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return False


def move_into(src: Path, dst: Path) -> None:
    """A rename into the data folder (same drive: instant); a folder merges into one already there, keeping what the
    destination has.  Whatever cannot be moved (a file in use) stays where it is."""
    if not dst.exists():
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, dst)
            return
        except OSError:
            if not src.is_dir():
                return
            dst.mkdir(parents=True, exist_ok=True)
    if src.is_dir() and dst.is_dir():
        for c in list(src.iterdir()):
            move_into(c, dst / c.name)
        try:
            src.rmdir()
        except OSError:
            pass


def repoint_config(cfg_file: Path, old: Path, new: Path) -> None:
    """A config whose model files moved from `old` to `new` points at them there (each path only if its file is
    now there and no longer at the old place)."""
    try:
        cfg = json.loads(cfg_file.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return

    def fix(v):
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, str):
            for d in DATA_ITEMS:
                o = str(old / d)
                if v == o or v.startswith(o + os.sep):
                    n = str(new / d) + v[len(o):]
                    if Path(n).exists() and not Path(v).exists():
                        return n
        return v

    new_cfg = fix(cfg)
    if new_cfg != cfg:
        write_config(cfg_file, new_cfg)


def data_folder(requested: str | None) -> tuple:
    """(the data folder, folders on other drives that still hold model files).  Moves the model files of this folder,
    of earlier GLM Strata folders and of the data folder used before on the same drive into the data folder, and
    points their configs there."""
    settings = load_settings()
    dest = Path(requested).expanduser().resolve() if requested else \
        Path(settings["data_dir"]) if settings.get("data_dir") else ROOT.parent / "Strata-GLM-data"
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as e:                                # e.g. no write access next to the Strata folder
        warn(f"cannot use {dest} for the model files ({e}): keeping them in {ROOT}")
        dest = ROOT
    elsewhere = []
    sources = [ROOT, *other_installs(settings)]
    if settings.get("data_dir") and Path(settings["data_dir"]) != dest:
        sources.append(Path(settings["data_dir"]))      # #198: a --data-dir to a new place moves them there
    seen = set()
    for folder in sources:
        if str(folder) in seen:
            continue
        seen.add(str(folder))
        if folder == dest or not has_data(folder):
            continue
        if not same_drive(folder, dest):
            elsewhere.append(folder)                    # another drive: used where it is (no 120 GB copy)
            continue
        move_into(folder / "models", dest / "models")   # file by file: the same file wherever it came from
        for c in model_config_files(folder):
            repoint_config(c, folder, dest)
        if has_data(folder):
            elsewhere.append(folder)                    # in use, or a copy the data folder already has
            warn(f"some model files are still in {folder} (in use, or already in {dest})")
        else:
            ok(f"model files from {folder} moved to {dest} (a new copy of Strata finds them there)")
    installs = [str(ROOT)] + [p for p in settings.get("installs", []) if p != str(ROOT) and Path(p).is_dir()]
    save_settings({**settings, "data_dir": str(dest), "installs": installs[:20]})
    return dest, elsewhere


# ------------------------------------------------------------------------------------------------ the run config
def write_config(path: Path, cfg: dict):
    """A run config, written whole or not at all (#459): to a temporary file first, then moved over the old one, so
    a setup stopped half-way (a closed window, a full disk) never leaves an empty strata-*.json behind.  A config that
    is a link (the Docker image links it to the /data volume, #1244) is written where it points."""
    path = Path(os.path.realpath(path))
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    os.replace(tmp, path)


# #629: the run config's keys setup writes itself (and rewrites on every setup run); any other key is the user's - a
# "sampling" or "mcp_servers" block, "allowed_hosts", "cors_origins", "open_browser" - and is kept when setup runs
# again.  The same for the engine options: setup writes SETUP_ARGS, any other one in "args" was added by hand.
SETUP_KEYS = frozenset({"exe", "args", "cwd", "log", "lib_dirs", "model_name", "port", "gpu", "host", "api_key"})
SETUP_ARGS = frozenset({"-m", "--ctx", "--park-mib", "--expert-profile"})


def flag_value(args: list, flag: str):
    """The value after `flag` in an argument list, or None."""
    args = [str(x) for x in args]
    for i, a in enumerate(args[:-1]):
        if a == flag:
            return args[i + 1]
    return None


def set_flag(args: list, flag: str, value: str) -> None:
    """`flag value` in an argument list: the value replaced, or both added."""
    if flag in args[:-1]:
        args[args.index(flag) + 1] = value
    else:
        args += [flag, value]


def arg_pairs(args: list) -> list:
    """An engine argument list as (option, value or None) pairs: the value is the next item when it is no option."""
    args, pairs, i = [str(x) for x in args], [], 0
    while i < len(args):
        val = args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("-") else None
        pairs.append((args[i], val))
        i += 1 if val is None else 2
    return pairs


def engine_args(first: Path, ctx: int, park: int, profile: Path | None) -> list[str]:
    """What setup writes into the config's "args", for strata-glm (the server adds --serve and --eos-ids): the model's
    first shard (the engine finds the others beside it), the context, the RAM for parked conversations and the expert
    profile (the use counts it loads at start and saves at exit, so the VRAM cache starts warm)."""
    args = ["-m", str(first), "--ctx", str(ctx), "--park-mib", str(park)]
    if profile is not None:
        args += ["--expert-profile", str(profile)]
    return args


def carry_over(old: dict, cfg: dict) -> list[str]:
    """#629: setup run again for an installed model keeps what the user added to its run config: every key setup does
    not write (SETUP_KEYS), every engine option setup does not write (SETUP_ARGS), and an expert profile of their own
    (#775) that still exists.  `cfg` (the new config) is updated in place; the names of what was kept are returned."""
    kept = []
    for k, v in old.items():
        if k not in SETUP_KEYS and k not in cfg:
            cfg[k] = v
            kept.append(k)
    oargs, nargs = old.get("args"), cfg.get("args")
    if isinstance(oargs, list) and isinstance(nargs, list):
        for flag, val in arg_pairs(oargs):
            if flag in SETUP_ARGS or flag in nargs:
                continue
            nargs += [flag] if val is None else [flag, val]
            kept.append(f"args {flag}")
        mine, ours = flag_value(oargs, "--expert-profile"), flag_value(nargs, "--expert-profile")
        if mine and ours and mine != ours and Path(mine).is_file():
            nargs[nargs.index("--expert-profile") + 1] = mine
            kept.append("args --expert-profile")
    return kept


def write_setup_config(cfg_path: Path, cfg: dict, source: Path | None = None) -> None:
    """#629: setup's run config, written over an earlier one for the same model without losing what the user added
    to it (carry_over), the earlier file kept as strata-<model>.json.bak when it changes.  `source`: an earlier
    install's config to carry the keys over from when this folder has none yet (a copy set up like the last one)."""
    old_path = cfg_path if cfg_path.is_file() else source
    old = None
    if old_path is not None and old_path.is_file():
        try:
            old = json.loads(old_path.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError):
            pass
        if not isinstance(old, dict):
            old = None
    kept = carry_over(old, cfg) if old is not None else []
    if cfg_path.is_file() and old != cfg:
        bak = cfg_path.with_name(cfg_path.name + ".bak")
        try:
            shutil.copyfile(cfg_path, bak)
            say(f"  the earlier run config is kept as {bak.name}")
        except OSError as e:
            warn(f"could not keep a copy of the earlier {cfg_path.name} ({e.strerror or e})")
    write_config(cfg_path, cfg)
    if kept:
        ok(f"kept from your earlier {old_path.name}: " + ", ".join(kept))


def readable_config(path: Path) -> bool:
    """#459: a config that parses as a JSON object; any other gets a one-line warning naming it."""
    text = None
    try:
        text = path.read_text(encoding="utf-8-sig")
        if isinstance(json.loads(text), dict):
            return True
        why = "not a JSON object"
    except OSError as e:
        why = e.strerror or str(e)
    except ValueError:                                 # JSONDecodeError, or bytes that are not UTF-8
        why = "the file is empty" if text is not None and not text.strip() else "not valid JSON"
    warn(f"skipped the earlier config {path} ({why}): setting this copy up without it")
    return False


def model_config_files(folder: Path) -> list:
    """#346: the strata-glm-*.json files of a folder that can be model configs. The server keeps a config's Chat
    settings next to it as strata-glm-<size>.shared-settings.json: it matches the pattern but is no config."""
    return [p for p in folder.glob("strata-glm-*.json") if not p.name.endswith(".shared-settings.json")]


def older_configs(folder: Path | None = None) -> list:
    """The strata-*.json model configs of the Qwen version of Strata this one replaced: never started here."""
    return [p for p in (folder or ROOT).glob("strata-*.json")
            if not p.name.startswith("strata-glm-") and not p.name.endswith(".shared-settings.json")]


def previous_config(elsewhere_first: list, settings: dict):
    """The most recently used model config of another GLM Strata folder on this PC, for a folder that has none yet.
    One that does not parse (an empty or cut-off file, #459) is skipped with a warning."""
    cands = []
    for folder in [*elsewhere_first, *other_installs(settings)]:
        cands += model_config_files(folder)
    cands = [c for c in dict.fromkeys(cands) if c.is_file()]
    return next((c for c in sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True) if readable_config(c)), None)


def choices_from_config(cfg_path: Path) -> dict:
    """The setup answers a config was written with (size, context, parking, network, GPUs, where its shards are)."""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    tag = cfg_path.stem[len("strata-"):]
    model = next((m for m in MODELS if model_tag(m) == tag), None)
    a = cfg.get("args") or []
    first = flag_value(a, "-m")
    ctx, park = flag_value(a, "--ctx"), flag_value(a, "--park-mib")
    return {"model": model, "context": int(ctx) if (ctx or "").isdigit() else None,
            "park_mib": int(park) if (park or "").isdigit() else None,
            "host": cfg.get("host"), "api_key": cfg.get("api_key"), "port": cfg.get("port"), "gpu": cfg.get("gpu"),
            "gguf_dir": str(Path(first).parent) if first and Path(first).is_file() else None}


def model_config(path: Path) -> bool:
    """#549: a model's run config (a JSON object with "exe" and "args"). Any other strata-glm-*.json in the folder (a
    file of the user's own, a cut-off one) is skipped with a warning naming it instead of stopping setup."""
    try:
        cfg = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(cfg, dict) and cfg.get("exe") and isinstance(cfg.get("args"), list):
            return True
        why = 'no "exe" or "args"'
    except OSError as e:
        why = e.strerror or str(e)
    except ValueError:
        why = "not valid JSON"
    warn(f"skipped {path.name} ({why}): it is not a Strata model config")
    return False


def installed_configs():
    return [p for p in sorted(model_config_files(ROOT), key=lambda p: p.stat().st_mtime, reverse=True)
            if model_config(p)]


def write_run_script(tag, cfg_path, port, open_browser=True):
    """run-<tag>.sh: the server with this config; `open_browser` False (#609: --no-browser) leaves --open out."""
    serve = server_command(cfg_path, port, open_browser)
    script = ROOT / f"run-{tag}.sh"
    script.write_text(f"#!/bin/sh\ncd {shlex.quote(str(ROOT))}\nexec " + " ".join(shlex.quote(x) for x in serve) + "\n",
                      encoding="utf-8")
    script.chmod(0o755)
    return script


# ------------------------------------------------------------------------------------------------ start
def update_install(have: list, a) -> int:
    """#475: `setup.py --update` (update.sh, after its git pull): what a plain ./setup.sh does to an install before it
    starts the model, without starting it - the Python packages and the engine (compiled again when its source
    changed; --build: always).  No question is asked and the model files are not touched."""
    have = [p for p in have if model_config(p)]        # #549: a strata-glm-*.json that is no model config is skipped
    if not have:
        say("  No model is installed in this Strata folder yet: run ./setup.sh to set it up - it finds an earlier")
        say("  install's model files in the data folder and reuses them.")
        return 0
    python_packages()
    update_installed_engine(force=a.build)
    for cfg_path in have:
        ok(f"{cfg_path.name}: up to date")
    say()
    ok(f"Strata is updated (engine {source_version()}). Start the model with ./setup.sh when you want it.")
    return 0


def settings_summary(cfg: dict, port=None) -> str:
    """#564: the settings a start uses, in one line: the config's engine options (the model's file paths left out)
    and the server's own fields, so a change made by hand to strata-glm-<size>.json can be checked without the log."""
    out_ = []
    for flag, val in arg_pairs(cfg.get("args") or []):
        if not flag.startswith("--"):
            continue                                   # -m: the model file
        if val is not None and ("/" in val or val.lower().endswith((".gguf", ".profile"))):
            continue                                   # a path: --expert-profile
        out_.append(flag if val is None else f"{flag} {val}")
    srv = [f"{cfg.get('host', '127.0.0.1')}:{port or cfg.get('port', PORT)}"]
    if cfg.get("api_key"):
        srv.append("api key set")
    if cfg.get("open_browser") is False:               # #609
        srv.append("no browser")
    for k in ("gpu", "fit_max_tokens", "reasoning_budget_tokens", "anthropic_thinking"):
        if cfg.get(k) is not None:
            v = cfg[k]
            srv.append(f"{k} {','.join(map(str, v)) if isinstance(v, list) else str(v).lower() if isinstance(v, bool) else v}")
    return " ".join(out_) + ("; " if out_ else "") + "server " + ", ".join(srv)


def model_gb(cfg: dict) -> float:
    """The size (GB) of the model files a config runs: every shard of its -m file."""
    first = flag_value(cfg.get("args") or [], "-m")
    try:
        return sum(s.stat().st_size for s in shard_paths(Path(first)) if s.exists()) / 1e9 if first else 0.0
    except OSError:
        return 0.0


def server_command(cfg_path: Path, port, open_browser: bool) -> list[str]:
    """The server with this config: .venv's Python, serve/server.py with the strata engine, the port."""
    return [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
            "--port", str(port)] + (["--open"] if open_browser else [])


def start(cfg_path: Path, port: int | None, gpu: int | list | None = None, keep=None) -> int:
    """Start the server with a model's config.  gpu: --gpus (a list: the model runs on these cards from now on, saved
    in its config) or --gpu N (this start only).  keep: settings given on this start that the model keeps from now on
    (--host, --api-key, --park-mib, --browser / --no-browser)."""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
    args = cfg["args"]
    first = flag_value(args, "-m")
    missing = [p for p in [cfg["exe"], first] if p and not Path(p).exists()]
    if missing:
        fail(f"{cfg_path.name} refers to missing files: {missing[0]}", "run ./setup.sh --setup to repair it")
    keep = {k: v for k, v in (keep or {}).items() if v is not None}
    saved = []
    park = keep.pop("park_mib", None)
    if park is not None and flag_value(args, "--park-mib") != str(park):
        set_flag(args, "--park-mib", str(park))
        saved.append(f"--park-mib {park}")
    for k, v in keep.items():                          # #179: a --host/--api-key on a start was ignored
        if cfg.get(k) != v:
            cfg[k] = v
            saved.append("api key" if k == "api_key" else ("the browser opens" if v else "no browser")
                         if k == "open_browser" else f"{k.replace('_', ' ')} {v}")
    found = gpus()
    if isinstance(gpu, list):                          # --gpus: saved, this model runs on these cards from now on
        check_gpus(gpu, found)
        if cfg.get("gpu") != config_gpu(gpu):
            cfg["gpu"] = config_gpu(gpu)
            saved.append("GPUs " + ",".join(map(str, gpu)))
        gpu = None
    elif gpu is not None:                              # --gpu N: this start only, on that card
        check_gpus([gpu], found)
    if saved:
        write_config(cfg_path, cfg)
        ok("saved for this model: " + ", ".join(saved))
    cfg_path.touch()                                   # the most recently used model
    browser = cfg.get("open_browser") is not False     # #609: "open_browser": false, --no-browser
    cmd = server_command(cfg_path, port or cfg.get("port") or PORT, browser)
    if gpu is not None:                                # before --port: the server's own --gpu, this start only
        cmd[cmd.index("--port"):cmd.index("--port")] = ["--gpu", str(gpu)]
    use = gpu if gpu is not None else cfg.get("gpu")
    byid = {g["index"]: g for g in found}
    cards = [byid[i] for i in (use if isinstance(use, list) else [use] if use is not None else byid) if i in byid]
    if cards:
        ok("GPUs: " + " + ".join(gpu_name(g) for g in cards))
    gb = model_gb(cfg)
    say()
    say("  " + "-" * 100)
    say(f"  Starting {cfg.get('model_name', MODEL_NAME)}: the engine reads "
        + (f"its {gb:.0f} GB of model files" if gb >= 1 else "the model files") + " into RAM before it answers.")
    say("  The first start after a restart reads them from the disk, which takes a few minutes: please wait and don't")
    say("  close this window - " + ("the browser opens when it is ready." if browser else
                                    "the server says when it is ready."))
    say("  Later, closing this window stops the model.")
    say("  " + "-" * 100)
    for n, line in enumerate(textwrap.wrap(f"Settings ({cfg_path.name}): {settings_summary(cfg, port)}", 100,
                                           break_on_hyphens=False)):   # #564: what this start uses
        say(("  " if n == 0 else "    ") + line)
    if os.environ.get("STRATA_EXECV"):
        # Replace this process instead of spawning a child. The Docker image sets STRATA_EXECV=1, so there the server
        # is PID 1 and docker stop's SIGTERM reaches the process that can answer the engine with QUIT.
        os.execv(cmd[0], cmd)
    return subprocess.call(cmd)


# ------------------------------------------------------------------------------------------------ main
def check_pc(a) -> tuple:
    """Step 1: the NVIDIA GPUs (compute capability 7.5 or newer) and their driver, the RAM and the CPU.  Stops when the
    engine cannot run here.  Returns (every GPU, the ones this install runs on, whether the config names them, RAM)."""
    step(1, "checking your PC")
    found = gpus()
    if not found:
        fail("no NVIDIA GPU found (nvidia-smi did not answer)",
             f"install the NVIDIA driver ({MIN_DRIVER} or newer; Ubuntu: sudo ubuntu-drivers install) and restart")
    sel, name_gpus = choose_gpus(a, found)            # a choice it cannot use stops with the table of GPUs
    gpu_table(found)
    cards = [g for g in found if g["index"] in sel]
    ok("GPUs: " + " + ".join(gpu_name(g) for g in cards) + (" (the model's layers are shared across them)"
                                                             if len(cards) > 1 else ""))
    old = min(cards, key=driver_major)
    if driver_major(old) < MIN_DRIVER:
        fail(f"the NVIDIA driver is too old ({old['driver']}; {MIN_DRIVER} or newer is needed: the engine is built "
             "with CUDA 13)", "update it (Ubuntu: sudo ubuntu-drivers install, or https://www.nvidia.com/drivers), "
                              "restart, and run this again")
    ok(f"driver {old['driver']}, compute capability " + ", ".join(sorted({cc(g) for g in cards}))
       + f" (the engine is compiled for {cuda_archs(card_archs(cards))})")
    ram = ram_gib()
    fits = [m for m in MODELS if model_fit(m, ram) != "no"]
    (ok if fits else warn)(f"RAM: {ram:.0f} GiB" + ("" if fits else " - less than either size's experts need beside "
                                                                     "Linux and the server"))
    cpu, avx2 = cpu_info()
    if avx2:
        ok(f"CPU: {cpu} (AVX2)")
    else:
        warn(f"CPU: {cpu} has no AVX2: the experts the GPUs do not hold are computed on the CPU, several times slower "
             "than on a CPU with AVX2")
    return found, sel, name_gpus, ram


def check_only(a) -> int:
    """--check: step 1 and which size fits this PC's RAM, nothing installed or moved.  Exit code 1 when neither does."""
    _found, _sel, _named, ram = check_pc(a)
    where = Path(a.models_dir or a.data_dir or load_settings().get("data_dir") or ROOT.parent).expanduser().resolve()
    while not where.exists():                          # a folder setup would make: the drive it would be on
        where = where.parent
    free = shutil.disk_usage(where).free / 1e9
    say()
    for m, d in MODELS.items():
        verdict = {"fits": "fits", "tight": "fits, with little RAM to spare", "no": "does not fit"}
        say(f"  {m:10s} {d['bits']}: ~{experts_gib(m):.0f} GiB of experts in RAM - "
            f"{verdict[model_fit(m, ram)]} ({fit_text(m, ram)}); download ~{d['download_gb']} GB"
            + ("" if free >= d["download_gb"] + 1 else f", but only {free:.0f} GB of disk space is free"))
    if not any(model_fit(m, ram) != "no" for m in MODELS):
        # #977: every size says "does not fit", so the verdict says so too (and the exit code, for scripts)
        say(f"\nThis PC cannot run GLM-5.3-Flash yet: neither size fits {ram:.0f} GiB of RAM. --model NAME --yes "
            "installs one anyway, slowly.")
        return 1
    say(f"\nThis PC can run GLM-5.3-Flash ({recommended_model(ram)} recommended). Run it again without --check to "
        "install.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS),
                    help="UD-IQ3_XXS = 3-bit (~120 GB download, ~109 GiB of experts in RAM), IQ4_XS = 4-bit (~170 GB, "
                         "~151 GiB; UD-Q4_K_XL when the repository has no IQ4_XS). Default: the one this PC's RAM fits")
    ap.add_argument("--context", type=int, metavar="N",
                    help=f"the longest conversation in tokens (up to {MAX_CONTEXT}); each token takes 11 KB of VRAM "
                         "(1.5 GB at 131072)")
    ap.add_argument("--park-mib", type=int, metavar="N",
                    help=f"RAM in MiB for parked conversations and checkpoints (default {PARK_MIB}, {PARK_MIB_TIGHT} "
                         "where little RAM is left beside the experts); saved for this model, also when starting")
    ap.add_argument("--port", type=int, help=f"the server's port (default {PORT}); when starting an installed model it is "
                                             "for that start only, and the model's own port is the default")
    ap.add_argument("--gpu", help="one GPU, numbered as nvidia-smi numbers them (default: every GPU that can run the "
                                  "engine; with --setup it is saved, when starting it is for that start only)")
    ap.add_argument("--gpus", help="the GPUs the model's layers are shared across, as nvidia-smi numbers them: \"0,2\", "
                                   "or \"all\". Saved, also when starting")
    ap.add_argument("--host", help="where the server listens: 127.0.0.1 = this PC only (default), 0.0.0.0 = also other "
                                   "devices on your network (issue #26; set --api-key too)")
    ap.add_argument("--api-key", help="require this key from clients (recommended with --host 0.0.0.0); several keys: "
                                      "key1,key2")
    ap.add_argument("--no-browser", dest="browser", action="store_false", default=None,
                    help="do not open the chat page in the browser when the model is ready (for a harness or an app "
                         "that uses the API; remembered for this model, also in run-<model>.sh)")
    ap.add_argument("--browser", dest="browser", action="store_true",
                    help="open the chat page again when the model is ready (the default; undoes --no-browser)")
    ap.add_argument("--data-dir", help="where the model files go (120-170 GB): default Strata-GLM-data next to this "
                                       "folder, remembered for every Strata folder on this PC")
    ap.add_argument("--models-dir", help="where the GGUF files go (default: <data folder>/models)")
    ap.add_argument("--gguf-dir", help="use GGUF files you already have: a folder with every shard of the size "
                                       "(<name>-00001-of-0000N.gguf ...), also in a sub-folder")
    ap.add_argument("--yes", action="store_true", help="accept the recommended answers")
    ap.add_argument("--setup", action="store_true", help="set up the other size or change settings")
    ap.add_argument("--no-start", action="store_true", help="install only, do not start the model")
    ap.add_argument("--update", action="store_true",
                    help="update the Python packages and the engine as a start would, without starting the model "
                         "(update.sh runs it after a git pull)")
    ap.add_argument("--build", action="store_true", help="compile the engine again, also when it is up to date")
    ap.add_argument("--check", action="store_true", help="only check this PC and exit")
    a = ap.parse_args()
    if a.context is not None and not 1024 <= a.context <= MAX_CONTEXT:
        ap.error(f"--context takes a number of tokens from 1024 to {MAX_CONTEXT}, e.g. --context 131072")
    if a.park_mib is not None and a.park_mib < 0:
        ap.error("--park-mib takes a number of MiB, 0 or more, e.g. --park-mib 4096")
    if a.gpu is not None:                             # --gpu 0,2 means --gpus 0,2 (a user tried it: issue report)
        if "," in a.gpu:
            a.gpus, a.gpu = a.gpus or a.gpu, None
        elif a.gpu.strip().isdigit():
            a.gpu = int(a.gpu)
        else:
            ap.error(f"--gpu takes a GPU number as nvidia-smi numbers them, e.g. --gpu 1 (or --gpus 0,2), not {a.gpu!r}")
    say("Strata - GLM-5.3-Flash on NVIDIA GPUs and system RAM")
    if not sys.platform.startswith("linux"):
        fail("Strata's GLM engine runs on Linux only", "run ./setup.sh on a Linux PC with an NVIDIA GPU")
    if a.check:                                        # before the data folder: a check moves and writes nothing
        return check_only(a)
    data, elsewhere = data_folder(a.data_dir)          # the model files: in the data folder, found from any copy
    roots = [data, *elsewhere]
    if a.models_dir is None:
        a.models_dir = str(data / "models")
    older = older_configs()
    if older:
        say(f"  ({', '.join(p.name for p in older)}: a config of the earlier Qwen version of Strata; this version "
            "runs GLM-5.3-Flash only and leaves it as it is)")

    # ---- 0. already installed: just start it
    have = installed_configs()
    if a.update:                                       # #475: update.sh - never starts the model
        return update_install(have, a)
    explicit = a.setup or a.model or a.no_start
    adopted = None                                     # #629: the earlier install this copy is set up like
    if not have and not explicit:                      # a new copy of Strata (an update unzipped elsewhere): set it
        prev = previous_config(elsewhere, load_settings())   # up like the last one, from the files already there
        if prev is not None:
            ch = choices_from_config(prev)
            if ch["model"]:
                say(f"  Found your earlier install in {prev.parent} ({prev.stem[len('strata-'):]}): setting up this "
                    "copy the same way - the model files are reused, nothing big is downloaded.")
                a.model, a.context = ch["model"], a.context or ch["context"]
                a.park_mib = a.park_mib if a.park_mib is not None else ch["park_mib"]
                a.host, a.api_key, a.port = a.host or ch["host"], a.api_key or ch["api_key"], a.port or ch["port"]
                a.gguf_dir = a.gguf_dir or ch["gguf_dir"]
                if a.gpus is None and a.gpu is None and ch.get("gpu") is not None:
                    a.gpus = ",".join(map(str, ch["gpu"] if isinstance(ch["gpu"], list) else [ch["gpu"]]))
                a.yes = True
                adopted = prev
    port = a.port or PORT                              # a new install's port (issue #32: --port for an existing one)
    keep = {"host": a.host, "api_key": a.api_key, "park_mib": a.park_mib, "open_browser": a.browser}
    if have and not explicit:
        update_installed_engine(force=a.build)
        run_gpu = parse_gpus(a.gpus, gpus()) if a.gpus else a.gpu
        if len(have) == 1:
            return start(have[0], a.port, run_gpu, keep=keep)
        say()
        for i, c in enumerate(have, 1):
            say(f"  {i}) {c.stem[len('strata-'):]}")
        say(f"  {len(have) + 1}) set up the other size / change settings")
        pick = int(ask("Which one?", [str(i) for i in range(1, len(have) + 2)], "1", a.yes))
        if pick <= len(have):
            return start(have[pick - 1], a.port, run_gpu, keep=keep)

    # ---- 1. the PC
    found, sel, name_gpus, ram = check_pc(a)
    cards = [g for g in found if g["index"] in sel]
    usable = [g for g in found if gpu_problem(g) is None]
    vram = sum(g["vram_gb"] for g in cards)

    # ---- 2. the questions
    step(2, "your choices")
    names = list(MODELS)
    rec = recommended_model(ram)
    if not a.model:
        for i, m in enumerate(names, 1):
            d = MODELS[m]
            say(f"  {i}) {m:10s} {d['bits']}, {d['about']}: download ~{d['download_gb']} GB, ~{experts_gib(m):.0f} "
                f"GiB of experts in RAM ({fit_text(m, ram)})" + ("   (recommended)" if m == rec else ""))
    model = a.model or names[int(ask("Which size?", [str(i) for i in range(1, len(names) + 1)],
                                     str(names.index(rec) + 1), a.yes)) - 1]
    d = MODELS[model]
    fit = model_fit(model, ram)
    if fit == "no":
        smaller = next((m for m in names if model_fit(m, ram) != "no"), None)
        confirm_risk(f"{model}'s experts (~{experts_gib(model):.0f} GiB) do not fit beside Linux and the server in "
                     f"this PC's {ram:.0f} GiB of RAM: the OS would page them from the disk, so it would be very slow, "
                     "or not start at all", bool(a.model), a.yes,
                     f"{model} needs about {experts_gib(model) + RAM_LEFT_MIN_GIB:.0f} GiB of RAM; this PC has "
                     f"{ram:.0f} GiB",
                     (f"choose {smaller}, " if smaller else "add RAM, ") + f"or --model {model} --yes to install it "
                     "anyway", "  Install it anyway?")
        warn(f"installing {model} with {ram:.0f} GiB of RAM, as you chose")
    park = park_mib(model, ram, a.park_mib)
    if fit == "tight" and a.park_mib is None:
        warn(f"{model} leaves only ~{ram_left(model, ram):.0f} GiB of RAM beside its experts: parked conversations get "
             f"{PARK_MIB_TIGHT} MiB instead of {PARK_MIB} (--park-mib {PARK_MIB_TIGHT}); close other big programs "
             "while it runs")
    ok(f"size: {model} ({d['bits']})")
    rec_ctx = recommended_context(vram)
    if a.context:
        ctx = a.context
    else:
        say()
        say("  Context = the longest conversation the model can hold (your chat, files, tool output). Each token of it")
        say("  takes 11 KB of VRAM in the attention cache, which the expert cache gives up:")
        for i, c in enumerate(CONTEXTS, 1):
            say(f"  {i}) {context_label(c):>4s} tokens  {mla_gb(c):4.1f} GB of VRAM"
                + ("   (recommended for your GPUs)" if c == rec_ctx else "")
                + ("   (the model's whole context)" if c == MAX_CONTEXT else ""))
        ctx = CONTEXTS[int(ask("Context?", [str(i) for i in range(1, len(CONTEXTS) + 1)],
                               str(CONTEXTS.index(rec_ctx) + 1), a.yes)) - 1]
    if mla_gb(ctx) > 0.5 * vram:
        warn(f"a {ctx}-token context takes {mla_gb(ctx):.1f} GB of the GPUs' {vram:.0f} GB for its attention cache: few "
             f"experts stay cached in VRAM, so answers are slower. Kept as you chose ({context_label(rec_ctx)} is the "
             "recommended one)")
    ok(f"context: {ctx} tokens ({mla_gb(ctx):.1f} GB of VRAM for the attention cache)")

    # the model's files: --gguf-dir, a finished download (here or on another drive), or the repository's listing
    models_root = Path(a.models_dir)
    plan = []                                          # (listed file, where it goes) for step 5
    found_model = None if a.gguf_dir else find_model(model, [models_root, *[r / "models" for r in roots]])
    if a.gguf_dir:
        quant, shards = gguf_dir_model(Path(a.gguf_dir).expanduser(), model)
        ok(f"model files: {shards[0].parent} ({quant}, {len(shards)} file{'s' if len(shards) > 1 else ''})")
    elif found_model:
        quant, shards = found_model
        ok(f"model files: {shards[0].parent} (already downloaded)")
    else:
        say(f"  Listing the model's files on {hf_endpoint()} ({REPO}) ...")
        pick = pick_shards(repo_files(), model)
        if pick is None:
            other = list(d.get("fallback", ()))
            fail(f"{REPO} has no {model} files" + (f" (nor {', '.join(other)})" if other else ""),
                 "the repository may have renamed them: download them by hand (hf download) and run setup with "
                 "--gguf-dir <their folder>")
        quant, files = pick
        folder = models_root / model_tag(quant)
        shards = [folder / f["path"].rsplit("/", 1)[-1] for f in files]
        plan = list(zip(files, shards))
        total = sum(f["size"] for f in files) / 1e9
        on_disk = sum(p.stat().st_size for s in shards for p in (s, s.with_name(s.name + ".part")) if p.is_file()) / 1e9
        ok(f"{quant}: {len(files)} file{'s' if len(files) > 1 else ''}, {total:.0f} GB")
        if quant != model:
            # its experts, taken as the same share of its files as the size's own: what it leaves of the RAM (GiB)
            experts = total * d["experts_gb"] / d["download_gb"] * 1e9 / GIB
            warn(f"the repository has no {model} files: the {d['bits']} model is its {quant} ({total:.0f} GB, about "
                 f"{experts:.0f} GiB of experts in RAM: "
                 + (f"~{ram - experts:.0f} GiB of RAM left beside them)" if ram >= experts else
                    f"more than this PC's {ram:.0f} GiB)"))
            if fit != "no" and ram - experts < RAM_LEFT_MIN_GIB:
                confirm_risk(f"{quant}'s experts do not fit beside Linux and the server in this PC's {ram:.0f} GiB of "
                             "RAM: the OS would page them from the disk, so it would be very slow, or not start at all",
                             bool(a.model), a.yes,
                             f"{quant} needs about {experts + RAM_LEFT_MIN_GIB:.0f} GiB of RAM; this PC has "
                             f"{ram:.0f} GiB",
                             f"choose {names[0]}, or --model {model} --yes to install it anyway", "  Install it anyway?")
            elif ram - experts < RAM_LEFT_GIB and a.park_mib is None and park != PARK_MIB_TIGHT:
                park = PARK_MIB_TIGHT
                warn(f"parked conversations get {PARK_MIB_TIGHT} MiB instead of {PARK_MIB} (--park-mib {PARK_MIB_TIGHT})")
        need = max(total - on_disk, 0) + 1
        if free_gb(folder) < need:
            fail(f"not enough free disk space in {folder}: need ~{need:.0f} GB" +
                 (f" ({on_disk:.0f} GB of the model is already there)" if on_disk >= 1 else ""),
                 "use --models-dir on a bigger drive (or --data-dir, remembered for later)")

    # ---- 3. python packages
    step(3, "Python packages")
    python_packages()

    # ---- 4. the engine
    step(4, "the Strata engine")
    llama = get_llama_cpp()
    ok(f"llama.cpp {LLAMA_CPP_COMMIT[:7]} (ggml, gguf-py)")
    exe = build_engine(card_archs(usable), a.yes, llama, force=a.build)

    # ---- 5. the model files
    step(5, f"downloading GLM-5.3-Flash {quant}")
    if plan:
        say(f"  The model files go in {shards[0].parent}")
        say("  Files you already have: put them there with their original names, or use --gguf-dir <their folder>.")
        if hf_endpoint() != HF_DEFAULT:
            say(f"  Downloading from {hf_endpoint()} (HF_ENDPOINT)")
    for f, s in plan:
        download(hf_file_url(f["path"]), s)
        verify_file(s, f["size"], f["sha256"])
    check_shards(shards)
    ok("model files present")

    # ---- 6. the run config and the start script
    step(6, "writing the start script")
    tag = model_tag(model)
    cfg_path = ROOT / f"strata-{tag}.json"
    cfg = {"exe": str(exe), "args": engine_args(shards[0], ctx, park, data / f"{tag}.profile"), "cwd": str(ROOT),
           "log": str(ROOT / f"strata-{tag}.log"), "lib_dirs": installed_build().get("lib_dirs") or [],
           "model_name": MODEL_NAME, "port": port}
    if name_gpus:
        cfg["gpu"] = config_gpu(sel)                   # the engine is told these cards (issue #51)
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if a.browser is not None:                          # #609: only when given (else an earlier choice is carried over)
        cfg["open_browser"] = a.browser
    write_setup_config(cfg_path, cfg, adopted if adopted is not None and adopted.name == cfg_path.name else None)
    script = write_run_script(tag, cfg_path, port, cfg.get("open_browser") is not False)
    ok(f"run config: {cfg_path.name}; start script: {script.name}")

    say()
    say("All set.")
    say(f"  Chat page:        http://127.0.0.1:{port}/")
    say(f"  API (OpenAI):     http://127.0.0.1:{port}/v1   (model name: anything)")
    say(f"  API (Anthropic):  http://127.0.0.1:{port}/v1/messages")
    if a.host and a.host not in ("127.0.0.1", "localhost"):
        say(f"  Other devices:    the server window prints this PC's address (http://<IP>:{port}/)"
            + ("" if a.api_key else " - no API key set: anyone on your network can use it"))
    say(f"  Next time:        just run ./setup.sh (or ./{script.name}) - it starts right away")
    if a.no_start:
        return 0
    return start(cfg_path, port)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\nstopped.")
        sys.exit(1)
