# Installing Strata

Everything about installing, starting, updating and configuring Strata: GLM-5.3-Flash on Linux with NVIDIA GPUs and
system RAM. The short version is in the [README](../README.md#install). An AI coding assistant can do all of this for
you with [AI_SETUP.md](AI_SETUP.md).

> **Nothing has run on a GPU or on the real model yet.** There are no measured speeds, load times or memory figures.
> The sizes on this page are calculated from the model's shapes, and setup's own figures come from `setup.py`. What
> has been checked: the CPU reference against llama.cpp on small random GLM models (`tools/glm53_selftest.sh`),
> conversation parking (`tools/glm53_park_check.py`), the server end to end on the CPU engine
> (`tools/glm53_server_check.sh`), the Python tests, the engine's C++ unit tests (`ctest`) and setup's own tests
> (`tools/test_setup_*.py`, which mock the GPUs, the downloads and the build). Setup itself, the systemd unit and the
> Docker image below have not been run on a real machine yet. The steps still to run on a real machine are in
> [GLM53_PORT.md](GLM53_PORT.md).

> **On this page:** [What you need](#what-you-need) · [Install with ./setup.sh](#install-with-setupsh) ·
> [Setup's options](#setups-options) · [Where files go](#where-files-go) · [The build](#the-build) ·
> [The run config](#the-run-config) · [Choosing the context](#choosing-the-context) ·
> [Several GPUs](#several-gpus-and-their-order) · [On napster-ai](#on-napster-ai) ·
> [As a service](#running-it-as-a-service) · [Updating](#updating) · [Docker](#docker) ·
> [By hand](#building-and-starting-by-hand)

## What you need

| | |
| --- | --- |
| **OS** | Linux on x86-64. Setup stops on any other operating system. On Ubuntu 22.04 and 24.04 setup can install the CUDA toolkit itself; on other distributions you install it. |
| **GPU** | One or more NVIDIA cards of compute capability 7.5 or newer (RTX 20, 30, 40, 50 series). Setup leaves out a card it cannot use and says why. It has no minimum VRAM: how much VRAM the layers besides the experts take on the real model is not measured yet. |
| **Driver** | NVIDIA 580 or newer when the engine is built with CUDA 13 (the toolkit setup installs when none is there); 525 or newer when setup builds it with a CUDA 12 toolkit you have installed (CUDA 12's minor version compatibility; with a driver older than 580 setup takes that toolkit, also beside a CUDA 13 one). Setup stops on an older one and says which it needs (Ubuntu: `sudo ubuntu-drivers install`, then restart). |
| **RAM** | Every routed expert stays in RAM: about 117 GB (109 GiB) for the 3-bit size, about 162 GB (151 GiB) for the 4-bit (see the table below). |
| **CPU** | AVX2. Setup warns on a CPU without it but goes on: the experts the GPUs do not hold are computed on the CPU. |
| **Disk** | About 120 GB (3-bit) or 170 GB (4-bit) free for the model files. Setup checks this before it downloads. The CUDA toolkit, when setup installs it, takes more (setup says about 8-10 GB with the compiler). |
| **Python** | 3.10 or newer with `venv`. `./setup.sh` installs it with apt, dnf or pacman when it is missing (asks for your sudo password). |
| **Network** | huggingface.co (the model; or a mirror, `HF_ENDPOINT`), github.com (llama.cpp's source at a pinned commit), PyPI (the Python packages), and developer.download.nvidia.com when setup installs CUDA. |

The two sizes are Unsloth's GGUF files from [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF).
Setup lists the repository's files when it runs and takes the shards of the size you chose.

| Size | `--model` | Download | Experts in RAM | A good fit from | Tight (warned) from |
| --- | --- | --- | --- | --- | --- |
| 3-bit | `UD-IQ3_XXS` | ~120 GB | ~117 GB (109 GiB) | 149 GiB of RAM | 121 GiB |
| 4-bit | `IQ4_XS` (`UD-Q4_K_XL` when the repository has no IQ4_XS) | ~170 GB | ~162 GB (151 GiB) | 191 GiB | 163 GiB |

These figures are calculated from the model's shapes and written into `setup.py`; they are not measured. The
downloads and the disk are in GB (10^9 bytes); the RAM is in GiB (2^30 bytes), as Linux reports it, and setup
compares the experts with it in GiB. A good fit leaves 40 GiB or more beside the experts, for Linux, the server and
parked conversations. Below 12 GiB left, the OS would read experts from the disk, so setup stops unless you insist
(`--model NAME --yes`). Setup reads the RAM from `/proc/meminfo` (the number `free -g` prints), which is a little
less than the memory installed. Setup's tests take 188 GiB there for a 192 GB PC; setup then prints about 79 GiB
left beside the 3-bit experts (recommended) and about 37 GiB beside the 4-bit ones (tight: parked conversations get
4096 MiB instead of 8192). When the 4-bit size falls back to UD-Q4_K_XL, setup works out that file's experts from
its download size and says what it leaves.

The disk matters most at the first start after a reboot: the engine reads every expert from the disk before it says it
is ready. 117 GB at 550 MB/s (the top speed of a SATA SSD) is about 3.5 minutes, 162 GB about 5 minutes. That is
calculated, not measured.

## Install with ./setup.sh

```bash
git clone -b glm53-flash https://github.com/Rarebigshot17/Strata.git
cd Strata
./setup.sh
```

Without git: download https://github.com/Rarebigshot17/Strata/archive/refs/heads/glm53-flash.zip, unzip it and run
`./setup.sh` in the folder it makes. `./setup.sh` makes a private Python environment in `.venv/` (your system Python is
left alone) and runs `setup.py` with it. Every option goes to `setup.py`.

What the first run does. Each step is skipped when it is already done, so a stopped setup picks up where it stopped:

1. **Checks the PC**: the NVIDIA GPUs (nvidia-smi), the driver, the RAM and the CPU.
2. **Asks the questions** (below), then lists the model's files on Hugging Face and checks the free disk space.
3. **Installs the Python packages** into `.venv` (pinned in `requirements.txt`).
4. **Compiles the engine** for your GPUs: downloads llama.cpp's source at the pinned commit, installs the build tools
   when they are missing (it asks), and builds `strata-glm` and `strata-glm-ref` into `build/`. See
   [The build](#the-build).
5. **Downloads the model's GGUF shards**. A download resumes where it stopped. Each shard is checked against the size
   and SHA-256 the repository lists; a wrong one is deleted, and the next run downloads it again. Then setup reads
   each shard's header to check that the file is whole.
6. **Writes the run config and the start script** (`strata-glm-<size>.json`, `run-glm-<size>.sh`) and starts the
   model.

At the end it prints the addresses:

```
  Chat page:        http://127.0.0.1:8101/
  API (OpenAI):     http://127.0.0.1:8101/v1   (model name: anything)
  API (Anthropic):  http://127.0.0.1:8101/v1/messages
  Next time:        just run ./setup.sh (or ./run-glm-ud-iq3_xxs.sh) - it starts right away
```

Port 8101 is the default because 3000, 8000 and 8080 are often taken by other services.

### The questions

Press Enter to take the answer in brackets. `--yes` takes it for every question. Without `--yes`, setup needs a
terminal: when its input ends (a pipe, a script) it stops and says to pass `--yes`.

| Question | Choices | Default |
| --- | --- | --- |
| **Which size?** | 1) `UD-IQ3_XXS` 3-bit, 2) `IQ4_XS` 4-bit. Each line gives the download, the RAM its experts take and what that leaves on this PC. Not asked with `--model`. | The biggest size that fits well; the 3-bit when none does. |
| **Context?** | 1) 32K, 2) 128K, 3) 256K, 4) 512K, 5) 1M tokens. Each line says the VRAM its attention cache takes ([Choosing the context](#choosing-the-context)). Not asked with `--context`. | 128K when the chosen GPUs have 16 GB of VRAM or more in all, else 32K. |
| **Install them now?** | Asked only when the C++ compiler or the CUDA toolkit is missing. `y` installs them with `sudo apt-get`: `build-essential` on any system with apt, the CUDA toolkit on Ubuntu 22.04 / 24.04 only. Without apt, or for CUDA on another distribution, setup stops and says what to install. | `y` |
| **Install it anyway?** | Asked only when the size you chose leaves less than 12 GiB of RAM beside its experts. | `n` |
| **Which one?** | Asked by a plain `./setup.sh` when both sizes are installed: start one of them, or set up the other size / change settings. | `1`, the one started last |

Setup recommends, it never forces. A bigger choice than it recommends (the 4-bit model on a PC it nearly fills, a long
context on little VRAM) is kept, with a warning that says what it risks. With `--yes`, a size you name with `--model` is
your consent to a size that does not fit; `--yes` alone keeps the stop.

### Later starts

`./setup.sh` with no options starts the installed model right away. Before it starts, it compiles the engine again when
its source changed (a `git pull`) or the PC has a GPU the build has no code for; if that compile fails, it starts the
engine you had. `./run-glm-<size>.sh` starts the server directly, without those checks.

The server checks that its port is free, starts the engine and opens the port only when the engine says it is ready.
Before that, the engine places its layers on the GPUs and reads every routed expert once into the page cache, so the
first request does not wait for the disk (the warm-up; `--no-warm` in the config's `args` skips it). The server window
shows each step: the layers on each GPU, the expert cache, and the warm-up's progress (`warming the experts: X of Y GB`
in the engine's log). How long this takes on the real model is not measured yet.

The server gives up on a start when the engine has not said it is ready within 900 seconds. While the warm-up makes
progress, the 900 seconds count again from its last progress line; other lines of the engine do not reset them. On a
disk so slow that one step takes longer, set `STRATA_ENGINE_READY_S` (seconds, 0 = no limit) before you start it.

To stop the model, press Ctrl+C or close the terminal. On Ctrl+C (and on SIGTERM) the server stops the engine itself:
`STOP` and `QUIT`, then, 20 seconds apart, a terminate and a kill. The engine writes its expert profile only when it
ends after `QUIT` or when its input closes. Read from the code (not tried on a GPU), these end it sooner, and that
run's profile is not written:

- **Ctrl+C** also reaches the engine, which runs in the server's process group and has no signal handler, so it ends
  at once. When `./setup.sh` started the server, setup also ends the server itself a quarter of a second after
  Ctrl+C.
- **Closing the terminal** sends SIGHUP, which neither the server nor the engine handles: both end at once.

To keep the profile, stop the server process alone: `kill <the server's PID>` (SIGTERM takes the server's Ctrl+C path,
but only the server gets it), `systemctl stop` with the unit [below](#running-it-as-a-service), or `docker stop`.

### Setup's options

`./setup.sh --help` (or `python3 setup.py --help`) prints them. This is that list, as setup.py gives it:

```
  --model {UD-IQ3_XXS,IQ4_XS}
                        UD-IQ3_XXS = 3-bit (~120 GB download, ~109 GiB of experts in RAM), IQ4_XS = 4-bit (~170 GB,
                        ~151 GiB; UD-Q4_K_XL when the repository has no IQ4_XS). Default: the one this PC's RAM fits
  --context N           the longest conversation in tokens (up to 1048576); each token takes 11 KB of VRAM (1.5 GB at
                        131072)
  --park-mib N          RAM in MiB for parked conversations and checkpoints (default 8192, 4096 where little RAM is
                        left beside the experts); saved for this model, also when starting
  --port PORT           the server's port (default 8101); when starting an installed model it is for that start only,
                        and the model's own port is the default
  --gpu GPU             one GPU, numbered as nvidia-smi numbers them (default: every GPU that can run the engine; with
                        --setup it is saved, when starting it is for that start only)
  --gpus GPUS           the GPUs the model's layers are shared across, as nvidia-smi numbers them: "0,2", or "all".
                        Saved, also when starting
  --host HOST           where the server listens: 127.0.0.1 = this PC only (default), 0.0.0.0 = also other devices on
                        your network (issue #26; set --api-key too)
  --api-key API_KEY     require this key from clients (recommended with --host 0.0.0.0); several keys: key1,key2
  --no-browser          do not open the chat page in the browser when the model is ready (for a harness or an app that
                        uses the API; remembered for this model, also in run-<model>.sh)
  --browser             open the chat page again when the model is ready (the default; undoes --no-browser)
  --data-dir DATA_DIR   where the model files go (120-170 GB): default Strata-GLM-data next to this folder, remembered
                        for every Strata folder on this PC
  --models-dir MODELS_DIR
                        where the GGUF files go (default: <data folder>/models)
  --gguf-dir GGUF_DIR   use GGUF files you already have: a folder with every shard of the size
                        (<name>-00001-of-0000N.gguf ...), also in a sub-folder
  --yes                 accept the recommended answers
  --setup               set up the other size or change settings
  --no-start            install only, do not start the model
  --update              update the Python packages and the engine as a start would, without starting the model
                        (update.sh runs it after a git pull)
  --build               compile the engine again, also when it is up to date
  --check               only check this PC and exit
```

What to know about them:

- **`--check`** prints step 1 and, for each size, whether it fits this PC's RAM and the free disk space of the drive
  the model files would go to. It installs, moves and writes nothing, and needs no `.venv`: `python3 setup.py --check`
  works on a fresh clone. Its exit code is 1 when neither size fits, or when it stops at step 1 (no NVIDIA GPU, a
  driver older than the CUDA toolkit needs: 580 for CUDA 13, 525 for CUDA 12).
- **On an installed model**, a plain start keeps these for the next starts too: `--gpus`, `--host`, `--api-key`,
  `--park-mib`, `--browser` / `--no-browser`. These are for that start only: `--port`, `--gpu`.
- **`--context` needs `--setup`** (or `--model`) on an installed model:
  `./setup.sh --setup --model UD-IQ3_XXS --context 262144`. Without `--setup`, `--model` or `--no-start`, setup
  starts the installed model and `--context` has no effect.
- **`--setup --yes` takes the recommended answers again.** Name the size, the context and the parking RAM you want
  (`--model ... --context ... --park-mib ...`), or setup picks its own. Keys you added to the run config by hand are
  kept ([The run config](#the-run-config)), but `host`, `api_key`, `gpu` and `port` are written from this run's options
  only: give `--host`, `--api-key`, `--gpus` and `--port` again, or the model listens on 127.0.0.1:8101 without a key
  on every usable GPU. The earlier config is kept as `strata-glm-<size>.json.bak`.
- **`--gguf-dir`** takes shards you downloaded yourself (for example with `hf download`), also in a sub-folder. Setup
  reads their headers to check that each is whole, and downloads nothing.
- **`--gpu 0,2`** means `--gpus 0,2`.
- **A new copy of Strata** (an update unzipped into another folder) with no model set up yet finds your last install
  and sets itself up the same way, without questions (size, context, parking, host, key, port, GPUs, where the shards
  are). `--setup`, `--model` or `--no-start` turns that off.

### Environment variables

For setup:

| Variable | What it does |
| --- | --- |
| `HF_ENDPOINT` | A Hugging Face mirror (e.g. `https://hf-mirror.com`) for the listing and the downloads. The same size and SHA-256 checks apply. |
| `STRATA_NVCC` | The `nvcc` to build with. When set, it is the only one considered (a path that does not exist is warned about, and setup looks as usual). Otherwise setup takes the newest toolkit on `PATH`, in `$CUDA_PATH` / `$CUDA_HOME`, `/usr/local/cuda*` or `/opt/cuda*` (the newest CUDA 12 one with a driver older than 580: a CUDA 13 engine does not run on it). |
| `CC`, `CXX`, `CUDAHOSTCXX` | The host compilers of the build. `CUDAHOSTCXX` (nvcc's host compiler) defaults to `CXX`. |
| `STRATA_UV=1` | `./setup.sh` makes `.venv` with uv. On NixOS this is the default when uv is installed. |
| `XDG_CONFIG_HOME` | Where setup keeps `strata-glm/settings.json` (default `~/.config`). |
| `STRATA_EXECV=1` | Setup replaces itself with the server instead of starting it as a child. The Docker image sets it. |

For the server (more in [DETAILS.md](DETAILS.md#settings)):

| Variable | What it does |
| --- | --- |
| `STRATA_ENGINE_READY_S` | Seconds a start may go without news from the engine before the server gives up (default 900, counted again from each warm-up progress line; 0 = no limit). |
| `STRATA_ENGINE_STALL_S` | Seconds an engine may be silent during a request while using no CPU time and doing no I/O before the server ends it as frozen (default 90, 0 = off). |
| `STRATA_API_KEY` | The server's API key, as `--api-key` or `"api_key"`. |
| `STRATA_ALLOWED_HOSTS` | Extra host names the server answers to, as `"allowed_hosts"`. |
| `STRATA_MAX_BODY_MIB` | The largest request body in MiB (default 256). |
| `STRATA_DROP_CACHE_ON_EXIT=1` | When the server stops, tell Linux to drop the model files from the page cache. Off by default, so a restart of the same model can read them from RAM. |
| `STRATA_REQUEST_LINES=1` | One line per request on the server's output (for a supervisor such as llama-swap, [LLAMA_SWAP.md](LLAMA_SWAP.md)). |

### Where files go

In the Strata folder (the one with `setup.py`):

| Path | What |
| --- | --- |
| `.venv/` | The private Python environment. |
| `third_party/llama.cpp/` | llama.cpp at the pinned commit: ggml for the build, gguf-py for the tools. |
| `build/` | The CMake build: `strata-glm`, `strata-glm-ref`, and `BUILD.json` (what they were compiled from and for). |
| `strata-glm-<size>.json` | The run config ([below](#the-run-config)). The earlier one is kept as `strata-glm-<size>.json.bak` when setup runs again or the web app's Settings view changes it. |
| `run-glm-<size>.sh` | The start script: `cd` to the Strata folder, then `serve/server.py` with this config and port. |
| `strata-glm-<size>.log` | The engine's log: its own output (stderr), and one line from the server at each start with the command it ran. New starts are appended. |
| `strata-glm-<size>.shared-settings.json` | The web app's Chat settings when you share them with other apps. Written by the server. |

`<size>` is the size in lower case: `ud-iq3_xxs` or `iq4_xs`.

In the data folder, `Strata-GLM-data` next to the Strata folder (or where `--data-dir` put it):

| Path | What |
| --- | --- |
| `models/glm-<quant>/` | The GGUF shards, each with a `<shard>.done` mark that keeps its checked SHA-256. `<quant>` is the files' own quantization: `glm-ud-q4_k_xl` when the 4-bit size fell back to UD-Q4_K_XL. `--models-dir` puts them elsewhere for that run. |
| `glm-<size>.profile` | The expert profile: how often each expert was used, written when the engine ends cleanly and loaded at the next start, so the VRAM cache starts full. |

`~/.config/strata-glm/settings.json` says where the data folder is and which Strata folders you ran, so a new copy of
Strata (an update unzipped elsewhere) finds the model files. Setup moves model files it finds in another GLM Strata
folder on the same drive into the data folder; files on another drive are used where they are. Folders and configs of
the earlier Qwen version of Strata are left alone.

What setup and the server print goes to the terminal (or the service's journal). Only the engine's own output goes to
the log file.

## The build

Setup builds the engine with CMake and Ninja (both from pip, in `.venv`). The commands it runs, with your values (it
also passes `-DCMAKE_MAKE_PROGRAM=<its ninja>`, and `CMAKE_C_COMPILER`, `CMAKE_CXX_COMPILER` and
`CMAKE_CUDA_HOST_COMPILER` when `CC`, `CXX` or `CUDAHOSTCXX` are set):

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release \
    -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=86 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc -DCUDAToolkit_ROOT=/usr/local/cuda-13.0 \
    -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
cmake --build build --target strata-glm strata-glm-ref -j <half the CPU threads>
```

- **`CMAKE_CUDA_ARCHITECTURES`** comes from the cards: nvidia-smi's compute capability without the dot (8.6 is `86`),
  each one once, joined with `;` (`86;89` for an RTX 30 and an RTX 40 card). An RTX 30 PC builds for `86` only. A
  later build keeps every architecture the last one had.
- **The toolkit:** an installed CUDA toolkit of 12.0 or newer is used as it is; an RTX 50 card (sm_120) needs 13.0 or
  newer. When none is there, setup asks and, on Ubuntu 22.04 or 24.04, installs `build-essential` and CUDA 13.0.
  The same steps by hand, if you prefer to run them yourself (`2404` for 24.04, `2204` for 22.04; setup downloads
  the keyring package itself instead of with wget):

  ```bash
  sudo apt-get install -y build-essential
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get install -y cuda-toolkit-13-0
  ```

  On other distributions install the toolkit from your packages or https://developer.nvidia.com/cuda-downloads (Arch:
  `pacman -S base-devel cuda`). Setup finds `nvcc` on `PATH`, in `/usr/local/cuda*` and in `/opt/cuda*`.
- **The libraries:** the engine loads the CUDA runtime and cuBLAS of the toolkit it was built with. Setup writes that
  toolkit's library folder into the run config (`lib_dirs`), and the server puts it first on the engine's
  `LD_LIBRARY_PATH`.
- **CUDA 13.2.0 and 13.2.1** compiled wrong code for RTX 50 cards (sm_120) in two reports on the upstream project. For
  such a card setup takes an older 13.x toolkit when one is installed, and warns otherwise. 13.2.2 is fine.
- **A failed build** is tried once more (CUDA's `ptxas` sometimes fails once on a file it just wrote).
- **Rebuilding:** setup compiles again when the engine's source changed (a hash of `CMakeLists.txt`, `src/`,
  `include/`, `third_party/` and the llama.cpp commit, kept in `build/BUILD.json`), when a card has an architecture the
  build lacks, or with `--build`. A rebuild after a `git pull` compiles only what changed. Setup says the first build
  takes 10-20 minutes; that is its estimate, not measured on a GLM build.
- **No github.com?** Put llama.cpp's tree at the commit `LLAMA_CPP_COMMIT` in `setup.py` names into
  `third_party/llama.cpp` yourself (it needs `ggml/CMakeLists.txt` and `gguf-py/`), and setup uses it.

## The run config

`strata-glm-<size>.json` is what the server starts with: `serve/server.py --engine strata --config <file>`. It is plain
JSON. Change it with a text editor, or the few keys the web app's Settings view lists, and restart the model: the
server reads it only when it starts. A bad value in most keys stops the start with a line that names the key.

Setup writes the keys in the first table. When setup runs again for the same size, it rewrites those and keeps every
other key, and every engine option in `"args"` it does not write itself (it writes `-m`, `--ctx`, `--park-mib` and
`--expert-profile`).

A config as setup writes it (the paths are an example):

```json
{
 "exe": "/home/me/Strata/build/strata-glm",
 "args": ["-m", "/home/me/Strata-GLM-data/models/glm-ud-iq3_xxs/<name>-00001-of-0000N.gguf",
          "--ctx", "131072", "--park-mib", "8192",
          "--expert-profile", "/home/me/Strata-GLM-data/glm-ud-iq3_xxs.profile"],
 "cwd": "/home/me/Strata",
 "log": "/home/me/Strata/strata-glm-ud-iq3_xxs.log",
 "lib_dirs": ["/usr/local/cuda-13.0/lib64"],
 "model_name": "glm-5.3-flash",
 "port": 8101
}
```

### The keys setup writes

| Key | What it does | When it is missing |
| --- | --- | --- |
| `exe` | The engine: `build/strata-glm`. A relative path is taken from `cwd`. | Required. |
| `args` | The engine's options, a list of strings ([below](#engine-options-args)). Must have `-m <first shard>`. | Required. |
| `cwd` | The engine's working folder. Relative paths in `exe`, `-m` and `chat_template` are taken from here. | The folder the server was started from. |
| `log` | The engine's log file, appended to. Give a full path. | No log (and the server cannot follow the warm-up). |
| `lib_dirs` | Folders put first on the engine's `LD_LIBRARY_PATH`: the CUDA toolkit's libraries the engine was built with (from `build/BUILD.json`). | The system's library path. |
| `model_name` | The name `/v1/models` lists. | The GGUF's `general.name`, else `glm-5.3-flash`. |
| `port` | The port `./setup.sh` starts the server on. The server itself takes `--port` (default 8101); `run-glm-<size>.sh` has the port written into it. | 8101. |
| `gpu` | The GPUs, as nvidia-smi numbers them: `2`, or `[0, 1, 2]` (also `"0,1,2"`) for a layer split, in that order ([Several GPUs](#several-gpus-and-their-order)). Setup writes it when you name cards (`--gpus`, `--gpu`), or when it must leave out a card it cannot use. | Every GPU CUDA sees, in CUDA's own order. |
| `host` | The address to listen on: `127.0.0.1` (this PC only), `0.0.0.0` (every network), or one address of this PC. Setup writes it with `--host`. | `127.0.0.1`. |
| `api_key` | The key clients must send (`Authorization: Bearer <key>` or `x-api-key`): a string, `"k1,k2"`, or a list. Setup writes it with `--api-key`. An empty one is warned about and means no key. | No key. |
| `open_browser` | `false`: do not open the browser when the model is ready, even when the run script passes `--open`. Setup writes it with `--browser` / `--no-browser`; it is kept when setup runs again. | The browser opens. |

### Engine options (`args`)

`strata-glm --serve` takes these and stops (exit code 2, with its usage line in the log) at any other option. The
server adds `--serve`, `--split` from the config's `split` key, and `--eos-ids` with GLM's end-of-turn tokens when
`args` names none.

| Option | What it does | Default |
| --- | --- | --- |
| `-m FILE` | The model's first GGUF shard. The engine finds the others beside it. | Required. |
| `--ctx N` | The longest conversation in tokens ([Choosing the context](#choosing-the-context)). The model's trained context is 1,048,576, and setup takes no more; the engine itself does not check it. | 32768 |
| `--park-mib N` | RAM in MiB for parked conversations and checkpoints. 0 = none (every switch of conversation reads the new one from the start). | 8192 |
| `--checkpoint-min N` | No checkpoint closer than N tokens to the start of a conversation. | 256 |
| `--expert-profile FILE` | The expert use scores: loaded at start (the VRAM cache starts full), written when the engine ends cleanly. | None |
| `--vram-reserve GB` | VRAM left free on each GPU after the expert cache. | 1 |
| `--max-swaps N` | Experts copied into the VRAM cache per token, all layers together. | 8 |
| `--no-expert-cache` | No VRAM expert cache. | Off |
| `--prefill-chunk N` | Prompt tokens read at once. 0 = one at a time. | 256 |
| `--threads N` (`-t N`) | CPU threads for the experts computed on the CPU. The warm-up uses at most 16 of them (8 when this is 0). | 0 = all hardware threads |
| `--devices 0,1,2` | CUDA device numbers, in layer order. Use `gpu` instead: it sets which cards CUDA shows the engine and in which order, and these numbers would count among those. | Every visible device |
| `--split w,w,w` | Each device's share of the layers. The config's `split` key gives the same. | Each card's free VRAM at start |
| `--no-warm` | Skip reading the experts into the page cache before the engine says it is ready. The first prompts then read them from the disk. | Off |
| `--eos-ids a,b,c` | The tokens that end a turn. The server adds them. | Set by the server |

What each option does inside the engine: [HOW_IT_WORKS.md](HOW_IT_WORKS.md). `strata-glm` also has a check mode
(`--tokens-file`, `--out`, `--check-cpu`, `--prefill`, `--max-tokens`) used by the GLM checks in `tools/`; those
options are not for the run config. The CPU reference `strata-glm-ref` takes `-m`, `--ctx`, `--eos-ids`, `--threads`,
`--park-mib` and `--checkpoint-min` (and ignores `--no-warm`); it runs the whole model on the CPU, one token at a
time, and is meant for the checks.

`--reasoning-effort` placed in `args` by mistake is taken out of them and used as `reasoning_effort` (below), with a
line that says so.

### Keys you can add

None of these is written by setup, and setup keeps each one when it runs again. The API side of most of them is
explained in [DETAILS.md](DETAILS.md#settings). Keys the server does not know are ignored.

| Key | What it does | Default |
| --- | --- | --- |
| `split` | Each card's share of the layers, in the order of `gpu`: `[24, 12, 10]` or `"24,12,10"`. One positive number per card. Passed as `--split`. | Each card's free VRAM |
| `env` | Extra environment variables for the engine: `{"NAME": "value"}`. | None |
| `chat_template` | A Jinja file to use instead of the template in the model's GGUF (also `--chat-template`). | The GGUF's, else `serve/chat_template.jinja` |
| `sampling` | Defaults for requests that set none: `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty` (or `repeat_penalty`), `penalty_last_n` (or `repeat_last_n`), `seed`. A value out of range (NaN, a temperature below 0, a `top_p` outside 0-1, a repetition penalty of 0 or less) stops the start and names the key. A request's own values win. | temperature 1.0, top_p 0.95 |
| `reasoning_effort` | Thinking for requests that do not say: `none` turns it off; `low`, `medium` and `high` turn it on (GLM thinks or does not, it has no levels). | On |
| `reasoning_budget_tokens` | The most tokens a request may think, then the model is told to wrap up. 0 = no cap. A request can set its own. | No cap |
| `reasoning_loop_recovery` | `"stop"` (or `true`): end a reply whose thinking repeats itself. | `false` |
| `reasoning_close_retry` | `true`: a reply that ends still inside its thinking, with no answer and no call, is continued once with the thinking closed. | `false` |
| `anthropic_thinking` | `"model"`: Anthropic requests think as the model does. `"on_request"`: only when they ask. | `"model"` |
| `tool_call_recovery` | `true`: a declared tool's call written as a JSON object (`{"name": ..., "arguments": {...}}`) inside `<tool_call>` also counts as the call. | `false` |
| `repeat_stop_tokens` | End a reply that repeats the same token this many times. 0 = off. | 256 |
| `fit_max_tokens` | `true`: shorten a `max_tokens` that does not fit the context instead of answering 400 (also `--fit-max-tokens`). | `false` |
| `aliases` | Other model names the server lists and answers to: a list, or one comma-separated string. | None |
| `lazy_load` | `true`: start without loading the model; the first request loads it (also `--lazy`). | `false` |
| `idle_unload_s` | Unload the model after this many seconds without requests (also `--idle-unload`). | Never |
| `min_free_vram_mib` | Load an unloaded model only when this much VRAM is free, else answer 503 (also `--min-free-vram-mib`). | Always load |
| `before_load` | A command (a string or a list) run before an unloaded model loads again (also `--before-load`). | None |
| `before_load_timeout_s`, `before_load_ignore_failure`, `before_load_strict` | How long `before_load` may run (default 120 s), and what a failure of it does. | |
| `engine_silence_s` | End a request when the engine prints nothing for this many seconds. 0 = wait. | 300 |
| `allowed_hosts` | Extra host names the server answers to when it has no API key (DNS rebinding protection): a name, a list, `".example.com"` for a domain, or `"*"` for any. With an API key the Host check is off. | None |
| `cors_origins` | Web pages (origins) that may call `/v1/*` from a browser. `"*"` = any. | None |
| `trusted_origins` | Origins that count as Strata's own page, for the web app behind a reverse proxy or tunnel. | None |
| `api_monitor` | `true`: keep the last 100 requests' prompts and answers in memory for `/api-monitor` (also `--api-monitor`). | `false` |
| `mcp_servers` (or `mcpServers`) | MCP servers whose tools the web app's chat can use, in Claude Desktop's format (also `--mcp-config FILE`). Their URLs are shown and logged without user name, password, query or fragment. | None |
| `mcp_for_api` | `true`: API clients may use those MCP tools too (their requests send `"strata_mcp": true`). Without it only Strata's own page can. | `false` |
| `mcp` | Settings for those MCP servers: `timeout_s` (a tool call), `start_timeout_s` (a server's start), `max_result_chars` (a tool result's text), `max_rounds` (tool rounds in one reply). | 60, 120, 20000, 8 |
| `codex_thread_titles`, `codex_compaction_cache` | Two opt-in helpers for Codex CLI on `/v1/responses`. | `false` |
| `prevent_sleep` | Windows only: on Linux the server says so and ignores it. | `false` |

The Settings view on the web page (GET / POST `/config`) changes only these: the four `sampling` defaults
(`temperature`, `top_p`, `top_k`, `min_p`), `reasoning_budget_tokens`, `fit_max_tokens`, `anthropic_thinking`,
`aliases`, `idle_unload_s`, `lazy_load`, `engine_silence_s`, `api_monitor`, `open_browser`, and the engine's
`--vram-reserve` and `--park-mib`. Everything else (the network and security keys, the MCP servers, `before_load`) is
changed in the file only. A change is used from the next start.

## Choosing the context

The context (`--ctx`) is the longest conversation the engine takes: the chat, files and tool output together. Each
token of it takes 11 KB of VRAM in the attention (MLA) cache: a 512-wide f16 latent in each of the 11 MLA layers. The
engine reserves it at start, and the expert cache gets the VRAM that is left. Calculated:

| Context | Tokens | Attention cache in VRAM |
| --- | --- | --- |
| 32K | 32,768 | 0.4 GB |
| 128K | 131,072 | 1.5 GB |
| 256K | 262,144 | 3.0 GB |
| 512K | 524,288 | 5.9 GB |
| 1M | 1,048,576 | 11.8 GB (the model's whole context) |

The indexer's pooled keys add under 0.5 KB per token, and the KDA layers keep a fixed-size state however long the text
is ([HOW_IT_WORKS.md](HOW_IT_WORKS.md)). Each MLA layer's cache sits on the card that holds the layer. Setup warns
when the attention cache would take more than half of the chosen cards' VRAM; it keeps your choice. How much a smaller
expert cache slows the answers is not measured yet.

A long context also costs RAM when a conversation is parked (several clients taking turns,
[below](#parking-and-several-clients)): about 0.15 GB for the KDA layers' state, plus about 11.6 KB per token (the
11 KB latent and, at most, 0.35 KB of the indexer's pooled keys). Calculated: about 1.7 GB for a 128K-token
conversation, about 6.2 GB for 512K, about 12.3 GB for 1M. The KDA figure assumes a short-convolution width of 4,
which is read from the model's file; it is not checked on the real GGUF yet.

To change the context of an installed model: `./setup.sh --setup --model <size> --context <tokens>`, or edit `--ctx` in
the config's `args` and restart.

### Parking and several clients

The engine runs one request at a time; the others wait in the server in the order they came. When a request
continues a different conversation than the last one, the engine copies the current state to RAM (parks it) and later
copies it back, so it reads only the new tokens. `--park-mib` caps that RAM. The least recently used conversation goes
first; one whose state does not fit at all is not kept, and its next request reads it again from the start. The
engine's log says so (`conversations: ... more than --park-mib; not kept`).

With the default 8192 MiB, about five 128K-token conversations fit, or one of 512K, and a 1M-token conversation does
not fit (calculated from the sizes above; checkpoints share the same RAM). Raise `--park-mib` when several clients
work with long conversations, but every MiB it takes is RAM that no longer holds the model's files in the page cache
beside the experts. Which value works best on a given PC is not measured yet.

## Several GPUs and their order

By default setup uses every GPU that can run the engine, and the engine splits the model's layers over them. Each card
holds its layers, their part of the attention cache and an expert cache for its own layers. The output head goes on
the last card.

- **Which cards:** `--gpus 0,2` or `--gpus all` (saved), `--gpu 1` (one card). The numbers are nvidia-smi's: run
  `nvidia-smi -L` to see them. In the config this is `"gpu"`.
- **Their order:** with `"gpu"` set, the server sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES` to those
  numbers in that order, so the layers go over the cards in the order you list them. Without `"gpu"`, CUDA's own
  order applies (fastest card first). To fix the order, name the cards: `./setup.sh --gpus 0,1,2` (or `--gpus all`,
  which writes every usable card in nvidia-smi's order).
- **Their shares:** by default each card gets layers in proportion to its free VRAM when the engine starts (the
  layers' weights and attention cache counted). `"split": [24, 12, 10]` in the config sets the shares yourself, one
  per card in the order of `"gpu"`.
- **What each card got:** the engine's log, and the server window at start, have one line per card
  (`GPU 0: layers 0-21, ... GB weights and buffers, ... GB expert cache`, with `+ output head` on the last) and one
  for the expert cache. Those GPU numbers count the cards CUDA shows the engine, from 0 in your order.

Which order and split are fastest is not measured yet.

## On napster-ai

The owner's PC: Linux, Xeon E5-2699 v4 (AVX2), 192 GB of DDR4, an RTX 3090 Ti (24 GB), an RTX 3080 Ti (12 GB) and an
RTX 3080 (10 GB), all compute capability 8.6, and a SATA SSD. What setup does there, from its code and its tests (it
has not been run there yet):

- **Size:** the 3-bit `UD-IQ3_XXS` is recommended (about 79 GiB of RAM left beside its experts, if `/proc/meminfo`
  says 188 GiB as the tests assume). The 4-bit is tight (about 37 GiB left) and gets `--park-mib 4096`.
- **Build:** all three cards can run the engine, so setup uses all three and builds for `86` only.
- **Context:** the three cards have 46 GB of VRAM in all, so setup recommends 128K. 512K takes 5.9 GB and 1M 11.8 GB
  of it (calculated); neither is more than half, so setup does not warn.
- **Port:** 8101. Open WebUI (3000), Portainer (8000) and the UPS dashboard (8080) keep theirs.
- **Clients:** four agents, a chat and a Discord bot take turns on the one engine. With contexts of 512K-1M tokens,
  the default `--park-mib 8192` keeps at most one 512K conversation and no 1M one
  ([Parking](#parking-and-several-clients)).
- **Other devices** (over Tailscale, 100.126.222.90): set `--host 0.0.0.0 --api-key <secret>`. `--host
  100.126.222.90` instead listens on the Tailscale address only; the apps on napster-ai itself then use that address
  too, since 127.0.0.1 no longer answers. A host name such as a Tailscale MagicDNS name works with an API key; without
  a key it must be added to `allowed_hosts`. See [SECURITY.md](../SECURITY.md).

A command for a first install there, without questions, with the cards named in nvidia-smi's order:

```bash
./setup.sh --yes --model UD-IQ3_XXS --context 524288 --gpus all --no-browser --no-start
```

## Running it as a service

The start script is enough for a service: it changes to the Strata folder and runs the server. This systemd unit is
an example; it has not been run on a real machine yet. Replace `<user>` and the paths with yours, and the script name
with the one setup printed.

```ini
# /etc/systemd/system/strata.service (an example)
[Unit]
Description=Strata (GLM-5.3-Flash) on port 8101
After=network-online.target
Wants=network-online.target

[Service]
User=<user>
WorkingDirectory=/home/<user>/Strata
ExecStart=/home/<user>/Strata/run-glm-ud-iq3_xxs.sh
Restart=on-failure
RestartSec=30
# SIGTERM to the server only: the server sends the engine STOP and QUIT, and the engine writes its expert profile.
# The default (control-group) also sends SIGTERM to the engine, which then ends at once without writing it.
KillMode=mixed
# the server gives the engine 20 s after QUIT, 20 s after a terminate and 20 s after a kill
TimeoutStopSec=120

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now strata
journalctl -u strata -f                         # the server's output
tail -f ~/Strata/strata-glm-ud-iq3_xxs.log      # the engine's log
```

Before you enable it:

- Add `"open_browser": false` to the run config, or run setup with `--no-browser` (a service has no browser to open).
- To reach it from other machines, set `"host": "0.0.0.0"` **and** `"api_key"`. Without a key, anyone who can reach
  the port can use the model.
- `TimeoutStopSec=120` is an example value: how long the engine takes to stop on the real model is not measured. A
  stop that is killed loses the expert profile of that run.
- The run script has the port written into it. To change the port, run setup again with `--setup --port N`, or edit
  the script.
- The run script does not compile anything. After an update ([below](#updating)), restart the service:
  `./update.sh` compiles the engine, the service does not.

## Updating

```bash
./update.sh
```

In a git clone it runs `git pull --ff-only`, then `./setup.sh --update`: the Python packages, and the engine compiled
again when its source changed. It asks nothing (unless the C++ compiler or the CUDA toolkit has gone missing: then
it asks to install them; `./update.sh --yes` takes the answer y without asking), does not touch the model files and
does not start the model. Options go on to `setup.py` (`./update.sh --build` compiles the engine even when it is up
to date). A running server keeps the engine it started; restart the model to use the new one. If `git pull` fails
(files you changed can stop it), nothing is updated and it says so. If the new engine does not compile, the one you
had stays and still starts.

In a copy that was not made with git, `./update.sh` cannot fetch new files. Download the new zip
(https://github.com/Rarebigshot17/Strata/archive/refs/heads/glm53-flash.zip), unzip it anywhere and run `./setup.sh` in
it. It finds the model files in the data folder and sets itself up like your last install (size, context, parking,
host, key, port, GPUs) without questions; nothing big is downloaded again. The engine is compiled once in the new
folder.

## Docker

The image compiles the engine during `docker build`, so a container start only downloads the model (on the first
start) and serves. The host needs an NVIDIA driver 580 or newer and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
The image is `nvidia/cuda:13.0.0-devel-ubuntu24.04` with Strata in `/opt/strata`.

```bash
docker build -t strata .
docker build -t strata --build-arg CUDA_ARCHITECTURES=86 .      # RTX 30 cards only: a shorter build
```

`CUDA_ARCHITECTURES` defaults to `75;80;86;89;120` (RTX 20, A-series, RTX 30, RTX 40, RTX 50). A card outside the set
is compiled for at its first start. The CPU code is built for any AVX2 CPU (`STRATA_PORTABLE=ON`); on a host without
AVX2 the entrypoint warns that the engine may stop with "illegal instruction".

```bash
docker run -d --name strata --gpus all \
  -p 127.0.0.1:8101:8101 \
  --ulimit memlock=-1 \
  -v /srv/strata-data:/data \
  -e MODEL=UD-IQ3_XXS -e CONTEXT=131072 \
  strata
```

- **`/data`** holds the model files (120-170 GB) in `/data/models`, the expert profile and the run config
  (`/data/config/strata-glm-<size>.json`, linked into `/opt/strata`). The engine is part of the image.
- **Port 8101.** The server listens on `0.0.0.0` inside the container and has no API key unless you set one.
  `-p 127.0.0.1:8101:8101` publishes it to this machine only; to publish it to the network (`-p 8101:8101`, which
  Docker's rules let past ufw), set `-e API_KEY=<secret>` too. Without a key the entrypoint warns at every start.
  `API_KEY` is passed on at every start and saved in the config, so a key added later counts without `REINSTALL`.
- **The setup choices** are environment variables: `MODEL` (`UD-IQ3_XXS`, the default, or `IQ4_XS`), `CONTEXT`
  (tokens; empty: setup's pick for the VRAM), `PARK_MIB`, `GPUS` (`"0,2"` or `"all"`) or `GPU` (one card), `HOST`
  (default `0.0.0.0`), `PORT` (default 8101), `API_KEY`, `GGUF_DIR` (a mounted folder of shards you already have:
  `-v /models/glm53:/gguf:ro -e GGUF_DIR=/gguf`), `CONFIG` (start with this config file instead).
- **The first start** runs setup with these choices, `--no-browser`, `--no-start` and `--yes`, and keeps the config on
  the volume. Later starts go straight to the server. `-e REINSTALL=1` runs setup again, to change the settings of a
  size already set up. Switching `MODEL` to a size already on the volume needs no setup pass.
- **Changing the config:** edit `/data/config/strata-glm-<size>.json` on the volume and restart the container, or use
  the web app's Settings view: it writes that file through the link, and keeps its `.bak` and the shared Chat
  settings (`strata-glm-<size>.shared-settings.json`) beside it on the volume.
- **Health:** the image has a `HEALTHCHECK` on `/health` with a start period of 30 minutes, since the port opens only
  once the model is loaded (and the first start downloads it). `docker logs strata` shows the server's output; the
  engine's log is inside the container: `docker exec strata tail -f /opt/strata/strata-glm-ud-iq3_xxs.log`.
- **RAM:** setup reads the host's RAM from `/proc/meminfo`, not a container memory limit. Do not set a limit below what
  the size needs.
- **Stopping:** the server is PID 1 (`STRATA_EXECV=1`) and stops the engine on SIGTERM. Give it time:
  `docker stop -t 90 strata` (an example value; the stop time is not measured).

## Building and starting by hand

Without setup, on any Linux with the CUDA toolkit, a C++ compiler and Python 3.10+:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# CMake 3.24 or newer: the venv has one (.venv/bin/cmake). Without STRATA_GGML_DIR, CMake clones llama.cpp at the
# pinned commit with git; -DSTRATA_GGML_DIR=<a llama.cpp checkout at that commit> uses one you have.
.venv/bin/cmake -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=86
.venv/bin/cmake --build build -j"$(nproc)" --target strata-glm strata-glm-ref
```

`nvidia-smi --query-gpu=compute_cap --format=csv,noheader` gives the architectures (8.6 is `86`). Without
`CMAKE_CUDA_ARCHITECTURES` CMake builds for the cards in this PC (`native`); below 75 it stops. `strata-glm-ref` (the
CPU reference) is built even without `STRATA_ENABLE_CUDA`. `STRATA_PORTABLE=ON` builds the CPU code for any AVX2 CPU
instead of this one. With `STRATA_BUILD_TESTS=ON` (the default) the engine's unit tests are built too; run them with
`ctest --test-dir build`.

Download a size's shards (the file names are listed in the repository):

```bash
.venv/bin/pip install -U huggingface_hub
.venv/bin/hf download unsloth/GLM-5.3-Flash-GGUF --include "*UD-IQ3_XXS*" --local-dir ~/models/glm53
```

Write a run config, `strata-glm-hand.json` (full paths; see [The run config](#the-run-config)):

```json
{
 "exe": "/home/me/Strata/build/strata-glm",
 "args": ["-m", "/home/me/models/glm53/<the first shard>.gguf", "--ctx", "131072",
          "--expert-profile", "/home/me/models/glm53/glm.profile"],
 "cwd": "/home/me/Strata",
 "log": "/home/me/Strata/strata-glm-hand.log",
 "lib_dirs": ["/usr/local/cuda-13.0/lib64"]
}
```

And start the server from the Strata folder:

```bash
.venv/bin/python serve/server.py --engine strata --config strata-glm-hand.json --port 8101
```

The server's own options (`.venv/bin/python serve/server.py --help`): `--engine` (`strata`, or `mock`: a scripted
engine and no model, for trying clients, with `--script` for its answers; `mock` is the default, so always pass
`--engine strata` for the model), `--config`, `--host`, `--port` (default 8101), `--api-key`, `--gpu`,
`--chat-template`, `--open`, `--fit-max-tokens`, `--mcp-config`, `--lazy`, `--api-monitor`, `--idle-unload`,
`--min-free-vram-mib`, `--before-load`. Most have a config key too ([The run config](#the-run-config)). What the API
offers: [DETAILS.md](DETAILS.md#the-api). When something does not work: [TROUBLESHOOTING.md](TROUBLESHOOTING.md).
