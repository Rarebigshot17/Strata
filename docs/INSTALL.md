# Installing Strata

Everything about installing, starting, updating and configuring Strata: GLM-5.3-Flash on Linux with NVIDIA GPUs and
system RAM. The short version is in the [README](../README.md#install). An AI coding assistant can do all of this for
you with [AI_SETUP.md](AI_SETUP.md).

> **Nothing has run on a GPU or on the real model yet.** There are no measured speeds, load times or memory figures.
> The sizes on this page are calculated from the model's shapes. What has been checked: the CPU reference against
> llama.cpp on small random GLM models (`tools/glm53_selftest.sh`), conversation parking
> (`tools/glm53_park_check.py`) and the server end to end on the CPU engine (`tools/glm53_server_check.sh`). The steps
> still to run on a real machine are in [GLM53_PORT.md](GLM53_PORT.md).

> **On this page:** [What you need](#what-you-need) · [Install with ./setup.sh](#install-with-setupsh) ·
> [Setup's options](#setups-options) · [Where files go](#where-files-go) · [The build](#the-build) ·
> [The run config](#the-run-config) · [Choosing the context](#choosing-the-context) ·
> [Several GPUs](#several-gpus-and-their-order) · [As a service](#running-it-as-a-service) ·
> [Updating](#updating) · [Docker](#docker) · [By hand](#building-and-starting-by-hand)

## What you need

| | |
| --- | --- |
| **OS** | Linux on x86-64. Setup stops on any other system. On Ubuntu 22.04 and 24.04 setup can install the CUDA toolkit itself; on other distributions you install it. |
| **GPU** | One or more NVIDIA cards of compute capability 7.5 or newer (RTX 20, 30, 40, 50 series). Setup leaves out a card it cannot use and says why. It has no minimum VRAM: how much the layers besides the experts take is not measured yet. |
| **Driver** | NVIDIA 580 or newer. Setup stops on an older one (Ubuntu: `sudo ubuntu-drivers install`, then restart). |
| **RAM** | Every routed expert stays in RAM: about 117 GB for the 3-bit size, about 162 GB for the 4-bit (see the table below). |
| **CPU** | AVX2. Setup warns on a CPU without it but goes on: the experts the GPUs do not hold are computed on the CPU. |
| **Disk** | About 120 GB (3-bit) or 170 GB (4-bit) free for the model files. Setup checks this before it downloads. The CUDA toolkit, when setup installs it, takes more (setup says 8-10 GB with the compiler). |
| **Python** | 3.10 or newer with `venv`. `./setup.sh` installs it with apt, dnf or pacman when it is missing (asks for your sudo password). |
| **Network** | huggingface.co (the model; or a mirror, `HF_ENDPOINT`), github.com (llama.cpp's source at a pinned commit), PyPI (the Python packages), and developer.download.nvidia.com when setup installs CUDA. |

The two sizes are Unsloth's GGUF files from [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF).
Setup lists the repository's files when it runs and takes the shards of the size you chose.

| Size | `--model` | Download | Experts in RAM | Setup calls it a good fit from | Tight (warned) from |
| --- | --- | --- | --- | --- | --- |
| 3-bit | `UD-IQ3_XXS` | ~120 GB | ~117 GB | 157 GB of RAM | 129 GB |
| 4-bit | `IQ4_XS` (`UD-Q4_K_XL` when the repository has no IQ4_XS) | ~170 GB | ~162 GB | 202 GB | 174 GB |

These sizes are calculated, not measured. A good fit leaves 40 GB or more beside the experts, for Linux, the server and
parked conversations. Below 12 GB left, the OS would read experts from the disk, so setup stops unless you insist
(`--model NAME --yes`). Setup reads the RAM from `/proc/meminfo`, the number `free -g` prints: a PC with 192 GB of
memory shows about 188 there. On such a PC the 3-bit size is recommended (~71 GB left beside its experts) and the 4-bit
is tight (~26 GB left; parked conversations then get 4096 MiB instead of 8192).

The disk matters most at the first start after a reboot: the engine reads every expert from the disk before it says it
is ready. 117 GB at 550 MB/s (the top speed of a SATA SSD) is about 3.5 minutes. That is calculated, not measured.

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
   when they are missing (it asks), and builds `strata-glm` and `strata-glm-ref` into `build/`. See [The build](#the-build).
5. **Downloads the model's GGUF shards**. A download resumes where it stopped. Each shard is checked against the size
   and SHA-256 the repository lists; a wrong one is deleted, and the next run downloads it again.
6. **Writes the run config and the start script** (`strata-glm-<size>.json`, `run-glm-<size>.sh`) and starts the
   model.

At the end it prints the addresses:

```
  Chat page:        http://127.0.0.1:8101/
  API (OpenAI):     http://127.0.0.1:8101/v1   (model name: anything)
  API (Anthropic):  http://127.0.0.1:8101/v1/messages
  Next time:        just run ./setup.sh (or ./run-glm-ud-iq3_xxs.sh) - it starts right away
```

### The questions

Press Enter to take the answer in brackets. `--yes` takes it for every question.

| Question | Choices | Default |
| --- | --- | --- |
| **Which size?** | 1) `UD-IQ3_XXS` 3-bit, 2) `IQ4_XS` 4-bit. Each line says how much RAM it leaves beside its experts. | The biggest size that fits well; the 3-bit when none does. |
| **Context?** | 1) 32K, 2) 128K, 3) 256K, 4) 512K, 5) 1M tokens. Each line says the VRAM its attention cache takes ([Choosing the context](#choosing-the-context)). | 128K when the chosen GPUs have 16 GB of VRAM or more in all, else 32K. |
| **Install them now?** | Asked only when the C++ compiler or the CUDA toolkit is missing. `y` installs them with `sudo apt-get` (Ubuntu 22.04 / 24.04). | `y` |
| **Install it anyway?** / **Go on anyway?** | Asked only when the size you chose does not fit the RAM. | `n` |
| **Which one?** | Asked by a plain `./setup.sh` when two sizes are installed: start one of them, or set up the other size. | `1`, the one used last |

Setup recommends, it never forces. A bigger choice than it recommends (the 4-bit model on a PC it nearly fills, a long
context on little VRAM) is kept, with a warning that says what it risks. With `--yes`, a size you name with `--model` is
your consent to a size that does not fit; `--yes` alone keeps the stop.

### Later starts

`./setup.sh` with no options starts the installed model right away. Before it starts, it compiles the engine again when
its source changed (a `git pull`) or the PC has a GPU the build has no code for; if that compile fails, it starts the
engine you had. `./run-glm-<size>.sh` starts the server directly, without those checks.

The server opens its port only when the engine is ready. Before that, the engine maps the model files and reads every
routed expert once, so they are in RAM rather than on the disk (its log says `warming the experts: ... GB`). How long
this takes on the real model is not measured yet. The server gives up when the engine has not said it is ready after 900
seconds; on a slow disk, set `STRATA_ENGINE_READY_S` (seconds, 0 = no limit) before you start it.

Ctrl+C, or closing the terminal, stops the model. On Ctrl+C the server stops the engine, which writes its expert
profile as it exits.

### Setup's options

`python3 setup.py --help` (or `./setup.sh --help`) prints them. This is that list, as setup.py gives it:

```
  --model {UD-IQ3_XXS,IQ4_XS}
                        UD-IQ3_XXS = 3-bit (~120 GB download, ~117 GB of experts in RAM), IQ4_XS = 4-bit (~170 GB,
                        ~162 GB; UD-Q4_K_XL when the repository has no IQ4_XS). Default: the one this PC's RAM fits
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

- **`--check`** prints step 1 and, for each size, whether it fits this PC's RAM and disk. It installs, moves and writes
  nothing, and needs no `.venv`: `python3 setup.py --check` works on a fresh clone. Its exit code is 1 when neither size
  fits.
- **On an installed model**, a plain start keeps these for the next starts too: `--gpus`, `--host`, `--api-key`,
  `--park-mib`, `--browser` / `--no-browser`. These are for that start only: `--port`, `--gpu`.
- **`--context` needs `--setup`** on an installed model: `./setup.sh --setup --model UD-IQ3_XXS --context 262144`.
  Without `--setup` (or `--model`, or `--no-start`) setup starts the installed model, and `--context` has no effect.
- **`--setup --yes` takes the recommended answers again.** Name the size and the context you want
  (`--model ... --context ...`), or setup picks its own. Keys you added to the run config by hand are kept
  ([The run config](#the-run-config)), but `host`, `api_key` and `gpu` are written from this run's options only:
  give `--host`, `--api-key` and `--gpus` again, or the model listens on 127.0.0.1 without a key on every GPU.
- **`--gguf-dir`** takes shards you downloaded yourself (for example with `hf download`). Setup reads their headers to
  check that each is whole, and downloads nothing.
- **`--gpu 0,2`** means `--gpus 0,2`.

### Environment variables

| Variable | What it does |
| --- | --- |
| `HF_ENDPOINT` | A Hugging Face mirror (e.g. `https://hf-mirror.com`) for the listing and the downloads. The same size and SHA-256 checks apply. |
| `STRATA_NVCC` | The `nvcc` to build with. When set, it is the only one considered. Otherwise setup takes the newest toolkit on `PATH`, in `$CUDA_PATH` / `$CUDA_HOME`, `/usr/local/cuda*` or `/opt/cuda*`. |
| `CC`, `CXX`, `CUDAHOSTCXX` | The host compilers of the build. `CUDAHOSTCXX` (nvcc's host compiler) defaults to `CXX`. |
| `STRATA_UV=1` | `./setup.sh` makes `.venv` with uv. On NixOS this is the default when uv is installed. |
| `XDG_CONFIG_HOME` | Where setup keeps `strata-glm/settings.json` (default `~/.config`). |
| `STRATA_EXECV=1` | Setup replaces itself with the server instead of starting it as a child. The Docker image sets it. |
| `STRATA_ENGINE_READY_S` | Seconds the server waits for the engine to be ready (default 900, 0 = no limit). |
| `STRATA_API_KEY` | The server's API key, as `--api-key` or `"api_key"`. |
| `STRATA_ALLOWED_HOSTS` | Extra host names the server answers to, as `"allowed_hosts"`. |

The server reads a few more (timeouts, request size, log lines): [DETAILS.md](DETAILS.md#settings).

### Where files go

In the Strata folder (the one with `setup.py`):

| Path | What |
| --- | --- |
| `.venv/` | The private Python environment. |
| `third_party/llama.cpp/` | llama.cpp at the pinned commit: ggml for the build, gguf-py for the tools. |
| `build/` | The CMake build: `strata-glm`, `strata-glm-ref`, and `BUILD.json` (what they were compiled from and for). |
| `strata-glm-<size>.json` | The run config ([below](#the-run-config)). The earlier one is kept as `strata-glm-<size>.json.bak` when setup runs again or the web app's Settings view changes it. |
| `run-glm-<size>.sh` | The start script: `serve/server.py` with this config and port. |
| `strata-glm-<size>.log` | The engine's log: its own messages and the server's notes on each start. New starts are appended. |
| `strata-glm-<size>.shared-settings.json` | The web app's Chat settings when you share them with other apps. Written by the server. |

`<size>` is the size in lower case: `ud-iq3_xxs` or `iq4_xs`.

In the data folder, `Strata-GLM-data` next to the Strata folder (or where `--data-dir` put it):

| Path | What |
| --- | --- |
| `models/glm-<quant>/` | The GGUF shards, each with a `<shard>.done` mark that keeps its checked SHA-256. `<quant>` is the files' own quantization: `glm-ud-q4_k_xl` when the 4-bit size fell back to UD-Q4_K_XL. `--models-dir` puts them elsewhere for that run. |
| `glm-<size>.profile` | The expert profile: how often each expert was used, saved when the engine exits and loaded at the next start, so the VRAM cache starts full. |

`~/.config/strata-glm/settings.json` says where the data folder is and which Strata folders you ran, so a new copy of
Strata (an update unzipped elsewhere) finds the model files. Setup moves model files it finds in another GLM Strata
folder on the same drive into the data folder; files on another drive are used where they are. Folders and configs of
the earlier Qwen version of Strata are left alone.

What setup and the server print goes to the terminal. Only the engine's own output goes to the log file.

## The build

Setup builds the engine with CMake and Ninja (both from pip, in `.venv`). The command it runs, with your values:

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
  newer. When none is there, setup asks and, on Ubuntu 22.04 or 24.04, installs `build-essential` and CUDA 13.0. These
  are the commands it runs (`2404` for 24.04, `2204` for 22.04), if you prefer to run them yourself:

  ```bash
  sudo apt-get install -y build-essential
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
  sudo dpkg -i cuda-keyring_1.1-1_all.deb
  sudo apt-get update
  sudo apt-get install -y cuda-toolkit-13-0
  ```

  On other distributions install the toolkit from your packages or https://developer.nvidia.com/cuda-downloads (Arch:
  `pacman -S base-devel cuda`). Setup finds `nvcc` on `PATH`, in `/usr/local/cuda*` and in `/opt/cuda*`.
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
server reads it when it starts. Setup writes the keys in the first table. When setup runs again for the same size, it
rewrites those and keeps every other key, and every engine option in `"args"` it does not write itself.

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
| `cwd` | The engine's working folder. Relative paths in `exe`, `-m` and `chat_template` are taken from here. | The server's own folder. |
| `log` | The engine's log file, appended to. Give a full path: a relative one is taken from the folder the server runs in. | No log. |
| `lib_dirs` | Folders put first on the engine's `LD_LIBRARY_PATH`: the CUDA toolkit's libraries the engine was built with (from `build/BUILD.json`). | The system's library path. |
| `model_name` | The name `/v1/models` lists. Requests may name any model. | The GGUF's `general.name`, else `glm-5.3-flash`. |
| `port` | The port `./setup.sh` starts the server on. The server itself takes `--port`; `run-glm-<size>.sh` has the port written into it. | 8101. |
| `gpu` | The GPUs, as nvidia-smi numbers them: `2`, or `[0, 1, 2]` (also `"0,1,2"`) for a layer split, in that order ([Several GPUs](#several-gpus-and-their-order)). Setup writes it when you name cards, or when it must leave out a card it cannot use. | Every GPU CUDA sees. |
| `host` | The address to listen on: `127.0.0.1` (this PC only) or `0.0.0.0` (every network). Setup writes it with `--host`. | `127.0.0.1`. |
| `api_key` | The key clients must send (`Authorization: Bearer <key>` or `x-api-key`): a string, `"k1,k2"`, or a list. Setup writes it with `--api-key`. | No key. |
| `open_browser` | `false`: do not open the browser when the model is ready, even when the run script passes `--open`. Setup writes it with `--browser` / `--no-browser`. | The browser opens. |

### Engine options (`args`)

`strata-glm --serve` takes these and stops (exit code 2) at any other option. The server adds `--serve`, and
`--eos-ids` with GLM's end-of-turn tokens when `args` names none.

| Option | What it does | Default |
| --- | --- | --- |
| `-m FILE` | The model's first GGUF shard. The engine finds the others beside it. | Required. |
| `--ctx N` | The longest conversation in tokens, up to 1,048,576 ([Choosing the context](#choosing-the-context)). | 32768 |
| `--park-mib N` | RAM in MiB for parked conversations and checkpoints. 0 = none. | 8192 |
| `--checkpoint-min N` | No checkpoint closer than N tokens to the start of a prompt. | 256 |
| `--expert-profile FILE` | The expert use counts: loaded at start (the VRAM cache starts full), written at exit. | None |
| `--vram-reserve GB` | VRAM left free on each GPU after the expert cache. | 1 |
| `--max-swaps N` | Experts copied into the VRAM cache per token, all layers together. | 8 |
| `--no-expert-cache` | No VRAM expert cache. | Off |
| `--prefill-chunk N` | Prompt tokens read at once. 0 = one at a time. | 256 |
| `--threads N` (`-t N`) | CPU threads (the experts computed on the CPU). | 0 = all hardware threads |
| `--devices 0,1,2` | CUDA device numbers, in layer order. Use `gpu` instead: it sets which cards CUDA shows the engine, and this option counts among those. | Every visible device |
| `--split w,w,w` | Each device's share of the layers. The config's `split` key gives the same. | By each card's free VRAM at start |
| `--no-warm` | Skip reading the experts into RAM before the engine says it is ready. The first prompts then read them from the disk. | Off |
| `--eos-ids a,b,c` | The tokens that end a turn. The server adds them. | Set by the server |

What each option does inside the engine: [HOW_IT_WORKS.md](HOW_IT_WORKS.md). `strata-glm` also has a check mode
(`--tokens-file`, `--out`, `--check-cpu`, `--prefill`, `--max-tokens`) used by the GLM checks in `tools/`; those
options are not for the run config. The CPU reference `strata-glm-ref` takes `-m`, `--ctx`, `--eos-ids`, `--threads`,
`--park-mib`, `--checkpoint-min` (and ignores `--no-warm`); it runs the whole model on the CPU, one token at a time.

### Keys you can add

None of these is written by setup, and setup keeps each one when it runs again. Most are explained in full in
[DETAILS.md](DETAILS.md#settings).

| Key | What it does | Default |
| --- | --- | --- |
| `split` | Each card's share of the layers, in the order of `gpu`: `[24, 12, 10]` or `"24,12,10"`. Passed as `--split`. | By free VRAM |
| `env` | Extra environment variables for the engine: `{"NAME": "value"}`. | None |
| `chat_template` | A Jinja file to use instead of the template in the model's GGUF (also `--chat-template`). | The GGUF's |
| `sampling` | Defaults for requests that set none: `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`, `frequency_penalty`, `repetition_penalty` (or `repeat_penalty`), `penalty_last_n` (or `repeat_last_n`), `seed`. A bad value stops the start. | temperature 1.0, top_p 0.95 |
| `reasoning_effort` | Thinking for requests that do not say: `none` turns it off; `low`, `medium` and `high` turn it on (GLM has no levels). | On |
| `reasoning_budget_tokens` | The most tokens a request may think. 0 = no cap. A request can set its own. | No cap |
| `reasoning_loop_recovery` | `"stop"` (or `true`): end a reply whose thinking repeats itself. | `false` |
| `reasoning_close_retry` | `true`: a reply that ends inside its thinking is closed once and continued. | `false` |
| `anthropic_thinking` | `"model"`: Anthropic requests think as the model does. `"on_request"`: only when they ask. | `"model"` |
| `tool_call_recovery` | `true`: also read tool calls written in forms close to the template's. | `false` |
| `repeat_stop_tokens` | End a reply that repeats the same token this many times. 0 = off. | 256 |
| `fit_max_tokens` | `true`: shorten a `max_tokens` that does not fit the context instead of answering 400. | `false` |
| `aliases` | Other model names the server lists and answers to. | None |
| `lazy_load` | `true`: start without loading the model; the first request loads it (also `--lazy`). | `false` |
| `idle_unload_s` | Unload the model after this many seconds without requests (also `--idle-unload`). | Never |
| `min_free_vram_mib` | Load an unloaded model only when this much VRAM is free, else answer 503 (also `--min-free-vram-mib`). | Always load |
| `before_load` | A command (a string or a list) run before an unloaded model loads again. | None |
| `before_load_timeout_s`, `before_load_ignore_failure`, `before_load_strict` | How long `before_load` may run (120 s), and what a failure of it does. | |
| `engine_silence_s` | End a request when the engine prints nothing for this many seconds. 0 = wait. | 300 |
| `allowed_hosts` | Extra host names the server answers to without an API key, or `"*"` for any (DNS rebinding protection). | None |
| `cors_origins` | Web pages (origins) that may call `/v1/*` from a browser. `"*"` = any. | None |
| `trusted_origins` | Origins that count as Strata's own page, for the web app behind a reverse proxy. | None |
| `api_monitor` | `true`: keep the last 100 requests' prompts and answers in memory for `/api-monitor`. | `false` |
| `mcp_servers` (or `mcpServers`) | MCP servers whose tools the web app's chat can use, in Claude Desktop's format. | None |
| `mcp` | `timeout_s`, `start_timeout_s`, `max_result_chars`, `max_rounds` for those MCP servers. | |
| `codex_thread_titles`, `codex_compaction_cache` | Two opt-in helpers for Codex CLI on `/v1/responses`. | `false` |
| `prevent_sleep` | Windows only: the server says so and ignores it on Linux. | `false` |

`--reasoning-effort` placed in `args` by mistake is taken out of them and used as `reasoning_effort`, with a line that
says so. An unknown key elsewhere in the file is ignored.

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
is ([HOW_IT_WORKS.md](HOW_IT_WORKS.md)). The cache is spread over the cards that hold MLA layers.
Setup warns when the attention cache would take more than half of the chosen cards' VRAM; it keeps your choice. How
much a smaller expert cache slows the answers is not measured yet.

A long context also costs RAM when a conversation is parked: about 0.15 GB for the KDA state plus 11 KB per token
(calculated: about 1.6 GB for a 128K-token conversation, about 6 GB for 512K). The default `--park-mib 8192` holds a
about five conversations of 128K tokens, or one of 512K; the least recently used goes first, and the next request in
it reads it again from the start.
`--park-mib` can be raised where RAM is left beside the experts, but the RAM it takes is no longer free to keep the
model's files in memory, and the experts are read from those files.

To change the context of an installed model: `./setup.sh --setup --model <size> --context <tokens>`, or edit `--ctx` in
the config's `args` and restart.

## Several GPUs and their order

By default setup uses every GPU that can run the engine, and the engine splits the model's layers over them. Each card
holds its layers, their part of the attention cache and an expert cache for its own layers. The output head goes on
the last card.

- **Which cards:** `--gpus 0,2` or `--gpus all` (saved), `--gpu 1` (one card). The numbers are nvidia-smi's: run
  `nvidia-smi -L` to see them. In the config this is `"gpu"`.
- **Their order:** with `"gpu"` set, the server sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES` to those
  numbers in that order, so the layers go over the cards in the order you list them. Without `"gpu"`, CUDA's own order
  applies (fastest card first). To fix the order, name the cards: `./setup.sh --gpus 0,1,2`.
- **Their shares:** by default each card gets layers in proportion to its free VRAM when the engine starts. `"split":
  [24, 12, 10]` in the config sets the shares yourself, one per card in the order of `"gpu"`.
- **What each card got:** the engine's log has one line per card (`GPU 0: layers 0-21, ... GB weights and buffers,
  ... GB expert cache`) and one for the expert cache. Those GPU numbers count the cards CUDA shows the engine, from 0
  in your order.

Which order and split are fastest is not measured yet. CUDA numbers the cards the same way as nvidia-smi only with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, which is why the server sets it when the config names them.

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
# the server stops the engine on SIGTERM; the engine writes its expert profile as it exits
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

- Add `"open_browser": false` to the run config (a service has no browser to open).
- To reach it from other machines, set `"host": "0.0.0.0"` **and** `"api_key"`. Without a key, anyone who can reach
  the port can use the model.
- `TimeoutStopSec=120` is an example value: how long the engine takes to stop is not measured. A stop that is killed
  loses the expert profile of that run.
- The run script has the port written into it. To change the port, run setup again with `--setup --port N`, or edit
  the script.
- After an update ([below](#updating)), restart the service. `./update.sh` compiles the engine; the service does not.

## Updating

```bash
./update.sh
```

In a git clone it runs `git pull --ff-only`, then `./setup.sh --update`: the Python packages, and the engine compiled
again when its source changed. It asks nothing, does not touch the model files and does not start the model. Options
go on to `setup.py` (`./update.sh --build` compiles the engine even when it is up to date). A running server keeps the
engine it started; restart the model to use the new one. If `git pull` fails (files you changed can stop it), nothing
is updated and it says so.

In a copy that was not made with git, `./update.sh` cannot fetch new files. Download the new zip
(https://github.com/Rarebigshot17/Strata/archive/refs/heads/glm53-flash.zip), unzip it anywhere and run `./setup.sh` in
it. It finds the model files in the data folder and sets itself up like your last install (size, context, parking,
host, key, port, GPUs) without questions; nothing big is downloaded again. The engine is compiled once in the new
folder.

## Docker

The image compiles the engine during `docker build`, so a container start only downloads the model (on the first
start) and serves. The host needs an NVIDIA driver 580 or newer and the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker build -t strata .
docker build -t strata --build-arg CUDA_ARCHITECTURES=86 .      # RTX 30 cards only: a shorter build
```

`CUDA_ARCHITECTURES` defaults to `75;80;86;89;120` (RTX 20, A-series, RTX 30, RTX 40, RTX 50). A card outside the set
is compiled for at its first start. The CPU code is built for any AVX2 CPU; on a host without AVX2 the entrypoint warns
that the engine may stop with "illegal instruction".

```bash
docker run -d --name strata --gpus all \
  -p 127.0.0.1:8101:8101 \
  --ulimit memlock=-1 \
  -v /srv/strata-data:/data \
  -e MODEL=UD-IQ3_XXS -e CONTEXT=131072 \
  strata
```

- **`/data`** holds the model files (120-170 GB), the expert profile and the run config
  (`/data/config/strata-glm-<size>.json`). The engine is part of the image.
- **Port 8101.** The server listens on `0.0.0.0` inside the container. `-p 127.0.0.1:8101:8101` publishes it to this
  machine only; to publish it to the network, set `-e API_KEY=<secret>` too.
- **The setup choices** are environment variables: `MODEL` (`UD-IQ3_XXS`, the default, or `IQ4_XS`), `CONTEXT`
  (tokens; empty: setup's pick for the VRAM), `PARK_MIB`, `GPUS` (`"0,2"` or `"all"`) or `GPU` (one card), `HOST`
  (default `0.0.0.0`), `PORT` (default 8101), `API_KEY`, `GGUF_DIR` (a mounted folder of shards you already have:
  `-v /models/glm53:/gguf:ro -e GGUF_DIR=/gguf`), `CONFIG` (start with this config file).
- **The first start** runs setup with these choices, `--no-browser` and `--yes`, and keeps the config on the volume.
  Later starts go straight to the server. `-e REINSTALL=1` runs setup again, to change the settings of a size already
  set up. Switching `MODEL` to a size already on the volume needs no setup pass.
- **Health:** the image has a `HEALTHCHECK` on `/health` with a start period of 30 minutes, since the port opens only
  once the model is loaded (and the first start downloads it). `docker logs strata` shows the server's output; the
  engine's log is inside the container: `docker exec strata tail -f /opt/strata/strata-glm-ud-iq3_xxs.log`.
- **RAM:** setup reads the host's RAM from `/proc/meminfo`, not a container memory limit. Do not set a limit below what
  the size needs.
- **Stopping:** the server is PID 1 and stops the engine on SIGTERM. Give it time: `docker stop -t 60 strata` (an
  example value; the stop time is not measured).

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

The server's own options (`python3 serve/server.py --help`): `--host`, `--port` (default 8101), `--api-key`, `--gpu`,
`--chat-template`, `--open`, `--fit-max-tokens`, `--mcp-config`, `--lazy`, `--api-monitor`, `--idle-unload`,
`--min-free-vram-mib`, `--before-load`. Most have a config key too. `--engine mock` runs the
server with a scripted engine and no model, for trying clients. What the API offers: [DETAILS.md](DETAILS.md#the-api).
