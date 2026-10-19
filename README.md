<h1 align="center">Strata</h1>

<p align="center"><b>Run GLM-5.3-Flash, a 320-billion-parameter AI model, on one Linux PC</b><br>
NVIDIA graphics cards and a lot of system RAM · Linux · free and open source</p>

Strata runs **GLM-5.3-Flash** (by Z.ai) on a PC with NVIDIA graphics cards and a lot of system RAM. It has a C++/CUDA
engine, a server with OpenAI-, Anthropic- and Responses-compatible APIs and a web app, and an installer. Your apps,
coding agents and bots talk to it as they would to a cloud model, and nothing leaves your PC. Strata is a fork of
[Niko1221/Strata](https://github.com/Niko1221/Strata), which runs Qwen3.8-Flash-Next on gaming PCs. This fork keeps
its design, its server and its web app, replaces the engine, and runs GLM-5.3-Flash only.

> **Status: the engine has not run on a GPU or on the real model yet.** There are no measured speeds, load times or
> memory figures. The numbers on this page are calculated from the model's shapes, and they say so. What is checked
> and what is not: [Status](#status).

## What you need

| | |
| --- | --- |
| **Graphics cards** | **NVIDIA**, compute capability 7.5 or newer (GeForce RTX 20 series and newer). One card or several: the model's layers are split over them. Driver **580** or newer: setup stops with an older one. |
| **RAM** | Every expert of the model stays in RAM: about 117 GB at 3-bit, about 162 GB at 4-bit (calculated). Setup wants at least 12 GB left beside them for Linux and the server, and recommends a size only when 40 GB are left. So about **130 GB** at the least for the 3-bit size and about **175 GB** for the 4-bit size. [Which size](#which-size-should-i-pick). |
| **Disk** | About 120 GB free for the 3-bit download, about 170 GB for the 4-bit one, and about 10 GB more when setup installs the CUDA toolkit. An SSD helps: the first start after a reboot reads every expert from it. |
| **System** | **Linux** only: setup stops on any other system. A CPU with **AVX2**: without it setup warns, because the experts the cards do not hold run on the CPU. |

The cards hold everything except the routed experts, plus the attention cache and a cache of the most used experts.
How much VRAM the part they must hold takes is not measured yet. Setup installs everything else: Python, the C++
compiler and the CUDA toolkit when they are missing (the toolkit by itself on Ubuntu 22.04 and 24.04 only).

**The reference machine** is the owner's PC, napster-ai: a Xeon E5-2699 v4 (AVX2, no AVX-512), 192 GB of DDR4, an
RTX 3090 Ti 24 GB + RTX 3080 Ti 12 GB + RTX 3080 10 GB (all compute capability 8.6), a SATA SSD, Linux. Strata is
planned for it: port 8101 (3000, 8000 and 8080 are taken there), contexts up to 512K-1M tokens, and four agents, a
chat and a Discord bot taking turns on one engine.

The full list: [docs/INSTALL.md](docs/INSTALL.md#what-you-need).

## Install

### Let your AI set it up

Do you use an AI coding assistant (Claude Code, Codex, Cursor, ...)? Paste this into it:

```text
Set up Strata on this PC for me: https://github.com/Rarebigshot17/Strata (branch glm53-flash) - follow
docs/AI_SETUP.md in that repository.
```

It checks your cards, RAM and disk, picks the size that fits, runs setup without questions, starts the server and
tells you how to connect your apps. The steps it follows: [docs/AI_SETUP.md](docs/AI_SETUP.md).

### Or do it yourself

```bash
git clone -b glm53-flash https://github.com/Rarebigshot17/Strata.git
cd Strata
./setup.sh
```

(Without git: download [the zip](https://github.com/Rarebigshot17/Strata/archive/refs/heads/glm53-flash.zip), unzip
it and run `./setup.sh` in the folder.)

`./setup.sh` makes a Python environment in `.venv` and runs `setup.py`. When Python 3.10 or newer is missing, it
installs it through apt, dnf or pacman first. Setup asks **two questions**: which size (3-bit or 4-bit) and how much
context. Press Enter for the recommended answer, or pass `--yes` to take them all. Then it does the rest, and skips
every step that is already done:

1. **Checks your PC:** the NVIDIA cards and driver, the RAM and the CPU. It stops with a plain message when the
   engine cannot run there.
2. **Asks the questions** and lists the model's files on Hugging Face
   ([unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF)), with their sizes. It stops
   when the disk has too little free space for them.
3. **Installs the Python packages** into `.venv`.
4. **Compiles the engine** (`strata-glm`, and the CPU reference `strata-glm-ref`) for your cards, with ggml from
   llama.cpp's source at a fixed commit. When the C++ compiler or the CUDA toolkit is missing, it asks first and
   installs them (sudo asks for your password).
5. **Downloads the model.** A stopped download resumes where it stopped. Each file's size and SHA-256 are checked
   against the repository's listing. The files go to `Strata-GLM-data/models` next to the Strata folder
   (`--data-dir` puts them elsewhere; `--gguf-dir` uses files you already have).
6. **Writes** the run config (`strata-glm-ud-iq3_xxs.json` or `strata-glm-iq4_xs.json`) and a start script
   (`run-glm-ud-iq3_xxs.sh` or `run-glm-iq4_xs.sh`), and starts the model on port **8101**. Your browser opens the
   web app when the model is ready.

**How long this takes is not measured yet.** The download depends on your connection: at 100 MB/s, 120 GB takes
about 20 minutes (calculated). Before the server says it is ready, the engine reads every expert into RAM once (the
warm-up; it prints its progress). From a SATA SSD, at most about 550 MB/s, 117 GB takes at least about 3.5 minutes
(calculated). Setup prints its own estimates for the compile and the toolkit install. None of them is measured on
this version.

**Next time**, run `./setup.sh` again (or the start script). It starts right away and downloads nothing twice.
Press Ctrl+C, or close the terminal, to stop the model. `./update.sh` updates Strata without starting it.
`./setup.sh --setup` sets up the other size or changes the answers. `./setup.sh --check` only checks the PC and
says which size fits. Docker, every option, where the files go and how to update:
[docs/INSTALL.md](docs/INSTALL.md) ([the run config](docs/INSTALL.md#the-run-config),
[updating](docs/INSTALL.md#updating)).

## Which size should I pick?

The model's experts are compressed ("quantized") to about 3 or 4 bits. All of them stay in RAM, so your RAM decides.
The 4-bit size is closer to the full model. How much better its answers are is not measured here.

| Size | Setup's name | Download | Experts in RAM |
| --- | --- | ---: | ---: |
| **3-bit** | `UD-IQ3_XXS` | about 120 GB | about 117 GB |
| **4-bit** | `IQ4_XS` (`UD-Q4_K_XL` when the repository has no IQ4_XS) | about 170 GB | about 162 GB |

These sizes are setup's own figures, calculated from the model's shapes (about 305 billion expert parameters at
3.06 or 4.25 bits each), not measured. Setup reads the real file sizes from the repository before it downloads.
When it takes `UD-Q4_K_XL` instead of `IQ4_XS`, it works out that file's share of experts from its real size and
says what that leaves.

What is left beside the experts holds Linux, the server, your other programs and the parked conversations
([Using it](#using-it)). Setup's verdict is your RAM minus the experts:

| Your RAM | 3-bit: RAM left | 4-bit: RAM left | Setup recommends |
| --- | --- | --- | --- |
| 128 GB | about 11 GB: does not fit | does not fit | neither: it asks before it installs one anyway |
| 144 GB | about 27 GB: tight | does not fit | 3-bit, with less RAM for parked conversations |
| **192 GB** | about 75 GB: fits | about 30 GB: tight | **3-bit** |
| 256 GB | about 139 GB: fits | about 94 GB: fits | 4-bit |

- **Fits:** 40 GB or more left.
- **Tight:** 12 to 40 GB left. Setup warns and gives parked conversations 4 GB instead of 8 (`--park-mib 4096`).
  Close other big programs while the model runs.
- **Does not fit:** under 12 GB left. Linux would read experts from the disk while the model answers: very slow, or
  no start at all. Setup asks before it goes on (the answer is no unless you type y), and `--yes` alone does not
  go on. `--model <size> --yes` does.

These figures are calculated with the RAM that is installed. Setup counts the RAM that Linux reports, which is a few
GB less, so the figures it prints are a few GB lower. `./setup.sh --check` prints them for your PC without
installing anything. On napster-ai (192 GB) setup recommends the 3-bit size.

## Using it

- **In the browser:** open `http://127.0.0.1:8101`. It has **Chat**, a live **Monitor** of the model and your
  GPUs, CPU and RAM, and **About** with the settings and addresses. In a terminal: `.venv/bin/python chat.py`.
- **Your apps and agents:** add an "OpenAI-compatible" provider with the base URL **`http://<host>:8101/v1`**
  (`http://127.0.0.1:8101/v1` on the same PC). Any model name works. The API key is the one you set, or anything
  when you set none.
  - Apps that use Anthropic's API: `http://<host>:8101/v1/messages` (Claude Code:
    `ANTHROPIC_BASE_URL=http://127.0.0.1:8101`).
  - Codex CLI and other apps that use OpenAI's Responses API: `http://<host>:8101/v1/responses`. It stores nothing:
    the client sends the whole conversation each time.
  - Plain text completions: `/v1/completions`. Every endpoint and field: [docs/DETAILS.md](docs/DETAILS.md#the-api).
- **Open WebUI in Docker, and other machines:** Strata must listen beyond this PC, and then it needs an API key:

  ```bash
  ./setup.sh --host 0.0.0.0 --api-key "$(openssl rand -hex 24)"
  ```

  Both are saved for the model. In Open WebUI add an OpenAI connection with that key and the address its container
  reaches the host at: `http://172.17.0.1:8101/v1` on Docker's default bridge, or `http://100.126.222.90:8101/v1`
  over Tailscale on napster-ai (the agents and the Discord bot can use that address too). Clients send the key as
  `Authorization: Bearer <key>` or `x-api-key`; the web page asks for it. Without a key, anyone who can reach the
  port can use the model: [SECURITY.md](SECURITY.md).
- **Thinking:** GLM-5.3-Flash thinks before it answers, or it does not. It has no levels. In the web app, open
  the sampling settings and choose Thinking **Off** or **On**. In the API, `reasoning_effort: "none"` turns it off,
  and low, medium and high all mean on. `"chat_template_kwargs": {"enable_thinking": false}` and
  `"thinking": {"type": "disabled"}` (Anthropic's, and Z.ai's on the OpenAI route) turn it off too. A request that
  says nothing gets thinking on.
- **Tool calls** are read from the model's own special tokens only. A call that the model copies as plain text, from
  a tool result, a web page or the user's message, stays text and is never run. A request that declares no tools,
  or sets `tool_choice` to `"none"`, gets no calls at all. This matters for an agent that runs shell commands. MCP
  tools set up on the server run only for Strata's own web page, unless the run config says `"mcp_for_api": true`.
- **Several clients:** Strata answers one request at a time. The others wait, in the order they arrived. When a
  request belongs to another conversation, the engine parks the current conversation's state in RAM. A later turn
  of a parked conversation reads only the new tokens, not the whole conversation again. Parked conversations get
  8 GB of RAM by default (`--park-mib`), and the one used least recently goes first. A parked 100K-token
  conversation takes about 1.3 GB (calculated). How fast a parked turn resumes is not measured yet.
- **Context:** the longest conversation the engine accepts. Setup offers 32K, 128K (recommended when the cards
  have 16 GB or more together), 256K, 512K and 1M tokens (the model's full context). Each token takes 11 KB of VRAM
  in the attention cache, and that VRAM is taken from the expert cache:

  | Context | 32K | 128K | 256K | 512K | 1M |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | VRAM for the attention cache (calculated) | 0.4 GB | 1.5 GB | 3.0 GB | 5.9 GB | 11.8 GB |

  On napster-ai's 46 GB of VRAM, 1M tokens leaves about 34 GB for the rest. Fewer cached experts means more of them
  run on the CPU; how much slower that is has not been measured. Setup warns when the attention cache would take
  more than half of the cards' VRAM. A request whose prompt plus `max_tokens` does not fit the context gets a 400;
  it is never cut. To change the context later: `./setup.sh --setup`, or `--ctx` in the
  [run config](docs/INSTALL.md#the-run-config).
- **Pictures:** GLM-5.3-Flash reads text only. A request with a picture gets a 400.
- **Speed on your PC:** `./bench.sh` measures your install (chat, 4K and 16K prompts, an agent-style session) and
  writes a report. No results are published yet.

Settings, token biases and monitoring: [docs/DETAILS.md](docs/DETAILS.md#settings),
[docs/LOGIT_BIAS.md](docs/LOGIT_BIAS.md) and [Prometheus and Grafana](docs/DETAILS.md#monitoring). Beside other
model servers under llama-swap: [docs/LLAMA_SWAP.md](docs/LLAMA_SWAP.md).

## Something went wrong?

- **"no NVIDIA GPU found" or "the NVIDIA driver is too old".** Install or update the NVIDIA driver (580 or newer;
  Ubuntu: `sudo ubuntu-drivers install`), restart, and run `./setup.sh` again.
- **The CUDA toolkit does not install.** Setup installs it by itself on Ubuntu 22.04 and 24.04 only. Elsewhere,
  install CUDA 12.0 or newer (13.0 for an RTX 50 card) from your distribution or from NVIDIA, then run `./setup.sh`
  again.
- **It stopped while downloading or compiling.** Run `./setup.sh` again. It continues where it stopped. A file with
  the wrong size or SHA-256 is deleted, and the next run downloads it again.
- **The model is very slow and the disk is busy all the time.** The experts do not fit in the free RAM, so Linux
  reads them from the disk. Close other big programs, or take the 3-bit size (`./setup.sh --setup`).
- **"port 8101 is already in use".** Strata (or another server) is already running. Stop it, or start with
  `--port <another port>`.
- **Another machine or a container gets 403 "Host ... is not allowed".** Without an API key Strata answers only
  names it knows. Set an API key (see [Using it](#using-it)), or add the name to `"allowed_hosts"` in the run
  config.

More problems and their fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Still stuck? Open an
[issue](https://github.com/Rarebigshot17/Strata/issues) and attach the engine's log, `strata-glm-<size>.log` in the
Strata folder. Found a security problem? Report it privately: [SECURITY.md](SECURITY.md).

## How does it work?

A model this size normally runs on servers with hundreds of gigabytes of graphics memory. Strata makes it fit by
**sharing the work across the whole PC**:

- **The model is a team of small specialists ("experts").** Each of its 42 expert layers has 288 of them, and each
  word needs only 8 per layer.
- **The graphics cards** do the part every word needs (the attention layers, the routers, the shared experts) and
  keep the experts that are asked for most often. They keep learning which ones those are.
- **The RAM** holds every expert. When a word needs one the cards do not have, **the processor** computes it while
  the cards do their part.
- **Several conversations** take turns on one engine. Each one's state is parked in RAM, so a new turn does not read
  the whole conversation again.

The longer explanation: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md).

## Status

What is checked, without a GPU and without the real model:

- **The CPU reference** (`strata-glm-ref`) matches llama.cpp to float rounding on small random GLM-5.3-Flash models
  (`tools/glm53_selftest.sh`).
- **Conversation parking and checkpoints:** every resumed reply is the same, token for token, as a fresh engine's
  (`tools/glm53_park_check.py`, on the CPU engine).
- **The server end to end** on the CPU engine: tokenizer, chat, streaming, a resumed second turn, completions,
  metrics, a request that does not fit (`tools/glm53_server_check.sh`).
- **The tests that need no GPU:** about 750 Python tests of the server, the setup and the tools' tests, and the
  engine's C++ unit tests (`ctest`). How to run them: [AGENTS.md](AGENTS.md).

What is **not** checked yet:

- **The GPU engine** (`strata-glm`) is written but has never run on a GPU. Its logits have not been compared with
  llama.cpp's on the real model.
- **The real model** has not run in Strata. The tests use GLM-4.7-Flash's chat template. GLM-5.3-Flash's own
  template and special tokens are read from its GGUF, and have not been checked against the server yet.
- **No speed, load time or memory use is measured.** Every such figure in these docs is calculated.
- **A full install** that ends with the real model answering has not run yet.

The steps that check these on napster-ai, one by one: [docs/GLM53_PORT.md](docs/GLM53_PORT.md) ("What to run on
napster-ai"). The server port's decisions, the engine protocol and the plan's state:
[docs/GLM53_SERVER_PORT.md](docs/GLM53_SERVER_PORT.md).

## Credits and license

- The model is GLM-5.3-Flash by Z.ai. The GGUF files setup downloads are
  [Unsloth's](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF). Their license applies to the model files.
- Strata comes from [Niko1221/Strata](https://github.com/Niko1221/Strata) by its author and contributors. This fork
  keeps its design (experts in RAM, the most used cached in VRAM, the rest on the CPU), its server, its web app and
  its installer, and replaces the Qwen engine with a GLM-5.3-Flash one.
- Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT), and every part of the engine
  is checked against llama.cpp.

Strata is open source under the [MIT License](LICENSE). A few parts and the model have their own licenses
([which ones](docs/HOW_IT_WORKS.md)).
