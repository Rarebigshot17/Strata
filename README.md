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
| **Graphics cards** | **NVIDIA**, compute capability 7.5 or newer (GeForce RTX 20 series and newer). One card or several: the model's layers are split over them. Driver **580** or newer for the CUDA 13 toolkit setup installs; 525 or newer when it builds with a CUDA 12 toolkit you have. Setup stops with an older one. |
| **RAM** | Every expert of the model stays in RAM: 112.5 GB (104.8 GiB) at 3-bit, counted in the real file; about 149 GB (139 GiB) at 4-bit, derived from its download ([Which size](#which-size-should-i-pick)). Setup counts RAM in GiB, as Linux reports it (`free -g`). It wants at least 12 GiB left beside the experts for Linux and the server, and calls a size a good fit when 40 GiB are left. So Linux must report about **117 GiB** at the least for the 3-bit size (a PC with 128 GB installed reports about 125) and about **151 GiB** for the 4-bit size (192 GB installed: about 188). |
| **Disk** | 120.4 GB free for the 3-bit download, 156.8 GB for the 4-bit one (the repository's listing), and about 10 GB more when setup installs the CUDA toolkit. An SSD helps: the first start after a reboot reads every expert from it. |
| **System** | **Linux** only: setup stops on any other system. A CPU with **AVX2**: without it setup warns, because the experts the cards do not hold run on the CPU. |

The cards hold everything except the routed experts, plus the attention cache and a cache of the most used experts.
How much VRAM the part they must hold takes is not measured yet. Setup installs everything else when it is missing:
Python (through apt, dnf or pacman), the C++ compiler (through apt) and the CUDA toolkit (by itself on Ubuntu 22.04
and 24.04 only).

**The reference machine** is the owner's PC, napster-ai: a Xeon E5-2699 v4 (AVX2, no AVX-512), 192 GB of DDR4
(planned; 91 GiB for now), an RTX 3090 Ti 24 GB + RTX 3080 Ti 12 GB + RTX 3080 10 GB (all compute capability 8.6), a
SATA SSD, Linux. Strata is planned for it: port 8101 (3000, 8000 and 8080 are taken there), contexts up to 512K-1M
tokens, and four agents, a chat and a Discord bot taking turns on one engine.

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
6. **Writes** the run config (`strata-glm-ud-iq3_xxs.json` or `strata-glm-ud-iq4_xs.json`) and a start script
   (`run-glm-ud-iq3_xxs.sh` or `run-glm-ud-iq4_xs.sh`), and starts the model on port **8101**. Your browser opens
   the web app when the model is ready.

**How long this takes is not measured yet.** The download depends on your connection: at 100 MB/s, 120 GB takes about
20 minutes (calculated). Before the server says it is ready, the engine reads every expert into RAM once (the
warm-up; it prints its progress). From a SATA SSD, at most about 550 MB/s, the 3-bit's 112.5 GB take at least about
3.4 minutes (calculated). Setup prints its own estimates for the compile and the toolkit install. None of them is
measured on this version.

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
| **3-bit** | `UD-IQ3_XXS` | 120.4 GB | 112.5 GB (104.8 GiB) |
| **4-bit** | `UD-IQ4_XS` (`UD-Q4_K_XL` when the repository has no UD-IQ4_XS) | 156.8 GB | about 149 GB (139 GiB), derived |

These are setup's own figures. The downloads are the repository's listing (2026-10-20). The 3-bit experts are what
`tools/glm53_inspect.py` counted in the real file on napster-ai; that count takes in the MTP layer's experts too,
which the engine does not load, so it keeps a little less in RAM. The 4-bit experts are not measured: they are its
download less the 7.9 GB the 3-bit file holds beside its experts. Setup reads the real file sizes from the
repository before it downloads. When it takes `UD-Q4_K_XL` (199.7 GB) instead of `UD-IQ4_XS`, it works out that
file's experts as the same share of its size and says what that leaves.

What is left beside the experts holds Linux, the server, your other programs and the parked conversations
([Using it](#using-it)). Setup's verdict is the RAM Linux reports minus the experts, both in GiB:

| Your RAM (Linux reports) | 3-bit: RAM left | 4-bit: RAM left | Setup recommends |
| --- | --- | --- | --- |
| 96 GB (about 94 GiB) | does not fit | does not fit | neither fits: 3-bit is the default answer, and setup asks before it installs it |
| 128 GB (about 125 GiB) | about 20 GiB: tight | does not fit | 3-bit, with less RAM for parked conversations |
| 160 GB (about 157 GiB) | about 52 GiB: fits | about 18 GiB: tight | 3-bit |
| **192 GB** (about 188 GiB) | about 83 GiB: fits | about 49 GiB: fits | **4-bit** |
| 256 GB (about 251 GiB) | about 146 GiB: fits | about 112 GiB: fits | 4-bit |

- **Fits:** 40 GiB or more left.
- **Tight:** 12 to 40 GiB left. Setup warns and gives parked conversations 4096 MiB instead of 8192
  (`--park-mib 4096`). Close other big programs while the model runs.
- **Does not fit:** under 12 GiB left. Linux would read experts from the disk while the model answers: very slow, or
  no start at all. Setup asks before it goes on (the answer is no unless you type y), and `--yes` alone does not
  go on. `--model <size> --yes` does.

The RAM Linux reports in this table is about 2% under the RAM installed (188 GiB for 192 GB, as setup's tests
take it); it differs a little from PC to PC. `./setup.sh --check` prints the figures for your PC without installing
anything. On napster-ai with 192 GB setup recommends the 4-bit size; with the 91 GiB it has for now, neither size
fits. The 4-bit column rests on the derived figure: its file's own count may move it a few GiB.

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
  port can use the model: [what the server exposes](docs/DETAILS.md#security).
- **Thinking:** GLM-5.3-Flash thinks before it answers, or it does not. It has no levels. In the web app, open
  the sampling settings and choose Thinking **Off** or **On**. In the API, `reasoning_effort: "none"` turns it off,
  and low, medium and high all mean on. `"chat_template_kwargs": {"enable_thinking": false}` and
  `"thinking": {"type": "disabled"}` (Anthropic's, and Z.ai's on the OpenAI route) turn it off too. A request that
  says nothing gets thinking on, unless the run config sets `reasoning_effort` (or `"anthropic_thinking":
  "on_request"` for Anthropic requests).
- **Tool calls** are read from the model's own special tokens only. A call that the model copies as plain text, from
  a tool result, a web page or the user's message, stays text: it is never returned as a call, and never runs as an
  MCP tool. This needs GLM's tags to be special tokens in the model's tokenizer; that is expected, but not checked
  on the real GGUF yet (the server window says at the start: a `WARNING` line means it reads them from the text). A
  request that declares no tools, or sets `tool_choice` to `"none"`, gets no calls at all. This matters for an agent
  that runs shell commands. MCP tools set up on the server run only for Strata's own web page, unless the run config
  says `"mcp_for_api": true`.
- **Several clients:** Strata answers one request at a time. The others wait, in the order they arrived. When a
  request belongs to another conversation, the engine parks the current conversation's state in RAM. A later turn
  of a parked conversation reads only the new tokens, not the whole conversation again. Parked conversations get
  8192 MiB of RAM by default (`--park-mib`), and the one used least recently goes first. A parked 100K-token
  conversation takes about 1.4 GB (calculated). How fast a parked turn resumes is not measured yet.
- **Context:** the longest conversation the engine accepts. Setup offers 32K, 128K (recommended when the cards
  have 16 GB or more together), 256K, 512K and 1M tokens (the model's full context). Each token takes 12.4 KB of
  VRAM in the attention cache (the MLA latent and the indexer's pooled keys), and that VRAM is taken from the expert
  cache:

  | Context | 32K | 128K | 256K | 512K | 1M |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | VRAM for the attention cache (calculated) | 0.4 GB | 1.7 GB | 3.3 GB | 6.6 GB | 13.3 GB |

  On napster-ai's 46 GB of VRAM, 1M tokens leaves about 33 GB for the rest (calculated). Fewer cached experts means
  more of them run on the CPU; how much slower that is has not been measured. Setup warns when the attention cache
  would take more than half of the cards' VRAM. A request whose prompt plus `max_tokens` does not fit the context
  gets a 400; it is never cut (`"fit_max_tokens": true` in the run config lowers `max_tokens` instead). To change the
  context later: `./setup.sh --setup`, or `--ctx` in the [run config](docs/INSTALL.md#the-run-config).
- **Pictures:** GLM-5.3-Flash reads text only. A request with a picture gets a 400. A picture in a tool's result is
  replaced by a short note instead, so an agent's history keeps working.
- **Speed on your PC:** `./bench.sh` measures your install (chat, 4K and 16K prompts, an agent-style session) and
  writes a report. No results are published yet.

Settings, token biases and monitoring: [docs/DETAILS.md](docs/DETAILS.md#settings),
[docs/LOGIT_BIAS.md](docs/LOGIT_BIAS.md) and [Prometheus and Grafana](docs/DETAILS.md#monitoring). Beside other
model servers under llama-swap: [docs/LLAMA_SWAP.md](docs/LLAMA_SWAP.md).

## Something went wrong?

- **"no NVIDIA GPU found" or "the NVIDIA driver is too old".** Install or update the NVIDIA driver (580 or newer,
  or 525 or newer with a CUDA 12 toolkit; the message says which; Ubuntu: `sudo ubuntu-drivers install`), restart,
  and run `./setup.sh` again.
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
  host names it knows (an IP address always passes). Set an API key (see [Using it](#using-it)), or add the name to
  `"allowed_hosts"` in the run config.

More problems and their fixes: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md). Still stuck? Keep the engine's
log, `strata-glm-<size>.log` in the Strata folder, and the server window's output:
[what to collect](docs/TROUBLESHOOTING.md#what-to-collect-when-something-goes-wrong).

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
- Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT). llama.cpp is also the
  reference the engine is checked against: so far the CPU reference, on small random models (see [Status](#status)).

Strata is open source under the [MIT License](LICENSE). A few parts have their own licenses: `third_party/ggml`
(MIT, llama.cpp / ggml) and the web app's font (SIL Open Font License 1.1, `serve/web/fonts/OFL.txt`). The model is
not part of this repository; its own license applies to its files.
