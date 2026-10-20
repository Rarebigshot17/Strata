# Setting up Strata for a user (instructions for AI assistants)

This page is for an AI coding assistant (Claude Code, Codex, Cursor, ...) that a user asked to set up Strata on their
PC. Strata runs one model, GLM-5.3-Flash, on Linux with NVIDIA GPUs and system RAM, and serves an OpenAI-, Anthropic-
and Responses-compatible API and a chat page on `http://127.0.0.1:8101`. The version for people, with every detail,
is [INSTALL.md](INSTALL.md).

Work through the steps in order. Tell the user what you are doing in plain words.

> **Nothing has run on a GPU or on the real model yet.** The project has no measured speeds, load times or memory
> figures. The sizes below are calculated. Do not promise the user a speed or a load time. If you time something on
> their PC, say what you timed and that it was one run on their PC.

## 0. Rules

- **Never expose the server beyond `127.0.0.1` without an API key.** Keep the default host. Only when the user asks
  for access from other devices, use `--host 0.0.0.0` **together with** `--api-key <a long random secret>`, and give
  the user the key. Make one with `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`. Never put a tunnel,
  a port forward or a published Docker port in front of a server without a key. The Docker image listens on
  `0.0.0.0` without a key unless you set `API_KEY`: publish it as `-p 127.0.0.1:8101:8101`.
- **You cannot type the user's sudo password.** `./setup.sh` uses sudo to install Python with venv when it is
  missing, and setup uses sudo to install the C++ compiler and the CUDA toolkit when they are missing. Check for them
  first (step 2). If something is missing, ask the user to run the commands in step 4 in their own terminal.
- **Do not install or change NVIDIA drivers** unless the user asks. Strata needs driver 580 or newer (525 or newer
  when an installed CUDA 12 toolkit builds the engine).
- **The download is big:** about 120 GB for the 3-bit size, about 170 GB for the 4-bit. Ask the user before you start,
  especially on a metered connection. At 100 Mbit/s, 120 GB takes about 2.7 hours (calculated).
- **Setup runs for a long time** (the build, the download). Run it in the background or with a long timeout and read
  its output; do not stop it because it is quiet for a while. It is resumable: the same command again continues where
  it stopped.
- **Always pass `--yes`.** Without a terminal, setup stops at its first question ("input ended before a setup answer
  was received").
- **Do not edit setup's or the server's code** to get past an error. Report the error to the user.

## 1. Get Strata

```bash
git clone -b glm53-flash https://github.com/Rarebigshot17/Strata.git
cd Strata
```

Without git: download https://github.com/Rarebigshot17/Strata/archive/refs/heads/glm53-flash.zip and unzip it. The
model files go to a `Strata-GLM-data` folder **next to** the Strata folder, so pick a drive with room for them, or use
`--data-dir <folder on a bigger drive>` in step 4.

## 2. Check the PC

Setup's own check prints the GPUs, the driver, the RAM, the CPU and, for each size, whether it fits this PC's RAM and
free disk space. It installs, moves and writes nothing, and runs with the system Python:

```bash
python3 setup.py --check
```

It ends with `This PC can run GLM-5.3-Flash (<size> recommended)` and exit code 0, or with exit code 1: no NVIDIA GPU
(`nvidia-smi did not answer`), a driver older than the CUDA toolkit needs (580 for CUDA 13, 525 for CUDA 12), or
`neither size fits <N> GiB of RAM`.

Check the rest by hand:

| What | Command | What it must show |
| --- | --- | --- |
| OS | `cat /etc/os-release` | Linux. Ubuntu 22.04 or 24.04 lets setup install the CUDA toolkit itself. |
| GPUs, VRAM, driver | `nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv` | Compute capability 7.5 or newer (RTX 20, 30, 40, 50 series), driver 580 or newer (525 or newer with an installed CUDA 12 toolkit). |
| RAM | `free -g` (the `total` of `Mem:`) | See step 3. |
| Free disk | `df -h .` (and the drive for `--data-dir`) | 120 GB (3-bit) or 170 GB (4-bit), plus about 8-10 GB if the CUDA toolkit is to be installed. |
| CPU | `grep -m1 -ow avx2 /proc/cpuinfo` | `avx2`. Without it setup goes on with a warning. |
| CUDA toolkit | `nvcc --version`, or `ls /usr/local/cuda*/bin/nvcc /opt/cuda*/bin/nvcc` | 12.0 or newer (13.0 or newer for an RTX 50 card). Missing: step 4. |
| C++ compiler | `g++ --version` | Present. Missing: step 4. |
| Python | `python3 -c "import sys, venv, ensurepip; print(sys.version)"` | 3.10 or newer, no error. On Ubuntu a missing `ensurepip` means `python3-venv` is missing. |

If the PC does not meet these, tell the user which part is missing and stop. Details:
[INSTALL.md](INSTALL.md#what-you-need).

## 3. Pick the size and the context

Every routed expert stays in RAM. Setup compares the RAM `/proc/meminfo` reports (the `total` of `free -g`, in GiB)
with what each size's experts take in GiB (calculated, from `setup.py`: about 109 GiB at 3-bit, 151 GiB at 4-bit):

| RAM (`free -g`) | Size | Why |
| --- | --- | --- |
| 191 GiB or more | `--model UD-IQ4_XS` (4-bit) | Its ~151 GiB of experts leave 40 GiB or more. Setup recommends it. |
| 163-190 GiB | `--model UD-IQ3_XXS` (3-bit) | Recommended. The 4-bit would leave under 40 GiB: setup allows it with a warning and gives parked conversations 4096 MiB. A PC with 192 GB installed (about 188 GiB) is here. |
| 149-162 GiB | `--model UD-IQ3_XXS` | Recommended; the 4-bit does not fit. |
| 121-148 GiB | `--model UD-IQ3_XXS` | Tight: setup warns and gives parked conversations 4096 MiB. Tell the user to close other big programs while it runs. |
| under 121 GiB | none | Neither size fits. Tell the user. Setup can be forced (`--model UD-IQ3_XXS --yes`), but the OS would then read experts from the disk: very slow, or no start at all. |

When the repository has no UD-IQ4_XS files, the 4-bit size takes UD-Q4_K_XL instead; setup says so and works out
what it leaves of the RAM.

Ask the user how long their conversations get (chats, files, tool output together), and pick the context from that.
Each token takes 12.4 KB of VRAM, which the GPUs' expert cache gives up (calculated):

| `--context` | VRAM for the attention cache |
| --- | --- |
| 32768 | 0.4 GB |
| 131072 | 1.7 GB (setup's default when the GPUs have 16 GB of VRAM or more in all) |
| 262144 | 3.3 GB |
| 524288 | 6.6 GB |
| 1048576 | 13.3 GB (the model's whole context) |

If several apps or agents will take turns on long conversations, read
[Parking and several clients](INSTALL.md#parking-and-several-clients) and tell the user about `--park-mib`.

## 4. Run setup without questions

If step 2 found no C++ compiler or no CUDA toolkit, ask the user to run these first (Ubuntu 24.04; for 22.04 use
`ubuntu2204`), or to run `./setup.sh` once in their own terminal, where setup asks and uses sudo itself:

```bash
sudo apt-get install -y build-essential python3-venv
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-13-0
```

Then, from the Strata folder:

```bash
nohup ./setup.sh --yes --model UD-IQ3_XXS --context 131072 --gpus all --no-browser --no-start > setup.out 2>&1 &
tail -f setup.out
```

| Option | Why |
| --- | --- |
| `--yes` | The recommended answer to every question; no prompts. |
| `--model UD-IQ3_XXS` or `UD-IQ4_XS` | The size from step 3. With `--yes`, naming it is also the user's consent to a size setup would otherwise stop at. |
| `--context N` | The context from step 3. |
| `--gpus all` | Every usable card, written into the config in nvidia-smi's order (or `--gpus 0,2` for some of them). Without it the engine uses CUDA's own order. |
| `--no-browser` | Do not open a browser when the model is ready. |
| `--no-start` | Install only. Setup would otherwise run the server in the foreground and block your shell. |
| `--data-dir DIR` | Only when the drive next to the Strata folder has no room: the model files go there (remembered). |
| `--host 0.0.0.0 --api-key KEY` | Only when the user asked for access from other devices, and always both. |

Every option: `./setup.sh --help`, or [INSTALL.md](INSTALL.md#setups-options).

Setup prints `=== Step N: ... ===` for its six steps. The first build takes a while (setup says 10-20 minutes; not
measured on this engine), then the download, then a SHA-256 check of each file. It succeeded when it prints
`[ok] run config: strata-glm-<size>.json; start script: run-glm-<size>.sh`, then `All set.` and the addresses, and
exits with code 0. A line starting with `[X]` is the reason it stopped, with a hint below it. A line starting with
`[!]` is a warning: pass it on to the user.

## 5. Start the server

```bash
nohup ./run-glm-ud-iq3_xxs.sh > strata-server.out 2>&1 &
echo $! > strata-server.pid
```

The script name is the one setup printed (`run-glm-ud-iq4_xs.sh` for the 4-bit; `run-glm-iq4_xs.sh` for a 4-bit
install made when setup called it `IQ4_XS`). For a server that starts with the PC, use the systemd unit in
[INSTALL.md](INSTALL.md#running-it-as-a-service) instead, with the user's consent (it needs sudo).

The port opens only when the engine is ready. Before that, the engine reads every routed expert into memory: about
117 GB (3-bit) or 162 GB (4-bit). At 550 MB/s, the top speed of a SATA SSD, that is about 3.5 or 5 minutes
(calculated, not measured). `strata-server.out` shows the progress (`[strata] reading the experts into memory ...`,
then `warming the experts: X of Y GB` about every 20 seconds). The server gives up when the engine has not said it
is ready within 900 seconds; each step the engine reports (the layers on each GPU, the expert cache, the warm-up's
progress) starts those 900 seconds again. The start is done
when the output has a line `ready: http://127.0.0.1:8101/v1 ...`.

To stop it: `kill $(cat strata-server.pid)`. That sends SIGTERM to the server; it stops the engine, and the engine
saves its expert profile. Do not `kill -9` the server or `strata-glm`: the profile of that run can then be lost. A
plain `kill` of `strata-glm` is ignored while the server runs.

## 6. Verify

Poll until the port answers (connection refused means it is still loading):

```bash
curl -s http://127.0.0.1:8101/health
```

It answers `{"status": "ok", "max_context": ..., "model": "...", "images": false, "api_key": false, "loaded": true,
"service": "strata"}`. `/health` needs no key. Then:

```bash
curl -s http://127.0.0.1:8101/v1/models
curl -s http://127.0.0.1:8101/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model": "glm-5.3-flash", "messages": [{"role": "user", "content": "Say hello in five words."}],
       "max_tokens": 64, "reasoning_effort": "none"}'
```

With an API key, add `-H "Authorization: Bearer <key>"` to the `/v1` requests. `"reasoning_effort": "none"` turns the
model's thinking off for this test, so the 64 tokens go to the answer; without it, the model thinks first and its
thinking comes back in `reasoning_content`. The engine's log is `strata-glm-<size>.log` in the Strata folder.

## 7. Connect the user's apps

| App | Setting |
| --- | --- |
| Browser | `http://127.0.0.1:8101/`: chat, a live monitor, settings. |
| Any OpenAI-compatible app | Base URL `http://127.0.0.1:8101/v1`. API key: the configured one, or any text when there is none. Model name: any (`/v1/models` lists `glm-5.3-flash`). |
| Anthropic-compatible apps | Base URL `http://127.0.0.1:8101` (they call `/v1/messages`). Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8101`, `ANTHROPIC_AUTH_TOKEN=<the key, or any text>`; the model name it sends is ignored. |
| Codex CLI and other Responses API apps | `http://127.0.0.1:8101/v1/responses` ([DETAILS.md](DETAILS.md#the-api)). |
| Open WebUI | Admin Panel, Settings, Connections, OpenAI API: URL `http://127.0.0.1:8101/v1` and the key. If Open WebUI runs in Docker, `127.0.0.1` there is the container itself: with host networking the URL above works; otherwise Strata must listen beyond `127.0.0.1` (`--host 0.0.0.0 --api-key KEY`, rule 0) and Open WebUI uses `http://host.docker.internal:8101/v1` (on Linux the container needs `--add-host=host.docker.internal:host-gateway`). |

What the user should know about it:

- **One request at a time.** Every app shares the one engine; requests wait in the order they came. A long answer for
  one app delays the others.
- **Thinking:** on by default. `"reasoning_effort": "none"` turns it off for a request; `low`, `medium` and `high` all
  mean on (GLM has no levels). `"reasoning_effort"` in the run config sets the default.
- **Agents that run tools** (a shell, the web): the model's tool calls come back as tool calls only when the request
  declares tools, and not with `"tool_choice": "none"`. When the model's tokenizer has GLM's tags as special tokens, a
  call is read only from those tokens, so call-like text copied from a tool result, a web page or the user stays text
  and is never run. Whether the real GGUF has them as special tokens is still to be checked on a real install
  ([GLM53_SERVER_PORT.md](GLM53_SERVER_PORT.md)): the server's start line says it, and a `WARNING` line about tool
  calls read from the plain text means it does not; tell the user. The agent still decides what it runs: keep its own
  confirmation for shell commands.
- **MCP tools** set up in Strata's run config run for Strata's own chat page only, unless the config says
  `"mcp_for_api": true`.
- API details: [DETAILS.md](DETAILS.md#the-api); every config key: [INSTALL.md](INSTALL.md#the-run-config).

## 8. When something fails

| What it says | What to do |
| --- | --- |
| `no NVIDIA GPU found (nvidia-smi did not answer)` | The NVIDIA driver is missing or not loaded. The user installs it (Ubuntu: `sudo ubuntu-drivers install`) and restarts. |
| `the NVIDIA driver is too old` | The user updates it to the version the message names (580 or newer for CUDA 13, 525 for CUDA 12) and restarts; then run setup again. |
| `input ended before a setup answer was received` | Add `--yes`. |
| `the build tools are needed`, or a sudo password prompt | Ask the user to run the commands in step 4, then run setup again. |
| `the CUDA Toolkit can be installed automatically on Ubuntu 22.04 / 24.04 only` | The user installs the toolkit from their distribution or https://developer.nvidia.com/cuda-downloads. |
| `not enough free disk space in ...` | Run setup again with `--data-dir <a folder on a bigger drive>`. |
| `cannot list the model's files` | No connection to huggingface.co. Check the network, or set `HF_ENDPOINT` to a mirror. |
| `... do not fit beside Linux and the server ...` | Not enough RAM for that size: choose the 3-bit size, or tell the user. |
| A download stopped | Run the same command again; it continues. |
| `port 8101 is already in use` | Strata (or another server) already runs: check `curl http://127.0.0.1:8101/health`. Or start with `--port N`. |
| `the engine did not start`, `the engine exited before it was ready`, `the engine did not report READY within 900 s` | Read the end of `strata-glm-<size>.log` and report it. |
| `... is not an address of this PC` | The `host` in the run config is wrong. |
| HTTP 401 | The server has an API key: send it. |
| HTTP 403 `Host ... is not allowed (DNS rebinding protection)` | The app uses a host name: set an API key, or add the name to `"allowed_hosts"` in the run config. |

More: [TROUBLESHOOTING.md](TROUBLESHOOTING.md) and [the README](../README.md#something-went-wrong). If it still fails,
collect the setup output, `strata-server.out` and `strata-glm-<size>.log`, and tell the user what you saw.

## 9. What to report to the user

When you are done, tell the user:

- **What is installed:** the size, the context, the GPUs (in their order), the Strata folder, the data folder with the
  model files, the run config (`strata-glm-<size>.json`) and the log.
- **How to start and stop it:** `./setup.sh` or `./run-glm-<size>.sh` in a terminal (Ctrl+C or closing it stops the
  model, and the engine saves its expert profile), or the service if you set one up.
- **The addresses:** the chat page, the OpenAI base URL, the Anthropic base URL. If you set an API key: the key, and
  that it is stored in plain text in the run config.
- **What you checked:** the `/health` answer and the test answer.
- **Every warning setup printed** (a `[!]` line): little RAM left, a long context on little VRAM, no AVX2.
- **What you could not do:** a sudo step, a driver, anything the user still has to run.
- **That speeds and load times of this model are not measured yet** by the project. If you timed something, say what
  and that it was one run on their PC.
