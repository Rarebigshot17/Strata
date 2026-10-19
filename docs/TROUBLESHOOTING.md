# Something went wrong?

Problems with Strata on GLM-5.3-Flash (Linux, NVIDIA GPUs) and what to do about them. The headings and the quoted
lines are the messages that setup, the server and the engine print, so you can search this page for the words you
see. Back to the [README](../README.md#something-went-wrong).

Nothing has run on a GPU or on the real model yet, so this page has no measured times, speeds or memory figures.
Where a fix depends on how long something takes, it says which line of the log to look at instead.

## Reading the log

Strata writes to two places:

- **The server window** (or wherever the output of `./setup.sh`, `run-glm-<size>.sh` or `serve/server.py` goes,
  for example the systemd journal). Its lines start with `[strata]`: what the engine does while it starts, then
  `ready: http://...`, then progress lines while a request runs (`reading the prompt: N tokens, S s so far`,
  `thinking: N of max M tokens, ...`) and a `done: N tokens in S s (...)` line when it ends, refused requests
  (`400 invalid request: ...`, some `503: ...`) and what happened when the engine ended.
- **The engine log**, `strata-glm-<size>.log` in the Strata folder (the run config's `"log"`). It holds the engine's
  own output: `GPU N: layers a-b, ... GB weights and buffers, ... GB expert cache`, `expert cache: ...`,
  `warming the experts: X of Y GB (...)`, the `conversations: ...` lines, and on a failure the engine's last words,
  such as `strata-glm: <reason>` or `glm gpu kernel <name>: <CUDA error>`. Each start begins with a line
  `[strata] <date> <time> engine started: <command>`.

Commands that help (`<size>` is `ud-iq3_xxs` or `iq4_xs`):

```bash
tail -n 80 strata-glm-<size>.log                         # the end of the engine log
grep -n "engine started" strata-glm-<size>.log | tail -3  # where the last starts begin
sudo dmesg | grep -i -E 'killed process|out of memory'   # did Linux end the engine for lack of RAM?
nvidia-smi                                               # the GPUs, their free VRAM and what else uses them
free -g                                                  # RAM: used, and the page cache (buff/cache)
curl -s http://127.0.0.1:8101/health                     # is the server up, is the model loaded
```

Two switches print more, both in the server's environment (for example `STRATA_DEBUG=1 ./run-glm-<size>.sh`):
`STRATA_REQUEST_LINES=1` prints one line per engine request (prompt tokens, reused tokens, times),
`STRATA_DEBUG=1` one line per request with the start of its last message, and the model's whole raw reply after
it. The second one puts what people wrote and what the model answered into the log, so turn it off again when you
are done.

## While installing

**"no NVIDIA GPU found (nvidia-smi did not answer)"** or **"the NVIDIA driver is too old"**.
Install or update the NVIDIA driver (580 or newer; on Ubuntu `sudo ubuntu-drivers install`), restart the PC and run
`./setup.sh` again. Everything already done is kept.

**"GPU N (...) cannot be used: not supported - compute capability X"**.
strata-glm needs compute capability 7.5 or newer (RTX 20 series and newer). This stops setup when `--gpu` or
`--gpus` names such a card. Setup prints the table of your GPUs and the cards it can use; leave the card out of
`--gpus` (for example `--gpus 0,1`), or give neither option: setup then uses every card that can run the engine
and leaves the others out. "none of your GPUs can run the engine" means no card is new enough.

**"The engine is compiled for your GPUs, which needs: ..."**.
Setup installs the C++ compiler and the CUDA toolkit for you on Ubuntu 22.04 and 24.04 (it asks first, and sudo
asks for your password). On other distributions it stops with "the automatic install is only done on
Ubuntu/Debian" or "the CUDA Toolkit can be installed automatically on Ubuntu 22.04 / 24.04 only": install them
with your distribution's packages (Arch: `pacman -S base-devel cuda`) or from
[developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads), then run it again. A CUDA
toolkit of 12.0 or newer that is already installed is used as it is (setup installs CUDA 13.0 only when none
of 12.0 or newer is); setup finds `nvcc` on `PATH`, under `CUDA_PATH` or `CUDA_HOME`, in `/usr/local/cuda*` and in
`/opt/cuda*`.

**The engine does not compile** (`command failed (exit N): cmake`).
The compiler's own error is printed above that line. Setup already tried the build twice. Two mismatches between
a CUDA toolkit and a newer Linux that users of the upstream project reported (not reproduced here):

- `unsupported GNU version`: the toolkit does not take your gcc. Install an older one (for example g++-14) and run
  `CXX=g++-14 ./setup.sh`. `CC`, `CXX` and `CUDAHOSTCXX` name the compilers; setup passes them to CMake, so a
  build folder that was configured before picks them up too.
- `exception specification is incompatible` for `cospi`/`sinpi`/`rsqrt`: the toolkit's math headers clash with
  your glibc (upstream issue #601). Use another toolkit: `STRATA_NVCC=/usr/local/cuda-12.8/bin/nvcc ./setup.sh`
  makes setup use that `nvcc` and no other.

Setup takes the newest toolkit it finds; `STRATA_NVCC=<path to nvcc>` picks another. `./setup.sh --build`
compiles the engine again in the same `build` folder; to build from scratch, delete the `build` folder and run
setup again.

**The download stopped, or a file is wrong.**
Run `./setup.sh` again: downloads continue where they stopped. What setup says and does:

- "cannot list the model's files": the Hugging Face listing did not answer. Check the internet connection (or
  `HF_ENDPOINT`, if you set one to a mirror).
- "could not finish downloading ...": the connection broke too often. Run it again; it resumes.
- "... is N bytes, not the M the repository lists: deleted" and "... has the wrong SHA-256 ...: deleted": the file
  was damaged. Setup deleted it; run it again to download it again.
- "unsloth/GLM-5.3-Flash-GGUF has no <size> files": the repository has none of that quantization (for the 4-bit
  model setup also tries UD-Q4_K_XL). Download the files by hand (`hf download`) and run setup with
  `--gguf-dir <their folder>`.
- "<folder> has no <size> GGUF file" (with `--gguf-dir`): no GGUF file under the folder (sub-folders included)
  has that quantization in its path. "... is short: N of M bytes": a shard is cut off. Delete it and download it
  again.

**"not enough free disk space in <folder>: need ~N GB"**.
The model files are about 120 GB (3-bit) or 170 GB (4-bit), calculated from the model's shapes; setup prints the
real total from the repository's listing before it downloads. Put them on a bigger drive: `--models-dir <folder>`,
or `--data-dir <folder>`, which is remembered for every Strata folder on this PC.

**"<size>'s experts (~N GB) do not fit beside Linux and the server in this PC's N GB of RAM"**.
Every routed expert stays in RAM: about 117 GB at 3-bit and about 162 GB at 4-bit (calculated from the model's
shapes, not measured). Setup subtracts that from the RAM Linux reports and prints what is left beside the experts.
On a PC with 192 GB that is about 75 GB for the 3-bit model and about 30 GB for the 4-bit one (a few GB less, since
Linux reports a little less than the installed RAM). Below 40 GB left setup warns and gives parked conversations
4096 MiB instead of 8192 (`--park-mib 4096`). Below 12 GB it asks "Install it anyway?" and stops unless you answer
`y` (or pass `--model <size> --yes`): the OS would then read experts from the disk while it answers. Choose the
3-bit model, or add RAM.

**`./update.sh` says "git pull did not succeed"**.
A file you changed in the Strata folder stops the pull. `git status` lists them; put your changes aside
(`git stash`) and run `./update.sh` again. Your model files, run configs and the engine are not tracked by git and
are never touched. A copy that was not made with git cannot update itself: update.sh prints where to download the
newest one.

## While starting

**"port 8101 is already in use - is Strata (or another server) already running?"**.
`ss -ltnp | grep 8101` shows which program has it. Close the other Strata, or start this one on another port:
`./setup.sh --port 8102` (for that start only) or `./setup.sh --setup --port 8102` (kept, and written into the run
script). Strata uses 8101 because 3000, 8000 and 8080 are often taken (on napster-ai by Open WebUI, Portainer and a
UPS dashboard).

**"cannot listen on HOST:8101: ... HOST is not an address of this PC"**.
The run config's `"host"` (or `--host`) names an address this PC does not have, at least not yet: a Tailscale
address before Tailscale is up at boot, for example. Use `"host": "0.0.0.0"` with an API key (see
[Open WebUI cannot connect](#open-webui-cannot-connect)).

**"the engine did not start: the engine exited before it was ready (exit code N)"**.
The server prints the engine log's last lines under it, and they say why. The common ones:

- **Exit code 2** and a `usage: strata-glm ...` text: the run config's `"args"` has an option strata-glm does not
  know. Take it out (the usage text lists the options it takes).
- **`strata-glm: CUDA cudaMalloc...: out of memory`**: the GPUs have too little free VRAM for the layers and the
  attention cache. The engine reserves the attention cache for the whole context at start: 11 KB per token
  (calculated, the figure setup uses), so 1.5 GB at 131,072 tokens and 11.8 GB at 1,048,576. Choose a shorter
  context (`./setup.sh --setup --context 262144`, or change the number after `--ctx` in the run config's
  `"args"`), close other programs that hold VRAM (`nvidia-smi` lists them), or remove a `"split"` you set by hand.
  Whatever VRAM is left after the layers goes to the expert cache, less `--vram-reserve` (1 GB per GPU by
  default).
- **A negative exit code** (`-9`) and nothing in the log: Linux ended the engine, most often for lack of RAM.
  `sudo dmesg | grep -i -E 'killed process|out of memory'` confirms it. See
  [The engine died](#the-engine-died) for what to do.
- **`strata-glm: architecture is '...'; this engine runs GLM-5.3-Flash (glm5-next)`** or `GGUF is missing ...`:
  `-m` in `"args"` does not name the first shard of a GLM-5.3-Flash GGUF (`...-00001-of-0000N.gguf`). The server
  checks the same before it starts the engine and stops with
  `[strata] cannot read the model's tokenizer: ... -m must name the model's first shard (...)` when `-m` names a
  file without the tokenizer, such as a later shard.

At the server's own start such a failure stops the server. When the engine is started again later (after it
died, or after an unload), the server tries three times, 15 seconds apart ("the engine did not start (try 1 of
3)"), and answers the waiting request with a 503 if all three fail.

**"<config> refers to missing files"**.
The engine or a model file moved or was deleted. `./setup.sh --setup` repairs the run config; it finds model files
in the data folder (`Strata-GLM-data` next to the Strata folder, or the `--data-dir` you chose).

**A `[strata] config ...` line, and the server stops.**
A key in the run config has a value the server cannot use, for example
`[strata] config sampling.temperature=-1: expected a number >= 0 (0 = greedy)` or
`[strata] config "mcp_for_api" must be true or false`. The line names the key and what it takes. The keys are
described in [the run config](INSTALL.md#the-run-config) and [the settings](DETAILS.md#settings).

**The start takes long: "reading the experts into memory ...", then "warming the experts: X of Y GB (...)"**.
Before it says it is ready, the engine reads every routed expert once (117 GB at 3-bit, 162 GB at 4-bit,
calculated), so the first request does not wait for the disk. After a reboot all of it comes from the disk, and
the PC can be slow meanwhile. The line shows how far it is and its GB/s; the time left is the GB still to read
divided by that rate. Wait for `ready: http://...`. A start soon after the last one is faster when Linux still has
the files in its page cache. `--no-warm` in `"args"` skips the warm-up; the first requests then read the experts
from the disk instead.

**"the engine did not report READY within 900 s"** (with "of its last warm-up progress (N GB read)" once the
warm-up had begun, or "of its last start line (...)" before it).
The server ends a start that makes no progress for 900 seconds. Every step the engine reports in its log starts the
900 seconds again: a new line (the layers on each GPU, the expert cache) or a warm-up line that moves on, so a slow
disk that keeps reading is not ended. A line that only repeats does not count. Mapping the files and placing the
layers on the GPUs print nothing until they are done, so the 900 seconds must cover them. If your PC needs longer
than that for one step, or the warm-up stalls for that long, give it more time in the server's environment:
`STRATA_ENGINE_READY_S=1800 ./run-glm-<size>.sh` (0 waits for ever).

## While running

### The engine died

**"the engine stopped unexpectedly (exit code N)"**, followed by why the server thinks it ended.
The request ends with an error (a 503 that ends in "the next request restarts it", or an error event in a stream
that had started), and the next request starts the engine again. What the server adds:

- "The usual cause is running out of RAM ...": the engine left no reason of its own; most often a signal ended
  it, sent by Linux's out-of-memory killer (check with `dmesg` as above). The experts are read through the page
  cache, which Linux can give back; what it cannot take back is the engine's own buffers, the parked conversations
  (`--park-mib`) and every other program. Close other programs, lower `--park-mib` (setup:
  `./setup.sh --park-mib 4096`), or use the 3-bit model.
- "The engine exited (code N). Its last log line: ...": the engine ended by itself and said why.
- "The engine exited (code N) after it reported: ...": the engine reported an error before it exited.

A 500 with "the engine failed: <reason>" is the same kind of failure reported by the engine itself (a CUDA error,
memory it could not get): the request's state is gone, and if the engine exits after it, the next request starts it
again.

**It is very slow and the disk is busy all the time.**
The page cache cannot hold every expert beside everything else, so the engine reads experts from the disk again
while it answers. `free -g` shows little `buff/cache`. Free RAM as above; the 4-bit model leaves the least.

### The engine went silent

**"the engine said nothing for N s during the request; the server ended the engine"**.
While a request runs, the engine prints a line at least every few seconds (a token, prompt progress, or a
heartbeat while it is busy). When it prints nothing for 300 seconds (longer while it reads the first part of a
long prompt), the server takes it as lost and ends it; the next request starts it again. If that was a slow PC
and not a hang, raise `"engine_silence_s"` in the run config (0 waits for ever). If it happens again on a working
PC, please report it with the end of the engine log.

**"the engine said nothing for N s and used no CPU or disk in that time (frozen ...)"**.
The engine was silent for 90 seconds and did no work in that time, so the server ended it without waiting for the
300 seconds. `STRATA_ENGINE_STALL_S` in the server's environment changes the 90 (0 turns this check off).

**"the engine has said nothing for N s but is still working; it is not ended"**.
Only a note: the engine is busy (CPU or disk), so it is left alone until `"engine_silence_s"`.

### The context is full

**400 "request exceeds the context window: prompt (N tokens) + max tokens (M) exceeds the context (C)"**, or
**"prompt (N tokens) leaves no room to answer in the context (C)"**.
Requests are never cut down to fit. Send a smaller `max_tokens` (the message says how much fits), start a new
conversation, or give the model a longer context: `./setup.sh --setup --context 524288` (up to 1,048,576; each
token takes 11 KB of VRAM, calculated). `"fit_max_tokens": true` in the run config shortens a `max_tokens` that
does not fit instead of refusing the request. A reply that reaches the end of the context while it is written stops
there with finish reason `length`.

### The first answer after a start is slow

Several things make the first requests after a start slower than later ones:

- Parked conversations live in the engine's RAM and end with it. After a start (or a restart after the engine
  died) each client's first request reads its whole conversation again, which for an agent at 512K tokens is the
  longest prompt there is. `reading the prompt: X of Y tokens` in the server window shows the progress.
- The VRAM expert cache starts from the expert profile (`--expert-profile`, which setup writes into the run
  config). The engine saves it only when it leaves its request loop: when the server sends it `QUIT` (`/unload`,
  an idle unload, or any ordinary stop of the server: Ctrl+C, closing its terminal, `kill <server pid>`, `docker
  stop`, `systemctl stop`), or when the server's end of the pipe closes while the engine is idle. It is not saved
  when the engine itself is killed (`kill -9`, the out-of-memory killer, a crash) or when the server is killed
  during a request. After such an end, or on the first start ever, the cache fills while it answers.
- With `--no-warm`, or when Linux gave the page cache to other programs, the experts come from the disk.
- With `"lazy_load": true` or `"idle_unload_s"`, the first request starts the engine, warm-up included.

### A client was cut off

- A request that is not streamed sends nothing until its answer is complete, and reading a long prompt takes time.
  A client or a reverse proxy with a read timeout gives up first. Use streaming (`"stream": true`): the server then
  sends a keep-alive at least every 10 seconds while the request waits for its turn and while it reads the prompt.
  Or raise the client's timeout.
- When a client closes the connection, the server stops the engine on that request, so the others in the queue
  go on.
- 408 "the request body stopped arriving (120 s without data)": the client stopped sending its request. A
  request's headers and body must keep arriving: 120 seconds without data ends it (a 408 for a body; headers that
  stop arriving just close the connection). The answer has no such limit, so a long streamed answer is never cut
  off by the server.

### A 400 that names a field

**400 with a message such as "temperature: a number >= 0 (0: greedy)"**, "top_p: a number from 0 to 1",
"repetition_penalty: a number > 0 (1: off)", "seed: an integer up to ...", "top_k: an integer" or
"temperature: a finite number".
A sampling value is out of range or of the wrong type: NaN and Infinity, a repetition penalty of 0 or less,
temperature below 0, top_p or min_p outside 0 to 1, or an integer too large for the engine. The message names the
field. The shared Chat settings (`POST /settings`, a 400 that names the field) and the run config's `"sampling"`
block (a `[strata] config sampling.<field>=...` line, and the server stops) are checked the same way, each with
its own wording; a shared temperature goes up to 2. "unknown reasoning effort ...: use none, low, medium or high"
is the same for `reasoning_effort`. The server window prints each refused request as
`[strata] 400 invalid request: ...`.

### A 503 or a 413 about the request's size

**503 "too many large requests at once (over 16 MiB each); try again shortly"**, with `Retry-After: 30`.
A request body over 16 MiB takes one of two places while its request runs; a third one is refused until a place is
free. The window says `[strata] 503: 2 requests with bodies over 16 MiB are already in the server`. Retry after
the time the header gives; a client that honours `Retry-After` does this by itself.

**413 "the request body is larger than 256 MiB"**: the most a request may hold. `STRATA_MAX_BODY_MIB` in the
server's environment changes it.

**503 "the engine is starting ...; try again shortly"**: the engine is (re)starting and has not said it is ready.

### MCP tools are missing for an API client

**403 "MCP tools (strata_mcp) run only for Strata's own page"**.
The MCP servers in the run config (`"mcp_servers"`, or `--mcp-config`) run their tools with your rights on this
PC, so by default only Strata's own web page may use them. To let API clients (an agent, a bot) use them too, set
`"mcp_for_api": true` in the run config and restart. The client must also ask for them with `"strata_mcp": true` in
a `/v1/chat/completions` request (the only route that offers them); without it a client gets only the tools it
declares itself. A browser page of another site gets a 403
"MCP tools can be used only from Strata's own page (or an origin in the config's trusted_origins)".

### A tool call came back as text

The server turns the model's output into a tool call only when:

- the request declares tools, and `tool_choice` is not `"none"`;
- the call is written with GLM's own tag tokens. When the tokenizer has all of GLM's tags (`<think>`,
  `<tool_call>`, `<arg_key>`, ...) as special tokens, a call that the model copies as plain text from a tool result,
  a web page or the user stays text and is never run. That matters for the automation agent: a command inside a
  fetched page is not run because the model repeated it.

A call written inside the thinking is run only when the turn shows it was meant: nothing but more calls follows it
before the thinking or the turn ends, and it is not quoted (in code or on a "> " line). The window then says
`tool calls inside the thinking: N read as calls, M kept as reasoning`. `"tool_call_recovery": true` in the run
config also accepts a declared tool's call written as JSON inside `<tool_call>`.

## Open WebUI cannot connect

Open WebUI runs in Docker, so `127.0.0.1` inside its container is the container itself, not the PC.

1. **The server must listen beyond this PC.** Set `"host": "0.0.0.0"` and an `"api_key"` in the run config, or
   start it once with `./setup.sh --host 0.0.0.0 --api-key <key>` (both are then saved for this model). The server
   then prints `from other devices: http://<IP>:8101/` for each address it has.
2. **The address.** In Open WebUI add an OpenAI connection with:
   - `http://172.17.0.1:8101/v1` when its container is on Docker's default bridge (172.17.0.1 is the PC's address
     there; on a network of your own, `docker network inspect <network>` shows its gateway);
   - `http://host.docker.internal:8101/v1` when the container was started with
     `--add-host=host.docker.internal:host-gateway`;
   - on napster-ai `http://100.126.222.90:8101/v1` over Tailscale works too, also for the agents and the Discord bot
     on other machines.
3. **The API key.** Put the server's key in the connection's key field. A wrong or missing key is a 401
   "missing or wrong API key". Clients send it as `Authorization: Bearer <key>` or `x-api-key: <key>`.
4. **Allowed hosts.** Without an API key the server answers only to the names it knows (protection against DNS
   rebinding): IP addresses, `localhost`, and while it listens beyond this PC also this PC's name and
   `host.docker.internal`. Another name gets a 403
   "Host '...' is not allowed (DNS rebinding protection)". Add it to `"allowed_hosts"` in the run config, or set an
   API key, which turns this check off.
5. **A firewall.** If nothing reaches the server at all (`curl` from inside the container times out), a firewall
   on the PC (ufw, for example) may drop traffic from the Docker network to port 8101. Allow the port from that
   network.

A request from inside the container answers when the address and the firewall are right, for example
`docker exec <container> curl -s http://172.17.0.1:8101/health` (if its image has `curl`).

## Opening an issue

Search [the open issues](https://github.com/Rarebigshot17/Strata/issues) first, then open one with:

- what you did, what you expected and what happened, with the exact message;
- the commit (`git rev-parse --short HEAD`) and the engine version (`build_info` in
  `curl -s http://127.0.0.1:8101/props` while the engine runs; add `-H "Authorization: Bearer <key>"` when the
  server has a key);
- `./setup.sh --check`, `nvidia-smi` and `free -g`;
- the run config `strata-glm-<size>.json`, **with the `"api_key"` removed**;
- the server window's output around the problem, and the end of `strata-glm-<size>.log` from the last
  `engine started` line on;
- for a crash, the `dmesg` lines above.

Leave out conversations and API keys. `STRATA_DEBUG=1` output contains parts of what was asked, so read it before
you attach it.
