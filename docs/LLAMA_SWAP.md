# Running Strata behind llama-swap

[llama-swap](https://github.com/mostlygeek/llama-swap) puts several model servers behind one address. It starts the
one a request names and stops the others to free the GPUs. Strata can be one of its models.

Nothing on this page has been run with the GLM engine yet. The entry below and the traps come from Strata's server
code; the traps were first found by the upstream project (Niko1221/Strata) with its Qwen engine behind llama-swap.

GLM-5.3-Flash needs every GPU it was set up with and, by setup.py's figures, 112.5 GB (3-bit, counted in the real
file) or about 149 GB (4-bit, derived from its download) of RAM for its experts. So Strata cannot stay loaded beside
another large model: put the others in a llama-swap group that does not run together with Strata, or let swapping
stop it.

## The entry

Set Strata up first without starting it:

```bash
./setup.sh --no-start
```

Setup writes the run config `strata-glm-<size>.json` (`strata-glm-ud-iq3_xxs.json` for the 3-bit model,
`strata-glm-ud-iq4_xs.json` for the 4-bit one; `strata-glm-iq4_xs.json` for a 4-bit install made when setup called it
`IQ4_XS`) and a start script `run-glm-<size>.sh`. The start script fixes the port (8101 unless setup was given
`--port`), so llama-swap calls `serve/server.py` itself, with its own port:

```yaml
healthCheckTimeout: 1800           # seconds: room for the expert warm-up, which is not measured yet (see below)

models:
  strata:
    cmd: >-
      sh -c 'cd /path/to/Strata && exec .venv/bin/python serve/server.py
      --engine strata --config strata-glm-ud-iq3_xxs.json --port ${PORT}'
    env:
      - "STRATA_ALLOWED_HOSTS=my-server"   # the host name your clients use - see "The trap" below
      - "STRATA_REQUEST_LINES=1"           # optional: one line per request in llama-swap's log
    checkEndpoint: /health
    ttl: 1800
```

- `exec` makes the Python server the process llama-swap stops. On SIGTERM the server ends its engine the way Ctrl+C
  does: `STOP` and `QUIT` (the engine saves its expert profile, if the config names one, and exits), up to 120 s
  for that (`STRATA_ENGINE_QUIT_S`), then a kill. How long the GLM engine takes to give its memory back is not
  measured.
- `checkEndpoint: /health`. The server opens its port only after the engine has said `READY`, so `/health` answers
  once the model is loaded. That is not so with `--lazy` or `"lazy_load": true` in the run config: the port then
  opens at once and `/health` answers 200 with `"loaded": false`. Leave lazy loading off behind llama-swap.
- `STRATA_REQUEST_LINES=1` prints one line per request on the server's output (prompt tokens, cached tokens, output
  tokens, prompt time, total time, prefill and decode tokens/s), for a supervisor that only sees that output.
- Parked conversations live in the engine's memory. When llama-swap stops Strata (`ttl`, or a swap), they are gone,
  and the next request of each client reads its whole prompt again.

## How long a start takes

Before it says `READY`, `strata-glm` maps the model, places its layers on the GPUs and then reads the routed experts
into the page cache (the warm-up), so the first request does not wait for the disk. While it does, the engine log
(the run config's `"log"`) gets a line every 2 s:

```text
warming the experts: <read> of <total> GB (<seconds> s, <rate> GB/s)
```

After a reboot the warm-up reads the experts from disk. No start on the real model has been timed yet. For scale
only: a SATA III link carries at most 0.6 GB/s, so reading 112.5 GB takes at least about 3.1 minutes and 149 GB about
4.1 minutes (calculated; a real drive is slower). When the files are still in the page cache from an earlier run, the
warm-up reads them from RAM instead.

So set `healthCheckTimeout` well above that until you have timed a start of your own: the last warm-up line gives the
time and the rate. The server has its own limit as well: the engine must say `READY` within 900 s, counted from its
start or from the latest step it reported (the layers on each GPU, the expert cache, the warm-up); otherwise the
server ends it
(`STRATA_ENGINE_READY_S` changes the 900; 0 waits for ever).

`STRATA_DROP_CACHE_ON_EXIT=1` gives the model files' page cache back when the server stops (off by default). It helps
when llama-swap then starts another large model, and it makes Strata's next start read the experts from disk again.

## The trap: "Host ... is not allowed"

The first request through llama-swap can fail at once with:

```text
403 Host 'my-server:8040' is not allowed (DNS rebinding protection). Reaching Strata under this name on purpose? ...
```

Without an API key, Strata answers only to host names it knows, so that a web page cannot reach it by pointing a DNS
name at 127.0.0.1. A proxy that passes the client's `Host` header on, as llama-swap did in the upstream test, makes
every client that reaches the proxy by name (`http://my-server:8040`) look like such a page. Requests by IP address
and to `localhost` always pass, which is why it can work in a quick test and fail from another machine. Any one of
these fixes it:

- `STRATA_ALLOWED_HOSTS=my-server` in the entry's `env` (comma-separated names; a port is ignored), or
- `"allowed_hosts": ["my-server"]` in `strata-glm-<size>.json`, or
- an `"api_key"` in the run config, which turns the check off. Every client must then send the key.

## Strata's own pages through llama-swap

llama-swap forwards `/upstream/<model>/...` to the model's server unchanged, so Strata's status pages stay reachable.
Calling one also starts the model, without a chat request:

```bash
curl http://my-server:8040/upstream/strata/health
curl http://my-server:8040/upstream/strata/v1/status     # needs the API key when one is set
```

## Without llama-swap

Strata can also share the GPUs with other programs by itself: it can unload the model after a time without requests,
answer 503 while too little VRAM is free, run a command before it loads the model again, and unload on request
(`POST /unload`). See [Settings](DETAILS.md#settings) and [the API](DETAILS.md#the-api).

## Claude Code and other Anthropic-API clients: thinking

Claude Code sends `"output_config": {"effort": ...}` with every request (the upstream project saw this by logging the
requests). GLM-5.3-Flash has no effort levels: its template only turns thinking on or off, so `low`, `medium` and
`high` all mean "think", and `CLAUDE_CODE_EFFORT_LEVEL` does not shorten the thinking here. A request's
`"thinking": {"type": "enabled", "budget_tokens": N}` caps it: the thinking ends after N tokens with a short wrap-up,
and the model answers ([the reasoning budget](DETAILS.md#thinking)). `"reasoning_budget_tokens": N` in the run config
caps the requests that send no budget: a request's own `budget_tokens` wins over it, also a larger one.
