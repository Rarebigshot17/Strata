# Security

## Reporting a vulnerability

Please report anything that should not be public until it is fixed **privately**, through GitHub's private
vulnerability reporting: the repository's **Security** tab, **Report a vulnerability**
([direct link](https://github.com/Rarebigshot17/Strata/security/advisories/new)). Say what you found, how to reproduce
it (the request, the run config keys involved, the Strata commit), and what an attacker gains. We answer there, fix
it, and credit you in the advisory unless you ask us not to.

Ordinary hardening ideas and findings that are safe to discuss in public are welcome as an
[issue](https://github.com/Rarebigshot17/Strata/issues) or as a pull request, which we review like our own code.

Supported: the newest commit of the `glm53-flash` branch. Fixes go there, not into older copies.

This repository is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata). A problem that is in the upstream
code too is worth reporting there as well, through its own Security tab.

## What the server exposes

Strata runs a model on your PC and serves it over HTTP (`serve/server.py`, port 8101). Out of the box it is reachable
from this PC only. The settings named below in quotes are keys of the run config (`strata-glm-<size>.json`), the
others are command-line options or environment variables: [docs/DETAILS.md](docs/DETAILS.md#settings) and
[the run config](docs/INSTALL.md#the-run-config).

- **Where it listens.** `127.0.0.1` by default. `--host 0.0.0.0` (or `"host"` in the run config) opens it to your
  network, and the server then warns when no API key is set.
- **API key.** `"api_key"` in the run config (or `--api-key`, or `STRATA_API_KEY`; several keys separated by
  commas) is required on `/v1/*`, on every endpoint that shows the model's state, requests or answers (`/status`,
  `/metrics`, `/settings`, `/mcp`, `/props`, `/slots`, `/api/requests`, `/config`) and on every `POST`. `/health`
  and the web page's own files stay open. Clients send it as `Authorization: Bearer <key>` or `x-api-key`. It is
  compared in constant time. An empty `--api-key` or `STRATA_API_KEY` stops the server at start; an empty
  `"api_key"` in the run config gets a warning and leaves the server without a key. Set a key before you open the
  server to your network, to a Docker bridge or to Tailscale, or put a tunnel in front of it.
- **Host check (DNS rebinding).** Without an API key the server answers only requests whose `Host` is a name it
  knows: `localhost`, an IP address, the address it listens on and, when it listens beyond this PC, this PC's name
  and `host.docker.internal`. Others get 403. `"allowed_hosts"` (or `STRATA_ALLOWED_HOSTS`) adds names, and `"*"`
  turns the check off. With an API key the key decides.
- **Origin check.** Without an API key, a browser `POST` to `/v1/*` from another web site's page (it carries an
  `Origin` header) gets 403 unless that origin is Strata's own page, `localhost`, an allowed host, or listed in
  `"trusted_origins"` / `"cors_origins"`. Changing settings (`/settings`, and `/config`: the few run config keys the
  page's Settings view may change, never the network, key, MCP or program keys), `/load` and `/unload` and the MCP
  tools are accepted only as JSON, and a request that carries an `Origin` header must come from Strata's own page
  (or a `"trusted_origins"` origin), so a page elsewhere cannot change settings or run tools. A client that is no
  browser sends no `Origin`; it needs the API key when one is set.
- **CORS** is off unless `"cors_origins"` lists origins, and then only for `/v1/*`.
- **Tool calls** are read from the model's own special tokens only, when the tokenizer has GLM's tags as special
  tokens. A call the model copies as plain text, from a tool result, a web page or the user, then stays text and is
  never returned as a call. GLM's tokenizer is expected to have them; the real GGUF has not been checked for it yet,
  and without them the server reads the tags from the text. Its start line says which: a `WARNING` line that names
  the missing tags means the text (requests with tools are still served). A request that declares no tools, or sets
  `tool_choice` to `"none"`, gets no calls. Running a call is the client's choice: an agent that runs shell commands
  decides what it runs.
- **MCP tools** are opt-in: only the servers you list in the run config (`"mcp_servers"`) or in `--mcp-config`, and
  only for requests from Strata's own page that ask for them. API clients get them only when the run config sets
  `"mcp_for_api": true` (the default is false). The tools run with your user's rights, and the model decides when to
  call them. An MCP server's address is shown and logged without its user name, password, query string or
  fragment.
- **Request size and time.** A request body may hold 256 MiB (`STRATA_MAX_BODY_MIB` changes it). Only two bodies
  over 16 MiB are held at once; another one gets 503 with `Retry-After`. A request's line, headers and body must
  keep arriving: 120 s without data ends it. A long streamed answer has no such limit. `response_format` patterns
  run with a time limit and a size limit (`serve/structured.py`).
- **The request monitor** (`/api-monitor`, which keeps the last 100 prompts and answers in memory) is off unless
  `"api_monitor": true` is set.

Strata has not had an outside security audit yet.
