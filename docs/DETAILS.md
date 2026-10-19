# Strata - the details

This is the reference for using Strata's server: every route, the request fields each API honours, the settings,
how several clients share the engine, monitoring, security and limits. How the engine works is in
[HOW_IT_WORKS.md](HOW_IT_WORKS.md); installing is in [INSTALL.md](INSTALL.md).

**What has been checked so far.** Nothing has run on a GPU or on the real GLM-5.3-Flash model yet, so this page has
no measured speeds, load times or memory figures. Figures marked "calculated" come from the model's shapes. What is
checked: the server's behaviour by its Python tests (a scripted engine stands in for the model), the server end to
end on the CPU reference engine with a small random GLM model (`tools/glm53_server_check.sh`), conversation parking
(`tools/glm53_park_check.py`) and the CPU reference against llama.cpp (`tools/glm53_selftest.sh`). Where something
depends on the real model's files (its chat template, its token types), the page says so.

Contents: [The API](#the-api) · [Settings](#settings) ·
[Several clients on one engine](#several-clients-on-one-engine) · [Monitoring](#monitoring) · [Security](#security) ·
[Limits](#limits)

## The API

The server listens on `http://127.0.0.1:8101` (setup's `--port` changes the port; `"host"` in the run config or
`--host 0.0.0.0` lets other machines in, see [Security](#security)).

| API | Base URL to give a client |
| --- | --- |
| OpenAI (Chat Completions, Completions, Responses) | `http://127.0.0.1:8101/v1` |
| Anthropic (Messages) | `http://127.0.0.1:8101` (the client adds `/v1/messages`) |

Any model name is accepted. The answer carries the model's own name (the run config's `"model_name"`, which setup
writes as `glm-5.3-flash`; without it the GGUF's `general.name`, else `glm-5.3-flash`), or an alias when the request
names one (see `"aliases"` under [Settings](#settings)).

```bash
curl http://127.0.0.1:8101/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "glm-5.3-flash", "messages": [{"role": "user", "content": "Write a haiku about GPUs."}],
  "max_tokens": 2048}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8101/v1", api_key="none")   # or the server's API key
r = client.chat.completions.create(model="glm-5.3-flash", messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

### Routes

"Key" means the route needs the API key when the server has one (`Authorization: Bearer <key>` or
`x-api-key: <key>`). Without a key, every request first passes the Host check ([Security](#security)). A path's
query string and trailing slash are ignored, so `/v1/messages?beta=true` is `/v1/messages`.

| Method | Path | What it does | Key |
| --- | --- | --- | --- |
| POST | `/v1/chat/completions` (also `/chat/completions`) | OpenAI Chat Completions, streamed or not | yes |
| POST | `/v1/completions` (also `/completions`) | OpenAI's plain completion: the prompt as given, no chat template ([below](#plain-completions)) | yes |
| POST | `/v1/messages` | Anthropic Messages, streamed or not | yes |
| POST | `/v1/messages/count_tokens` | Anthropic's token count, `{"input_tokens": N}`: the prompt is rendered and tokenized, the model does not run | yes |
| POST | `/v1/responses` | OpenAI Responses, stateless ([below](#the-responses-api-and-codex-cli)) | yes |
| POST | `/v1/responses/<id>...` | 404: the server keeps no responses | yes |
| GET | `/v1/models` (also `/models`) | The model and its aliases, with `meta.n_ctx` and `status.value` (`loaded`, `unloaded` or `loading`) | yes |
| GET | `/health` (also `/api/health`) | `{"status": "ok", "model", "max_context", "loaded", "api_key" (whether one is needed), "images": false, "service": "strata"}` | no |
| GET | `/status` | What the model is doing now: busy, queued, phase, prompt tokens, tokens so far, the end of the running answer | yes |
| GET | `/v1/status` | What the server is: model, context, loaded, the APIs, structured-output support, the last request's timings, GPU and RAM | yes |
| GET | `/props` (also `?model=<name>`) | llama-server's properties: the generation defaults, `n_ctx`, the chat template and `chat_template_caps`, `total_slots: 1` | yes |
| GET | `/slots` | llama-server's slot list: one slot with `n_ctx`, `is_processing` and `n_prompt_tokens` (`[]` while unloaded) | yes |
| POST | `/slots/<id>?action=save\|restore` | 501: slot save and restore do not exist here; the engine parks conversations itself | yes |
| GET | `/metrics` | The Monitor tab's JSON, or the Prometheus text format ([Monitoring](#monitoring)) | yes |
| GET | `/metrics/prometheus` | Prometheus text with llama-server's metric names | yes |
| GET | `/mcp` | The MCP servers, their state and their tools | yes |
| GET, POST | `/settings` | The shared settings ([Settings](#shared-settings-for-other-clients)); POST only from Strata's own page | yes |
| GET, POST | `/config` | The Settings view of the run config ([Settings](#the-settings-view)); POST only from Strata's own page | yes |
| POST | `/load`, `/unload` | Load the model now; unload it between requests ([below](#loading-and-unloading)) | yes, own page |
| POST | `/v1/load`, `/v1/unload` | The same as JSON with the model's status | yes, own page |
| GET | `/` | The web app: Chat, Monitor and About | no |
| GET | `/web/<file>`, `/fonts/<file>` | The web app's styles, script, icons and font | no |
| GET | `/api-monitor` | The request monitor page, when `"api_monitor": true` (else 404) | no (its data needs the key) |
| GET | `/api/requests` (also `?id=<id>`) | The request monitor's records, when it is on | yes |
| OPTIONS | any | A CORS preflight: 204, with CORS headers only for the `cors_origins` | no |

"Own page" means the request must be `Content-Type: application/json` and carry no `Origin` of another site (Strata's
own page, an origin in `"trusted_origins"`, or no `Origin` at all, as curl sends). Any other path is a 404.

### Request fields

What each API honours, beyond its messages. The OpenAI and Anthropic shapes are converted to the same chat template
messages, so most fields behave the same on all routes; the differences are said.

Messages:

- OpenAI: `system`, `developer` (read as system), `user`, `assistant` (with `tool_calls` and `reasoning_content`)
  and `tool` messages; content as a string or text parts. A system message that is not the first becomes a user
  message in place, so the start of the prompt stays the same from turn to turn.
- Anthropic: `system` (a string or text blocks), and `text`, `thinking`, `tool_use` and `tool_result` blocks.
  Claude Code's billing header at the start of its system prompt changes (`cch=...` on every request, the 4th part
  of `cc_version=` on every session); the server pins both stamps to `f`s, as llama.cpp does, so the prompt start
  stays the same.
- GLM-5.3-Flash reads text only: a picture in a message is a 400. A picture inside a tool's result becomes the note
  `[image omitted: the model reads text only]`, so a conversation with one keeps working.
- A tool list that comes back with the same tools in another order is put back in the order first seen, so the
  prompt start (the template writes the tools first) does not change.

#### Sampling

The same fields on every route (the request is also the sampling settings). A field the request leaves out (or sets
to `null`) takes the default: the shared settings, then the run config's `"sampling"` block, then GLM's
recommendation, temperature 1.0 and top_p 0.95 ([Settings](#sampling-defaults)).

| Field | Valid values | Meaning |
| --- | --- | --- |
| `temperature` | a number >= 0 | 0 is greedy (always the most likely token) |
| `top_p` | 0 to 1 | 1 is off |
| `top_k` | a whole number up to 2,147,483,647 | 0 (or more than the vocabulary) is off; a negative value is not passed on, so the engine's default (off) applies |
| `min_p` | 0 to 1 | 0 is off |
| `repetition_penalty` (also `repeat_penalty`) | a number > 0 | 1 is off |
| `frequency_penalty`, `presence_penalty` | finite numbers | 0 is off |
| `penalty_last_n` (also `repeat_last_n`) | a whole number up to 2,147,483,647 | the window of the three penalties; 64 when a penalty is on and no window is given |
| `seed` | a whole number from 1 to 2^64-1 | the same seed and request give the same draw; unset, 0 or negative is random |
| `logit_bias` | token id -> -100..100 | a per-request token bias, checked before the answer starts on `/v1/chat/completions` and `/v1/completions` ([LOGIT_BIAS.md](LOGIT_BIAS.md)) |

A value of the wrong type or out of range is a 400 that names the field (`top_p: a number from 0 to 1`). That
includes NaN and Infinity (which JSON parsers in Python accept), `true` for a number, a number with a decimal point
(even `40.0`) for a whole number, and a whole number too large for the engine. A number sent as a string (`"0.7"`)
is read as that number. The engine applies the bias, then the penalties, then top-k, top-p and min-p (both at
temperature 1), and draws at the temperature.

#### Output length and the context

- OpenAI: `max_completion_tokens` or `max_tokens`; Anthropic: `max_tokens`; Responses: `max_output_tokens` (a
  positive whole number). The thinking counts as output.
- Unset, 0 or -1 means "as long as the context allows": the rest of the context after the prompt.
  `/v1/completions` is the exception: unset or 0 means 256 tokens there (-1 still means the rest).
- A request whose prompt plus its output cap does not fit the context (less 8 tokens of margin) is refused with a 400
  that says how much would fit; requests are never cut. With `"fit_max_tokens": true` in the run config (or
  `--fit-max-tokens`) the cap is shortened to the room left instead. A prompt that leaves no room at all is always a
  400.
- The shared settings' `max_tokens` fills in for a request that sets none.

#### Stop strings

OpenAI's `stop` (a string or a list of up to 4) or Anthropic's `stop_sequences` (a list). The answer is cut before the
first one and the reply ends there (`finish_reason` `"stop"`; Anthropic `stop_reason` `"stop_sequence"` with
`stop_sequence` set). Only the answer's text is matched: not the thinking and not a tool call. Text that could be the
start of a stop string is held back until it is clear, so a stop string split across tokens is still found and
never sent. A bad value is a 400 before anything runs.

#### Thinking

GLM-5.3-Flash thinks before it answers by default. Its chat template has no effort levels: thinking is on or off.
So `low`, `medium` and `high` (also `xhigh`, `max`, `maximum`) mean "on", and `none`, `off`, `minimal`, `disabled`
and `false` mean "off". Any other value is a 400.

| API | Thinking off | Thinking on |
| --- | --- | --- |
| OpenAI chat | `"reasoning_effort": "none"`, `"thinking": {"type": "disabled"}` (Z.ai's form), or `"chat_template_kwargs": {"enable_thinking": false}` | `"reasoning_effort": "high"` (or `low`, `medium`), `"reasoning": {"effort": ...}`, `"thinking": {"type": "enabled"}` |
| Anthropic | `"thinking": {"type": "disabled"}`, or `"output_config": {"effort": "none"}` | `"output_config": {"effort": ...}`, or `"thinking": {"type": "enabled", "budget_tokens": N}` |
| Responses | `"reasoning": {"effort": "none"}` | `"reasoning": {"effort": ...}` |

- A request that says nothing gets the shared settings' level, then the run config's `"reasoning_effort"`, then the
  template's default (on). An Anthropic request that says nothing thinks too, unless the run config has
  `"anthropic_thinking": "on_request"` (Anthropic's own rule: thinking only when asked).
- Anthropic's `budget_tokens` only turns thinking on here (it picks an effort level, which GLM's template does not
  have). It does not cap the thinking: `reasoning_budget_tokens` does (below).
- The thinking comes back as `reasoning_content` (OpenAI, in the stream's deltas and the message), as `thinking`
  blocks (Anthropic, with an empty `signature`) or as a `reasoning` output item (Responses). Send it back with the
  conversation as the API does. GLM's template keeps the thinking of the turns after the last user message and drops
  the older ones, unless `"chat_template_kwargs": {"clear_thinking": false}` (this is the behaviour of the template
  Strata falls back to, `serve/chat_template.jinja`; the server uses the template inside the model's GGUF, which is
  still to be checked on the real file).

**The reasoning budget.** `"reasoning_budget_tokens": N` in a request (chat, Anthropic or Responses), in the shared
settings or in the run config caps the thinking at N tokens; a request's own value wins, and 0 means no budget. When
the thinking reaches N, the server ends it at a clean point (no tag or character cut in half) with a short wrap-up and
`</think>`:

```text
I have thought about this long enough; time to give my answer.</think>
```

and the model answers from there. The wrap-up is part of the thinking the client sees and counts as output tokens.
The engine stops its pass exactly where the budget ends, so the wrap-up continues the engine's state and nothing is
read again. A budget that would leave no room for the answer is cut down: with an output cap M, the thinking is
capped at M minus a reserve of M/4 (at least 512 tokens, at most M/2), and the server window says so. A reply that
reaches its output cap while still thinking has no answer (`finish_reason` `"length"`); the server window then
suggests a budget.

Three guards in the run config; the first is on by default, the other two are opt-in:

- `"repeat_stop_tokens": N` (default 256; 0 is off): a reply that repeats one token N times in a row is ended there as
  `"length"`.
- `"reasoning_loop_recovery": "stop"` (or `true`): every 512 output tokens the thinking is checked; when at least 25 %
  of its last 2,000 words lie in 12-word passages seen three times, or its last 2,048 characters repeat with a period
  of at most 64, the reply is ended there as `"length"`.
- `"reasoning_close_retry": true`: a reply that ends on its end-of-turn token still inside its thinking, with no answer
  and no tool call, is continued once with the thinking closed (`</think>`).

If the engine reports non-finite logits (NaN), it drops its cached state. A request that had sent nothing yet is sent
once more without the client seeing it; one that had started ends as `"length"`; a second failure is a 503.

#### Tools and tool_choice

Tools are given in each API's own shape: OpenAI `{"type": "function", "function": {"name", "description",
"parameters"}}`, Anthropic `{"name", "description", "input_schema"}`, Responses `function`, `namespace` and `custom`
tools ([below](#the-responses-api-and-codex-cli)). A tool without a name, or a schema that is not an object, is a 400
that names the tool. Calls come back in the API's shape: OpenAI `tool_calls` (`finish_reason` `"tool_calls"`),
Anthropic `tool_use` blocks (`stop_reason` `"tool_use"`), Responses `function_call` items. The model may write several
calls in one turn.

`tool_choice`:

| Value | What happens |
| --- | --- |
| `"auto"` (or absent); Anthropic `{"type": "auto"}` | The model decides. |
| `"none"`; Anthropic `{"type": "none"}` | The tools are not offered, and nothing in the reply becomes a call. |
| `"required"`; Anthropic `{"type": "any"}` | The server opens the call itself (`<tool_call>`), so the model must go on with one and names the tool. With exactly one tool this is the named case. |
| `{"type": "function", "function": {"name": N}}` (also the flat `{"type": "function", "name": N}`); Anthropic `{"type": "tool", "name": N}` | The server writes `<tool_call>N`, and the model writes the arguments. |

There is no grammar: the server writes the call's opening and the model continues it. Without thinking, the opening
ends the prompt. With thinking, the server writes it the moment the thinking ends (the engine's pass stops at
`</think>`, so the opening continues its state); a call the model starts by itself is left alone. A choice the server
cannot honour (an unknown shape, a name that is not one of the tools, `"required"` with no tools) is logged and acts as
`"auto"`. A forced choice together with MCP tools is a 400.

#### How tool calls are recognised

GLM writes a call as `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`, and its thinking
ends with `</think>`. The server reads these tags only from the model's own special tokens, when the tokenizer has
every one of them as a special token (`output_tags` in `serve/server.py`; `ORDINARY_LT` and `OUTPUT_TAGS` in
`serve/frontend.py`). The same text spelled with ordinary tokens - a call the model copies from a tool result, a web
page or the user - stays text and is never run, and an ordinary `</think>` does not end the thinking. This matters for
an agent that runs shell commands: a call written into a fetched page does not run because the model repeats its
text; only a call the model writes itself, with its own tokens, is a call.
GLM's GGUF files are expected to list these tags as special tokens; on the real file this is still to be checked
([GLM53_SERVER_PORT.md](GLM53_SERVER_PORT.md), "Known risks"). Should one of them not be a special token, the
server reads all of them from the text instead, and that protection is gone.

Further rules:

- With no tools declared, or `tool_choice` `"none"`, nothing becomes a call: the tags are text.
- Only a declared tool's name followed by `<arg_key>` or `</tool_call>` is a call; anything else after `<tool_call>`
  is prose.
- A `<tool_call>` inside a code fence or inline code is text. Inside the thinking, one on a `> ` quote line is text
  too; in the answer a `> ` line does not stop a call.
- A call written inside the thinking is held. It becomes a call only when `</think>` or the end of the turn follows it
  (only whitespace or more calls between); any other text after it means it was quoted, and it stays thinking. A turn
  cut off by its output cap keeps such calls as thinking.
- A call cut off by the end of the reply is not returned as a call to run. In a stream its name and part of its
  arguments may already have gone out; its JSON is closed so it parses, and the finish reason is not `"tool_calls"`.
- Arguments: a parameter the schema types as `string` is the text as written; any other is read as JSON (the text
  itself when it is not JSON).
- `"tool_call_recovery": true` in the run config (off by default) also reads a call written as a JSON object
  (`{"name": ..., "arguments": {...}}`, bare or in a ```` ```json ```` fence) inside `<tool_call>`, for a declared tool.

#### Structured output (response_format)

`/v1/chat/completions` takes `response_format`; `/v1/responses` takes `text.format` with the same types:

- `{"type": "json_object"}`: one JSON object.
- `{"type": "json_schema", "json_schema": {"name": "...", "strict": true, "schema": {...}}}`: one JSON object that
  matches the schema. The schema's root must accept only objects (`"type": "object"`, or `anyOf`/`oneOf` of object
  schemas, an `allOf` with one, or a local `$ref` to one); only local `#...` references are allowed.

There is no constrained decoding. The server adds an instruction with the schema to the system message, the model
writes one answer, and the server checks it before anything reaches the client: valid JSON (a code fence around it
is removed, no duplicate keys, no NaN or Infinity), the schema, and a reply that ended normally (not cut by its output
cap). The answer is returned as compact JSON. A failed check is a 502 with `error.code` `"structured_output_failed"`;
a bad schema in the request is a 400. There is no hidden retry. A streamed structured answer is held back (keep-alive
comments meanwhile) and sent once checked.

The schema is checked with the Python package `jsonschema` when it is installed (`python -m pip install
"jsonschema>=4.23,<5"`; setup does not install it). Without it, only "one JSON object" is checked, and the server
says so once. A schema's `pattern` and `patternProperties` run with the `regex` package under two limits
(`serve/structured.py`): together they may take 2 s on one answer, and a pattern may have at most 10,000 parts with
its repeats written out (`a{100000}` has 100,000). A pattern over either limit fails the answer's check with a 502;
the size limit is only checked then, after the model has written the answer.

On `/v1/chat/completions`, a JSON format together with tools or MCP is a 400. On `/v1/responses` it may come with
tools: the schema applies to the final answer, and a turn that calls a tool is passed on as written.

#### Streaming

`"stream": true` sends everything as it is made: the thinking, the answer and tool calls (a call's name first, then
its arguments piece by piece: a string parameter character by character, any other value whole once complete).

- OpenAI routes: Server-Sent Events `data: {...}` chunks, ending with `data: [DONE]`. Anthropic: its typed events
  (`message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`,
  `message_stop`). Responses: its typed events ([below](#the-responses-api-and-codex-cli)).
- While the request waits for its turn, while the prompt is read and while a tool argument is held back, the stream
  sends SSE comments (`: keep-alive`), at least every 10 s, so clients with an idle timeout do not give up.
- The last OpenAI chunk carries the finish reason, `usage` and `timings`. With
  `"stream_options": {"include_usage": true}` (chat and completions) the usage moves to one more chunk with empty
  `choices`, sent after it and before `[DONE]`; only that chunk carries the usage, as OpenAI and llama.cpp send it.
- `"return_progress": true` (llama.cpp's flag, chat route) puts the prompt progress on the stream: chunks with an
  empty delta and `prompt_progress` `{"total", "cache", "processed", "time_ms"}`, with llama.cpp's meanings.
- An error after the stream has started arrives as a last `data: {"error": ...}` line and `[DONE]` (OpenAI), an
  `error` event (Anthropic) or `response.failed` (Responses).
- Closing the connection stops the model: the server checks every 0.5 s whether the client is still there.
- Streams carry `X-Accel-Buffering: no`, so nginx-style proxies pass each event on at once.

#### chat_template_kwargs

`"chat_template_kwargs": {...}` (chat and Anthropic routes; the vLLM and llama.cpp convention) passes keys to the
chat template. `enable_thinking` and `reasoning_effort` are read as the thinking switch above; every other key
reaches the template as it is, for example `"clear_thinking": false` (keep the thinking of earlier turns), except the
names the server sets itself (`messages`, `tools`, `add_generation_prompt`, `raise_exception`, `strftime_now`). What
a key does depends on the template in the model's GGUF.

### Usage numbers

Every answer says how many tokens it took:

| API | Fields |
| --- | --- |
| OpenAI chat and completions | `usage`: `prompt_tokens`, `completion_tokens`, `total_tokens`, `prompt_tokens_details.cached_tokens` (prompt tokens the engine did not read again: a resumed conversation or checkpoint); `timings` with llama.cpp's names (`cache_n`, `prompt_n`, `prompt_ms`, `prompt_per_second`, `predicted_n`, `predicted_ms`, `predicted_per_second`, ...) from the engine's own clock |
| Anthropic | `message_start` has the whole prompt as `input_tokens`; the final `message_delta` (and a non-streamed answer) has `input_tokens` without the reused part, `cache_read_input_tokens` (the reused part) and `output_tokens` |
| Responses | `usage`: `input_tokens`, `input_tokens_details.cached_tokens`, `output_tokens`, `output_tokens_details.reasoning_tokens` (tokens written while thinking), `total_tokens` |

`completion_tokens` / `output_tokens` include the thinking, a budget's wrap-up and a forced call's opening written
after the thinking. After MCP rounds, `prompt_tokens` is the last round's prompt and the output counts every round.

### Plain completions

`POST /v1/completions` takes `prompt` as one string (or a list holding one string) and returns the model's text as it
is. No chat template is applied, and nothing is read as thinking or as a tool call. Special tokens written in the
prompt are those tokens, so a client can write GLM's chat format itself. `n` must be 1. Stop strings, the sampling
fields, `logit_bias`, `stream` and `stream_options` work as on the chat route; without `max_tokens` it writes at most
256 tokens.

### Errors

| Status | When |
| --- | --- |
| 400 | A malformed request: bad JSON or the wrong shape, a field out of range, a prompt that does not fit the context, a picture, an unknown effort, a bad schema; the template's own refusal |
| 401 | The API key is missing or wrong |
| 403 | A Host name the server does not answer to, a web page of another site, `strata_mcp` from an API client without `"mcp_for_api"`, or an own-page route from another page |
| 404 | An unknown path or model name (`/props?model=`, `/v1/load`), a stored response, `/config` without a run config, the request monitor while it is off |
| 408 | The request body stopped arriving (120 s without data) |
| 409 | `/unload` while a request runs or waits (`{"status": "busy"}`), `/v1/load` and `/v1/unload` likewise (`model_busy`) |
| 413 | The body is over the size limit |
| 415 | Not `application/json` where JSON is required |
| 500 | The engine failed (`ERR internal`), or the run config cannot be read or written |
| 501 | `POST /slots/...` |
| 502 | A structured answer failed its check |
| 503 | The engine is starting, died, or cannot be started (too little free VRAM, a failed `before_load`), its output went non-finite twice, or too many large bodies are in the server; often with `Retry-After` |

Errors are `{"error": {"message", ...}}`, most with a `type` as well; on `/v1/responses` they have the Responses form
`{"error": {"message", "type", "param", "code"}}`. An engine that died is started again by the next request.

### The Responses API and Codex CLI

`POST /v1/responses` speaks OpenAI's Responses API, which Codex CLI uses. It runs on the chat path, so the thinking
switch, the reasoning budget, the sampling fields, stop strings, the conversation cache and the security checks are
the same.

It is **stateless**: nothing is stored, so the client sends the whole conversation in `input` every time (Codex does,
with `store: false`). `previous_response_id`, `conversation`, `background`, stored `prompt` templates, `item_reference`
items and the retrieve, delete and cancel routes are refused with an error that says so.

| Request | What Strata does |
| --- | --- |
| `input` | A string, or items: `message` (`user`, `assistant`, `system`, `developer`; text parts), `reasoning`, `function_call`, `function_call_output`, `custom_tool_call(_output)`, `additional_tools` |
| `instructions` | The system message; leading `developer` messages join it, later ones become user messages in place |
| `tools` | `function` tools; `namespace` tools (the model sees `namespace.name`, calls come back with `namespace` and `name`); `custom` tools (one free-form `input` string). Hosted tools (`web_search`, `file_search`, ...) are left out, with one line in the server window |
| `tool_choice` | `"none"` hides the tools; `"required"` and a named `function` or `custom` tool open the call as on the chat route |
| `reasoning.effort` | Thinking on or off, as above |
| `max_output_tokens` | The output cap; running out ends the response `incomplete` (`max_output_tokens`) |
| `text.format` | `json_object` and `json_schema`, as [above](#structured-output-response_format) |
| `include: ["reasoning.encrypted_content"]` | The reasoning item also carries `encrypted_content`: the thinking in base64, not encrypted (the client already has the text) |

The thinking comes back as a `reasoning` item with `reasoning_text` content; `summary` is empty, as GLM writes no
summaries. Send the reasoning items back: their text goes back into the prompt, so it matches what the model wrote
and the conversation cache is reused. Streamed events come in the official order: `response.created`,
`response.in_progress`, per item `response.output_item.added`, its deltas and `...done` events and
`response.output_item.done`, and at the end `response.completed`, `response.incomplete` or `response.failed`. While a
long prompt is read, a `response.in_progress` event goes out every 15 s, because Codex's idle timer does not count SSE
comments.

Two opt-in helpers for Codex in the run config, both off by default:

- `"codex_thread_titles": true`: Codex's thread-title request (marked in `client_metadata["x-codex-turn-metadata"]`)
  is answered without the engine, with the user's line cut to 36 characters as the title, so the conversation the
  engine holds is not interrupted.
- `"codex_compaction_cache": true`: Codex's compaction request (sent without tools) is rendered with the tools of the
  same Codex conversation's last prompt, so its prompt starts as the one the engine holds and is not read again from
  the start. Only the prompt changes; the response keeps the request's own (empty) tool list.

**Codex CLI.** In `~/.codex/config.toml`:

```toml
model = "glm-5.3-flash"                 # any name; Strata answers with its model
model_provider = "strata"
model_context_window = 131072           # the context you chose in setup: Codex compacts before it gets there
show_raw_agent_reasoning = true         # show the model's thinking (it writes no summaries)
# model_reasoning_effort = "minimal"    # thinking off; low, medium or high all mean on (GLM has no levels)

[model_providers.strata]
name = "Strata (local)"
base_url = "http://127.0.0.1:8101/v1"
wire_api = "responses"
stream_idle_timeout_ms = 600000         # room for reading a long first prompt (prompt speed is not measured yet)
# env_key = "STRATA_API_KEY"            # only if the server has an API key: the variable that holds it
```

Then run `codex` as usual. Codex may warn that it has no metadata for the model name; that is expected for a local
model. Codex with Strata's GLM engine has not been tried yet.

### Other clients

- **Open WebUI** and other chat apps: an OpenAI connection with base URL `http://<address>:8101/v1` and the API key
  (any text when the server has none). From a Docker container on the same machine the address is the host as the
  container sees it (`172.17.0.1` on Docker's default bridge), or the machine's Tailscale address; both need the
  server to listen beyond 127.0.0.1 ([Security](#security)).
- **Claude Code**: `ANTHROPIC_BASE_URL=http://127.0.0.1:8101`, `ANTHROPIC_AUTH_TOKEN` set to the API key (any text
  without one), and `ANTHROPIC_MODEL` set to a model name Claude Code accepts; Strata answers whatever the name. The
  upstream project saw it send an effort (`output_config.effort`) with every request; any level but `none` means
  "think" here. `"reasoning_budget_tokens"` in the run config caps the thinking.
- **OpenCode** and other OpenAI-compatible agents: base URL `http://127.0.0.1:8101/v1`; set the client's context
  limit to the context chosen in setup, and keep its output limit well under it (or set `"fit_max_tokens": true`).
- **Zed's llama.cpp provider** reads `/props` and offers tools from `chat_template_caps`. Its generic OpenAI provider
  does not; add `"capabilities": {"tools": true, "chat_completions": true}` to the model in Zed's settings.

### The MCP tool loop

Strata can give the model tools from [MCP](https://modelcontextprotocol.io) servers and run them itself
(`serve/mcp.py`). List the servers in the run config under `"mcp_servers"` (Claude Desktop's `"mcpServers"` block can
be pasted as it is), or in a separate file given with `--mcp-config <file>` (`{"mcpServers": {...}}`; a name in both
takes the file's entry):

```json
"mcp_servers": {
  "files": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/home/me/notes"]},
  "search": {"url": "http://127.0.0.1:3100/mcp", "headers": {"Authorization": "Bearer ..."}}
},
"mcp": {"timeout_s": 60, "max_result_chars": 20000, "max_rounds": 8, "start_timeout_s": 120},
"mcp_for_api": false
```

- **A program** (`command`, `args`, optional `env` and `cwd`) is started by Strata and spoken to over its stdin and
  stdout. **An address** (`url`, optional `headers`) uses MCP's Streamable HTTP transport; the older SSE transport
  (`"type": "sse"`) is refused at start. `"disabled": true` leaves an entry out. A wrong entry stops the start.
- The servers start with Strata, in the background. The server window says how many tools each one offers, or why it
  did not start; its tools are then left out and the rest works. A server that stops later is started again at its
  next call. Restart Strata after changing the list.
- The model sees each tool as `<server>__<tool>`. A request's own tool of the same name wins.
- A request opts in with `"strata_mcp": true` on `/v1/chat/completions`. The server offers the MCP tools beside the
  request's own; when the model calls one, the server runs it, adds the call and its result to the conversation and
  lets the model go on, up to `max_rounds` times per answer. A call to one of the request's own tools ends the turn as
  usual (the client runs it), and MCP calls written in the same answer are then not run. A tool that fails, or takes
  longer than `timeout_s`, gives the model a result starting with `error:`. A result longer than `max_result_chars`
  is cut, with a note, before the model reads it. Stopping the request stops a running tool too.
- The stream carries the tool activity as chunks with an empty delta and a `strata_mcp` field (`start`, `call`,
  `result`, `limit` events); a non-streamed answer lists them in `message.strata_mcp`. Each round is a new pass on
  the engine and waits its turn like any request.

**Who may use them.** By default only Strata's own web page: its chat sends `"strata_mcp": true` (the Sampling
drawer's **Use tools from MCP servers** switch), and its requests name the page in `Origin`. A request with
`"strata_mcp": true` and no `Origin` (an API client: an SDK, a bot, curl, anyone with the key) is a 403 unless the run
config has `"mcp_for_api": true`; a page of another site is refused unless its origin is in `"trusted_origins"`.
Clients that do not send `strata_mcp` see the API as it is, with their own tools only.

**Security.** MCP tools run on this machine with the server's user rights, and the model decides when to call them,
also because of what it reads (a web page or a file can contain instructions). Give a filesystem server only the
folders it needs, prefer read-only tools, and add no server you do not trust. A server's address appears in errors
and the log without its user name, password, query string or fragment, which can hold a token; `GET /mcp` shows no
addresses.

## Settings

Settings live in three places: the chat page (this browser only), the shared settings (every client, kept by the
server) and the run config (`strata-glm-<size>.json`, read when the server starts). The full list of run config keys,
including the engine's, is in [INSTALL.md](INSTALL.md#the-run-config).

### The chat page's settings

The Sampling drawer (the gear button in the chat) holds: **Thinking** off or on (default on), **Temperature**
(0-1.5, default 1.0; 0 is greedy), **Top-p** (default 0.95) and **Top-k** (default 0, off), **Max tokens** (empty:
until done), **Seed** (empty: random), **Show thinking**, **Use tools from MCP servers** (shown when MCP servers are
configured) and **Use for other apps too** (the shared settings, below). They are kept in this browser. With each
request the chat sends the temperature (and top-p and top-k when the temperature is above 0), the seed and max
tokens when they are set, and `"reasoning_effort": "none"` when thinking is off. What it does not send falls back to
the defaults below: with thinking **On** it sends nothing, so a run config or shared setting that turns thinking off
applies. The About tab holds the API key (also kept in this browser) and the theme.

### Shared settings for other clients

**Use for other apps too** makes the chat page's settings the defaults for every client: other apps get them for
whatever they leave out. The server keeps them next to the run config (`strata-glm-<size>.shared-settings.json`), so
they survive a restart. Over the API:

```bash
curl http://127.0.0.1:8101/settings                       # {"shared": true|false, "defaults": {...}}
```

`POST /settings` with `{"defaults": {...}}` sets them and `{"defaults": null}` clears them; it is taken only from
Strata's own page (JSON, no other site's `Origin`). The keys and their checks: `reasoning_effort` (`none`, `low`,
`medium`, `high`), `temperature` (0-2), `top_p` (above 0, at most 1), `top_k` (a whole number >= 0), `seed` and
`max_tokens` (positive whole numbers) and `reasoning_budget_tokens` (a whole number >= 0). Any other key, NaN or
Infinity is a 400 that names the key. A request's own value always wins.

### The Settings view

The About tab's **Model settings** card shows and changes a few keys of the run config (`GET /config`; `POST /config`
with `{"set": {"<key>": <value or null>}}` from Strata's own page only). A change is used from the next start of the
server; an empty field removes the key, so its default applies. Every other key of the file stays as it is, and the
earlier file is kept as `strata-glm-<size>.json.bak`.

| Key | What it does |
| --- | --- |
| `sampling.temperature`, `sampling.top_p`, `sampling.top_k`, `sampling.min_p` | Sampling defaults for requests that send none ([below](#sampling-defaults)) |
| `reasoning_budget_tokens` | Cap every request's thinking at this many tokens (0 or empty: no cap) |
| `fit_max_tokens` | Shorten an output cap that does not fit the context instead of a 400 |
| `anthropic_thinking` | `model` (Anthropic requests that do not ask think anyway) or `on_request` |
| `aliases` | Other model names the server lists and answers to |
| `idle_unload_s` | Unload the model after this many seconds without requests (0 or empty: never) |
| `lazy_load` | Start without loading the model; the first request loads it |
| `engine_silence_s` | End a request when the engine says nothing for this long (default 300 s, 0 waits for ever) |
| `api_monitor` | Keep the last 100 requests' prompts and answers in memory for `/api-monitor` |
| `open_browser` | Open the chat page when the model is ready |
| `vram_reserve_gb` | The engine's `--vram-reserve`: GB left free on each GPU after its expert cache (engine default 1) |
| `park_mib` | The engine's `--park-mib`: RAM in MiB for parked conversations and checkpoints (engine default 8192, 0: none) |

The network and key settings, the MCP servers and `before_load` (which run programs) cannot be changed there.

### Sampling defaults

A request that leaves a sampling field out gets, in this order: the shared settings, the run config's `"sampling"`
block, and GLM's recommendation, **temperature 1.0 and top_p 0.95**. The engine on its own would sample greedily; the
server always sends these defaults. The `"sampling"` block takes `temperature` (>= 0), `top_p` (above 0, at most 1),
`min_p` (0-1), `top_k`, `penalty_last_n` (also `repeat_last_n`), `seed`, `presence_penalty`, `frequency_penalty`
(>= 0) and `repetition_penalty` (also `repeat_penalty`, above 0):

```json
"sampling": {"temperature": 0.7, "top_p": 0.95, "top_k": 40}
```

A bad value stops the start with a line that names it; an unknown key is named and ignored. The server window lists
the block's values at the start. `GET /props` shows the defaults in effect, under llama.cpp's names.

### Run config keys the server reads

Beside the engine's keys (`exe`, `args`, `cwd`, `log`, `gpu`, `split`, `env`, `lib_dirs`: see
[INSTALL.md](INSTALL.md#the-run-config)), the server reads these. Each is optional.

| Key | Default | What it does |
| --- | --- | --- |
| `host` | `127.0.0.1` | Where the server listens; `0.0.0.0` lets other machines in (set `api_key`) |
| `api_key` | none | The key clients must send; several as `"k1,k2"` or a list ([Security](#security)) |
| `allowed_hosts`, `trusted_origins`, `cors_origins` | none | Host names, own-page origins and CORS origins ([Security](#security)) |
| `model_name`, `aliases` | the GGUF's name, none | The model's name and other names it answers to |
| `chat_template` | the GGUF's | A Jinja template file to use instead (also `--chat-template`) |
| `sampling` | none | Sampling defaults ([above](#sampling-defaults)) |
| `reasoning_effort` | none | Thinking for requests that do not say: `none` is off, `low`, `medium`, `high` are on |
| `reasoning_budget_tokens` | 0 | A thinking cap for every request ([Thinking](#thinking)) |
| `anthropic_thinking` | `model` | `on_request`: Anthropic requests think only when they ask |
| `fit_max_tokens` | false | Shorten an output cap that does not fit instead of a 400 |
| `repeat_stop_tokens` | 256 | End a reply that repeats one token this many times in a row (0: off) |
| `reasoning_loop_recovery` | false | `"stop"`: end a reply whose thinking repeats whole passages |
| `reasoning_close_retry` | false | Close the thinking once when a reply ends inside it with no answer |
| `tool_call_recovery` | false | Also read a declared tool's call written as JSON inside `<tool_call>` |
| `codex_thread_titles`, `codex_compaction_cache` | false | The Codex helpers ([above](#the-responses-api-and-codex-cli)) |
| `mcp_servers` (or `mcpServers`), `mcp`, `mcp_for_api` | none, the defaults above, false | [The MCP tool loop](#the-mcp-tool-loop) |
| `api_monitor` | false | The request monitor ([Monitoring](#the-request-monitor)) |
| `lazy_load`, `idle_unload_s`, `min_free_vram_mib`, `before_load`, `before_load_timeout_s`, `before_load_strict`, `before_load_ignore_failure` | off | [Loading and unloading](#loading-and-unloading) |
| `engine_silence_s` | 300 | Seconds the engine may say nothing during a request before the server ends it (0: no limit) |
| `open_browser` | the start script's `--open` | `false` keeps the browser closed even when the start script passes `--open` |

`--reasoning-effort` written into `"args"` by mistake is not an engine option: the server takes it out and uses it as
`"reasoning_effort"`, with a line at the start.

### Loading and unloading

The model loads when the server starts, and the port opens once the engine is ready. Other options:

- `"lazy_load": true` (or `--lazy`): the server starts without the engine; the first generation request, or
  `POST /load`, starts it.
- `"idle_unload_s": N` (or `--idle-unload N`): unload the model after N seconds without requests; the next request
  loads it again.
- `POST /unload` unloads it between requests: `{"status": "unloaded"}`, `"not loaded"`, or 409 `"busy"` while a
  request runs or waits. `?wait=1` waits up to 300 s for the running request instead; `?wait=<seconds>` waits that
  long (at most an hour). `POST /load` loads it now (`{"status": "loaded"}`, or a 503 that says why).
  `POST /v1/load` and `/v1/unload` take `{}` or `{"model": "<name>"}` and answer with the model's status; they are a
  409 while a request runs or waits and a 404 for another model name. All four take JSON from Strata's own page or
  from a client without `Origin`:

  ```bash
  curl -X POST http://127.0.0.1:8101/unload -H "Content-Type: application/json" -d '{}'
  ```

- `"min_free_vram_mib": N` (or `--min-free-vram-mib N`): an unloaded model is loaded again only when the first GPU of
  the config has N MiB free (waiting up to 15 s for memory another program just gave back); otherwise the request
  gets a 503 with `Retry-After`. With three GPUs only the first is checked.
- `"before_load": "<command>"` (a string or a list; or `--before-load`): a command run before the model is loaded
  again, for example another server's unload. It may run `"before_load_timeout_s"` seconds (default 120). When it
  fails, the model is loaded anyway, unless `min_free_vram_mib` is set and too little VRAM is free (503).
  `"before_load_strict": true` refuses (503) whenever it fails; `"before_load_ignore_failure": true` always loads,
  without the VRAM check.

Parked conversations live in the engine's memory and are gone after an unload. To run Strata behind llama-swap
instead, see [LLAMA_SWAP.md](LLAMA_SWAP.md).

### Command line

`serve/server.py --engine strata --config strata-glm-<size>.json` is what the start script runs, with `--port` and
`--open`. The other options: `--host`, `--port` (default 8101), `--gpu` (one card or a list, as nvidia-smi numbers
them), `--chat-template FILE`, `--api-key` (also `STRATA_API_KEY`), `--mcp-config FILE`, `--lazy`, `--api-monitor`,
`--idle-unload SECONDS`, `--min-free-vram-mib N`, `--before-load CMD` and `--fit-max-tokens`. Each has a run config
key, except `--port`, which the start script passes (`--open` has `"open_browser"`). `--engine mock`, the default
when `--engine` is not given, runs a scripted stand-in for the model (`--script TEXT`), for trying clients without a
GPU.

### Environment variables

| Variable | What it does |
| --- | --- |
| `STRATA_API_KEY` | The API key (as `--api-key`); an empty value stops the start rather than turning the key off |
| `STRATA_ALLOWED_HOSTS` | More host names the server answers to, comma-separated |
| `STRATA_MAX_BODY_MIB` | The request body limit in MiB (default 256) |
| `STRATA_HTTP_BACKLOG` | Connections the operating system may hold for the server (default 256) |
| `STRATA_ENGINE_READY_S` | How long a starting engine may go without progress before the server ends it (default 900; 0: no limit) |
| `STRATA_ENGINE_STALL_S` | How long an engine may stay silent with no CPU time, no disk I/O and an idle GPU before the server ends it as frozen (default 90; 0: off; needs psutil) |
| `STRATA_REQUEST_LINES` | `1`: one line per request on the server's output (for a supervisor such as llama-swap) |
| `STRATA_DROP_CACHE_ON_EXIT` | `1`: give the model files' page cache back when the server stops |
| `STRATA_DEBUG` | `1`: one line per request and the raw model text in the server window |
| `STRATA_KEEP_EMPTY_TURNS` | `1`: render empty earlier assistant turns in the prompt (they are left out by default) |

## Several clients on one engine

Agents, a chat and a bot can share one engine. Requests run **one at a time, in the order they arrived** (a FIFO).
A waiting request gets keep-alives every 10 s; a request whose client goes away while it waits gives up its place.
`GET /status` and the Monitor show how many are queued.

The KDA layers' state is a running sum that cannot be rewound, so the engine can only continue a state whose tokens
are the start of the new prompt. To keep each client's conversation anyway, the engine keeps several
(`src/glm/serve_loop.cpp`):

- **Parking.** When a request starts a different conversation, the current one's whole state is copied to RAM. A
  later request that continues it gets it back and reads only its new tokens. Each tool round of an agent is a new
  request that waits its turn, so between two rounds another client's request may run; its conversation is then
  parked and resumed.
- **Checkpoints.** The server tells the engine where each reply begins (`cp=` on the engine's `GEN` line: the prompt
  without the generation prompt `<|assistant|><think>`). Once a user message follows, the template renders the earlier
  reply without its thinking, so the prompt differs from the engine's state from that point on; the engine restarts
  from the checkpoint and reads the reply and the new message, not the whole conversation. A regenerated reply
  restarts there too. The engine keeps the two latest checkpoints per conversation, and none closer than 256 tokens
  to the start (`--checkpoint-min`). The engine's line protocol (`GEN` with `cp=`, `end=`, `logit_bias=` and the
  sampling keys) is in [GLM53_SERVER_PORT.md](GLM53_SERVER_PORT.md).
- **Memory.** `--park-mib` (engine default 8192; setup writes 8192, or 4096 when less than 40 GB of RAM is left
  beside the experts) caps what parked conversations and checkpoints use. The conversation used least recently goes
  first, then the live one's older checkpoints.

How big these are, calculated from the engine's snapshot code with the model's shapes (not measured): a parked
conversation is about 0.15 GB of KDA state plus about 11 KB per token (the MLA cache's 512-wide f16 latent in 11
layers, and the indexer's pooled keys); a checkpoint is about 0.15 GB. So `--park-mib 8192` holds one conversation of
up to roughly 700,000 tokens, or several shorter ones. A conversation that does not fit is not parked (the engine log
says `... more than --park-mib; not kept`): when another client takes a turn, it is lost with its checkpoints, and
its next request reads its prompt again from the start. A 1,048,576-token conversation needs `--park-mib` of about
12,000 or more to be parked, and that RAM must be free beside the experts.

On the CPU reference engine this does not change the answers: `tools/glm53_park_check.py` checks each resumed reply
against a fresh engine's, token for token (with small random models). On the GPU engine, chunked prompts and the
expert cache change the rounding, so a resumed reply may differ from a fresh one in its later tokens; that check has
not run on a GPU yet.

The **Conversation cache** card in the Monitor tab shows: the parked conversations and checkpoints and the RAM they
use against `--park-mib`; how many requests reused part of their prompt; the last request's reused tokens of its
prompt; the reused tokens since the start (and their share of all prompt tokens); how many conversations were
parked, resumed and dropped; and the last switch. It reads the engine's log (`conversations: ...` lines). With
parking off (`--park-mib 0`) the engine keeps only the live conversation, and the card says so. `/metrics` has the
same under `"conversation_cache"`.

## Monitoring

### The Monitor tab

The web app's Monitor tab polls `GET /metrics` once a second:

- **Model state**: idle, reading the prompt (with progress), generating (with the phase: thinking, answering,
  writing a tool call), queued; the model, engine version and backend.
- Eight cards with a minute of history each: **Speed** (decode and prefill tokens/s), **GPU load**, **VRAM**, **GPU
  temp**, **Power**, **PCIe** (link and traffic to the GPU), **CPU** and **Disk read**. With several GPUs, a card
  shows the sum (VRAM, power), the mean (load) or the hottest GPU (temperature), and each GPU's own value.
- **Context fill** (the running request's prompt and output against the context), **Experts in VRAM** (the expert
  cache's slots and size), **Expert-cache hits** (the last request's and since the start), **System RAM** and **GPU
  temperature**.
- **Recent requests**: time, status, prompt, reused, output, prefill and decode tokens/s, the expert cache's hit rate
  and the duration, with the totals since the start. **Show all** lists every request kept (up to 500).
- **Conversation cache** ([above](#several-clients-on-one-engine)) and, when configured, the **MCP servers** with
  their state and tools.

GPU readings come from NVIDIA's NVML library (installed with the driver); CPU, RAM and disk from `psutil` (setup
installs it). A reading that is not available shows as a dash.

### GET /metrics (JSON)

| Key | What it holds |
| --- | --- |
| `engine` | `model`, `max_context`, and the engine's `INFO` line: `version`, `backend` (`cuda` or `cpu-reference`), `park_mib`, `logit_bias`, `expert_slots`, `expert_cache_mib` |
| `live` | `state` (`unloaded`, `idle`, `reading`, `generating`), `queued`, `phase`, `prompt_tokens`, `prompt_read` / `prompt_total` while reading, `generated`, `max_tokens`, `elapsed_s`, `tok_s` (over the last 2 s), `tok_s_mean`, `prefill_tok_s_mean` |
| `requests` | The last 12 finished requests, newest first (`?requests=all`: every one kept, up to 500): `time`, `duration_s`, `finish`, `prompt_tokens`, `reused`, `prompt_read`, `output_tokens`, `prompt_ms`, `decode_ms`, `decode_tok_s`, `hit_rate`, `hits`, `lookups`, `reasoning_repeat_coverage` |
| `totals` | Since the server started: `requests`, `prompt_tokens`, `reused`, `output_tokens`, `prompt_ms`, `decode_ms`, `hits`, `lookups` |
| `conversation_cache` | `enabled`, `budget_mib`, `parked`, `checkpoints`, `bytes`, `parks`, `restores`, `evictions`, `last_event`, `last_tokens`, `requests`, `requests_reused`, `reused_tokens`, `prompt_tokens`, `last_reused`, `last_prompt` |
| `hardware`, `hardware_static`, `history` | The latest readings (`gpu_util`, `gpu_mem_used`, `gpu_temp`, `gpu_power`, `gpu_pcie_*`, `gpus` per card, `cpu`, `ram_used`, `ram_total`, `disk_read_mb`, ...), the fixed facts (GPU and CPU names, cores), and 60 readings per series |

`hit_rate` is the share of expert lookups that found the expert in the VRAM cache while the reply was written (the
engine counts decode only).

### Prometheus

`GET /metrics` answers a scraper in the Prometheus text format when the request says `Accept: text/plain` or
`application/openmetrics-text` (as Prometheus does), or has `?format=prometheus`; anything else gets the JSON. With an
API key the scraper sends it as a bearer token. The names are vLLM's, so dashboards written for vLLM read Strata, and
every sample has the label `model_name` (`serve/prometheus.py`):

| Metric | Type | What it counts |
| --- | --- | --- |
| `vllm:num_requests_running` | gauge | 1 while a request reads its prompt or generates |
| `vllm:num_requests_waiting` | gauge | Requests waiting for their turn |
| `vllm:kv_cache_usage_perc` | gauge | The running request's prompt and output as a share of the context (0 when idle) |
| `vllm:prompt_tokens_total` | counter | Prompt tokens of the finished requests |
| `vllm:generation_tokens_total` | counter | Generated tokens |
| `vllm:request_success_total` | counter | Finished requests |
| `vllm:prefix_cache_queries_total`, `vllm:prefix_cache_hits_total` | counter | Prompt tokens, and those the engine already held (resumed or checkpointed) |
| `vllm:num_preemptions_total` | counter | Always 0: Strata does not preempt |
| `vllm:time_to_first_token_seconds` | histogram | Time to the first token |
| `vllm:inter_token_latency_seconds` | histogram | Time between tokens: each request's mean gap, once per gap |
| `vllm:e2e_request_latency_seconds` | histogram | A request from start to end |

The facts vLLM has no name for come in the same scrape under `strata:`, named after their JSON key, and a value the
server does not have (no GPU telemetry) has no sample rather than a zero: `strata:live_state{state="..."}` (1 for the
current one of `unloaded`, `idle`, `reading`, `generating`), `strata:live_tok_s`, `strata:live_prefill_tok_s_mean`,
`strata:live_prompt_read`, `strata:engine_max_context`, `strata:totals_prompt_seconds_total`,
`strata:totals_decode_seconds_total`, `strata:last_hit_rate` and `strata:last_decode_tok_s` (the last request's),
`strata:gpu_util`, `strata:gpu_mem_used_bytes`, `strata:gpu_mem_total_bytes`, `strata:gpu_temp_celsius`,
`strata:gpu_power_watts` (one sample per card, label `gpu`), `strata:cpu`, `strata:ram_used_bytes` and
`strata:ram_total_bytes`.

`GET /metrics/prometheus` gives the totals under llama-server's names instead, without labels:
`llamacpp:prompt_tokens_total` (tokens read), `llamacpp:prompt_tokens_cached_total`, `llamacpp:prompt_seconds_total`,
`llamacpp:tokens_predicted_total`, `llamacpp:tokens_predicted_seconds_total`, `llamacpp:prompt_tokens_seconds` and
`llamacpp:predicted_tokens_seconds` (average rates), `llamacpp:requests_processing`, `llamacpp:requests_deferred`,
`strata:requests_total` and `strata:model_loaded`.

Ready to use, in [docs/monitoring/](monitoring/): `prometheus.yml` (a scrape config for port 8101, the API key as a
bearer token), `servicemonitor.yaml` (the same for the Prometheus Operator) and `grafana-strata.json`, a Grafana
dashboard to import (it asks for the Prometheus data source). Its first row uses only vLLM's names (requests running
and waiting, tokens per second, time to first token, time between tokens, request duration, prompt tokens served from
the cache, requests per minute, context in use), so it also reads a vLLM server; the second row has the `strata:`
panels (engine state, rates, expert cache hit rate, engine time, GPU and RAM).

### Status routes

- `GET /status`: `busy`, `queued`, and while a request runs its `phase`, `prompt_tokens`, `generated`, `max_tokens`,
  `thinking_budget`, `tool`, `elapsed_s`, `tokens_per_s` and the last 600 characters of the answer (`tail`), which is
  why it needs the key.
- `GET /v1/status`: `service`, `model`, `loaded`, `auto_load`, `engine` (version), `uptime_s`, `context`,
  `concurrency` (`serving: 1`), `dialects` (the four API routes), `structured_output` (`prompt_and_validate`, no
  constrained decoding, streams buffered), `vision` (off), `activity`, `last_timings` (llama.cpp's names) and
  `machine` (the GPU and RAM in use).

### The request monitor

Off by default, because it keeps prompts and answers in memory. `"api_monitor": true` in the run config (or
`--api-monitor`) turns it on; otherwise nothing is recorded and its routes answer 404. `/api-monitor` then shows the
model state, load and unload buttons, the running and queued requests and each request's body, answer, thinking,
timings (queue, load, first token, decode) and status. `GET /api/requests` lists the kept requests;
`GET /api/requests?id=<id>` returns one. It keeps the newest 100 requests until the server stops, each text field cut
at 262,144 characters (and marked so); headers and keys are not recorded. Treat it as sensitive: anyone with the key,
or anyone who can reach a server without one, can read it, and the server window says so at the start.

## Security

The server listens on `127.0.0.1` (this machine only) unless the run config's `"host"` or `--host` says otherwise.
Listening beyond it without an API key prints a warning: anyone on the network could then use the model.

- **API keys.** `"api_key"` in the run config, `--api-key` or `STRATA_API_KEY`. Clients send it as
  `Authorization: Bearer <key>` or `x-api-key: <key>`. Several keys: `"k1,k2"` (as llama.cpp takes them) or a list in
  the config (the way to use a key with a comma in it); any one passes. Keys are compared in constant time, and spaces
  around a key are dropped. An empty `--api-key` or `STRATA_API_KEY` stops the start rather than turning the check
  off; an empty `"api_key"` in the run config leaves the server without a key, with a warning. The routes that need it
  are marked in [Routes](#routes).
- **Host names (DNS rebinding).** A web page of another site can point its own name at 127.0.0.1 and then reach the
  server as if it were its own. So without an API key the server answers only requests whose `Host` is a name it knows:
  `localhost` and `*.localhost`, any IP address, the address it listens on and, when it listens beyond this machine,
  the machine's name (`name`, `name.local`) and `host.docker.internal`, and the hosts of `"trusted_origins"`, on any
  port. Others get a 403 that names the setting, and one line in the server window. To reach it under another name (a
  reverse proxy, a tunnel, a DNS name), add `"allowed_hosts": ["strata.example.com"]` or `STRATA_ALLOWED_HOSTS`
  (comma-separated; `".example.com"` allows that name and every name below it; `"*"` turns the check off). With an API
  key the check is off: such a page cannot send the key. Requests without a `Host` header pass.
- **Web pages without an API key.** A `POST` to `/v1/...` that carries an `Origin` (a browser sent it) is answered only
  for Strata's own page, pages on `localhost` or on a name the server answers to, the origins in `"trusted_origins"` or
  `"cors_origins"`, and browser extensions and desktop apps (`chrome-extension://`, `moz-extension://`, `app://`),
  and only with a JSON body (else 415). Other pages and `Origin: null` get a 403. Clients that send no `Origin` (curl,
  the SDKs, other servers) are not affected. With an API key, the key decides.
- **Own-page routes.** `POST /settings`, `/config`, `/load`, `/unload`, `/v1/load`, `/v1/unload` and the MCP tools
  take only JSON and no other site's `Origin`, also with an API key. When the web app is reached through a reverse
  proxy or tunnel whose address differs, add that address to `"trusted_origins"` (for example
  `["https://strata.example.com"]`).
- **CORS.** Off by default. `"cors_origins": ["https://chat.example.com"]` lets pages of those origins call `/v1/*`
  from a browser; `["*"]` lets any page do it (only sensible with an API key; the start says so). It never opens the
  other routes.
- **The request monitor** keeps prompts and answers; see [above](#the-request-monitor).
- **MCP tools** run programs with the server's rights; see [The MCP tool loop](#the-mcp-tool-loop).
- **Bodies and timeouts.** A request body may hold at most 256 MiB (`STRATA_MAX_BODY_MIB`); a larger one is a 413
  before it is read. A body over 16 MiB takes one of two places for its request's whole time, because a parsed body
  takes many times its size in memory; a third is a 503 with `Retry-After: 30`. A request's line, headers and body must
  keep arriving: 120 s without data ends it (a 408 for the body). The answer has no such limit, so a long stream is
  not cut. Control routes (`/load`, `/unload`) take at most 64 KiB. Every answer closes its connection.

For reporting a security problem, see [SECURITY.md](../SECURITY.md).

## Limits

- **One request at a time.** The others wait in arrival order ([above](#several-clients-on-one-engine)).
- **Context.** The engine's `--ctx` in the run config (setup offers 32,768 to 1,048,576 tokens; 1,048,576 is
  GLM-5.3-Flash's trained context). A prompt plus its output cap must fit, less 8 tokens; requests are never cut. Each
  token of context takes about 11 KB of VRAM in the MLA cache (calculated: 1.5 GB at 131,072 tokens, 5.9 GB at
  524,288, 11.8 GB at 1,048,576), which the expert cache gives up.
- **Text only.** No pictures.
- **No slot save or restore** (`POST /slots/...` is a 501), no batching, no stored responses.
- **Body size.** 256 MiB per request by default ([Security](#security)).
- **Engine watchdogs.** A running request whose engine says nothing for `engine_silence_s` (300 s; longer while a
  prompt chunk is read) ends with an error, and the next request starts the engine again. An engine that is silent
  for 90 s with no CPU time, no disk I/O and an idle GPU is ended as frozen (`STRATA_ENGINE_STALL_S`, needs psutil).
  A starting engine must say `READY` within 900 s (`STRATA_ENGINE_READY_S`); while the warm-up that reads the experts
  into memory before `READY` moves on (its progress lines in the engine log), the 900 s count from its latest line.
  How long the warm-up takes on the real model is not measured yet.
