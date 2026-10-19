# Per-request token biases

`POST /v1/chat/completions` and `POST /v1/completions` accept `logit_bias`, and so do `/v1/messages` and
`/v1/responses` (the request is also the sampling settings there). The same bias applies to every token the request
generates. Nothing is biased by default, and a request without the field has no bias.

## Finding a token's ID

Token IDs belong to the model's tokenizer, so an ID from another model means nothing here. Leading spaces and
letter case give different tokens (`Hello` and ` Hello` are usually two different IDs), and one word can take
several tokens. The tokenizer is in the model's GGUF file. This prints the IDs of a piece of text:

```bash
python3 -c "import sys; sys.path.insert(0, 'tools'); from strata_tokenizer import Tokenizer; \
t = Tokenizer.from_gguf('/path/to/the/first/shard.gguf'); print(t.encode('Hello'), t.encode(' Hello'))"
```

Run it in the Strata folder (with `.venv/bin/python` if `python3` lacks the `regex` package). It reads the
tokenizer from the file's metadata, not the weights.

## Example

To make the model avoid a word, ban its token. With the ID that the command above prints for `Hello` in place
of `<ID>`:

```json
{
  "model": "glm-5.3-flash",
  "messages": [{"role": "user", "content": "Reply with one English greeting, one word only."}],
  "max_tokens": 8,
  "temperature": 0,
  "reasoning_effort": "none",
  "logit_bias": {"<ID>": -100}
}
```

The equivalent llama.cpp-style pair list is `"logit_bias": [[<ID>, false]]`. Numeric pair values work too, for
example `[[42, 2.5]]`.

## Semantics and limits

Biases must be finite numbers in `[-100, 100]`. Negative values discourage a token, positive values encourage it,
and **-100 excludes it** (`false` in a pair list means the same). The engine adds the bias to the logits before
the repetition, frequency and presence penalties, temperature and the top-k, top-p and min-p filters. It works the
same with greedy decoding (`temperature` 0). Banning a token does not ban a word or a language: other tokens
can still spell the same text.

The bias belongs to its request only. It is not kept in a parked conversation or checkpoint, and it does not
change the model's state, so the next request without the field is unbiased again.

The server refuses (HTTP 400) malformed entries, duplicate IDs in a pair list, IDs outside the vocabulary,
out-of-range biases and a ban of the whole vocabulary. On every route this happens before streaming starts.
The engine says `logit_bias=1` in its `INFO` line. A server talking to an engine that lacks it refuses a non-empty
bias instead of ignoring it.

There is no startup bias file: send the field with each request that needs it.
