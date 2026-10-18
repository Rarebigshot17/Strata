# How does Strata work?

GLM-5.3-Flash has 320 billion parameters. Models this size normally run on servers with hundreds of gigabytes of
graphics memory. A PC with a few gaming cards has 24-50 GB. Strata makes the model fit by **sharing the work across
the whole PC**. Think of a kitchen: the things you use all the time stay on the counter, and the rest waits in the
pantry. Back to the [README](../README.md#how-does-it-work).

## Who does what

- **The model is a team of small specialists ("experts").** Each of its 42 expert layers has 288 of them, 12,096
  in all, and each word it writes needs only 8 per layer. So the graphics cards don't need all of them at once.
- **The graphics cards** do the part of the work that every word needs, and keep the experts that are asked for
  most often. They keep learning which ones those are while you use the model.
- **The RAM** holds every expert. When a word needs one the cards don't have, **the processor** works on it while
  the cards do their part.

The same in technical terms:

- **GPU (VRAM):** everything except the routed experts. That is the 34 KDA (linear attention) layers and the 11
  MLA (sparse attention) layers with their indexer, the mHC residual mixing, the routers, the shared experts and
  the output head. VRAM also holds the MLA cache and the KDA state. The layers are split over the cards in order,
  by free memory or by `--split`. What VRAM is left holds an **expert cache** of the experts each card's layers
  use most.
- **The expert cache:** every expert has a use score that halves over about 700 tokens. A missed expert takes a
  free slot, or replaces a cached one whose score is clearly lower. At most 8 experts are copied in per token
  (`--max-swaps`), so a change of topic moves the cache over a few dozen tokens instead of stalling one.
  `--expert-profile FILE` saves the scores at exit and loads them at the next start, so the cache starts full.
- **RAM:** every routed expert stays in the GGUF file, mapped into memory. At 3-bit that is about 117 GB, at 4-bit
  about 162 GB (calculated from the model's shapes). The CPU computes the experts the cache doesn't hold, in place,
  while the GPU computes the shared expert. Before the server says it is ready, the engine reads the experts once
  so they are in RAM rather than on disk (`--no-warm` skips this).
- **Reading a prompt:** the prompt is read in chunks of 256 tokens. Each expert that a chunk uses runs once on the
  GPU for all of the chunk's tokens. It comes from the expert cache when it is there. Otherwise it streams in from
  RAM through two buffers, so one expert copies in while the previous one computes.

## Long conversations

- **The KDA layers** keep a fixed-size state (64 heads of 128 x 128 per layer), however long the text is.
- **The MLA layers** keep a 512-wide latent per token: 11 KB per token over the 11 layers, in VRAM (1.5 GB at 131,072
  tokens, 11.8 GB at 1M; calculated). Their indexer pools the keys over 16 tokens and, for each new token, picks
  the 128 pools (2,048 tokens) that matter most. So past
  about 2,000 tokens of context each new token attends to a fixed amount of the text, not all of it. Below that
  the indexer picks every pool and attention is plain causal.

The context (`--ctx`, up to 1,048,576 tokens) is the longest conversation the engine accepts. Its MLA cache comes
out of the VRAM that the expert cache would otherwise use.

## Several clients

Agents, a chat and a bot can share one engine. Requests wait in the server and run one at a time, in the order
they arrived. The KDA state is a running sum that cannot be rewound, so the engine keeps each conversation's state
instead:

- **Parking.** When a request starts a different conversation, the current one's whole state is copied to RAM. A
  later request that continues it copies it back and reads only the new tokens.
- **Checkpoints.** The server marks where each reply begins. The next turn renders the earlier reply without its
  thinking, so it differs from the engine's state from that point on. The engine restarts from the checkpoint and
  reads the reply and the new message, not the whole conversation. A regenerated reply restarts there too.
- **Memory.** `--park-mib` (default 8192) caps what parked conversations and checkpoints use. The conversation used
  least recently goes first.

None of this changes the answers. `tools/glm53_park_check.py` checks each resumed reply against a fresh engine's,
token for token.

## How it is checked

- **The CPU reference** (`strata-glm-ref`) runs the whole model on the CPU, one token at a time, with ggml's own
  dot products. On small random GLM-5.3-Flash models it matches llama.cpp to float rounding
  (`tools/glm53_selftest.sh`, which needs no GPU and no download).
- **The GPU engine** (`strata-glm`) is checked against the CPU reference layer by layer (`--check-cpu`,
  `tools/glm53_gpu_check.sh`). On the real model, its logits are compared with llama.cpp's (`tools/glm53_compare.py`).
- The steps, and what each one has shown so far: [GLM53_PORT.md](GLM53_PORT.md).

## Credits

- Model: GLM-5.3-Flash by Z.ai. The GGUF files Strata downloads are
  [Unsloth's](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF). Their license applies to the model files.
- Strata itself: [Niko1221/Strata](https://github.com/Niko1221/Strata), which ran Qwen3.8-Flash-Next. This fork
  keeps its design (experts in RAM, the most used cached in VRAM, the rest on the CPU), its server and its web app,
  and replaces the engine.
- Built with parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp) (MIT): the quantized dot products
  on the CPU and GPU, and the reference that every part of the engine is checked against.

## License

Strata is open source under the [MIT License](../LICENSE). A few parts have their own licenses: `third_party/ggml`
(MIT, llama.cpp / ggml) and the web app's font (SIL Open Font License 1.1, `serve/web/fonts/OFL.txt`). The model
is not part of this repository. Its own license applies to its files.
