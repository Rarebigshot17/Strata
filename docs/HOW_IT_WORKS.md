# How does Strata work?

GLM-5.3-Flash has 320 billion parameters. Models this size normally run on servers with hundreds of gigabytes of
graphics memory. A PC with a few gaming cards has far less: the three cards of napster-ai, the PC Strata is built
for, have 46 GB together. Strata makes the model fit by **sharing the work across the whole PC**. Think of a
kitchen: the things you use all the time stay on the counter, and the rest waits in the pantry. Back to the
[README](../README.md).

The real model has not run yet, and there are no measured speeds; on napster-ai's cards the engine has run small
random test models. The sizes on this page come from the real files or are calculated or derived, and say which;
[How it is checked](#how-it-is-checked) says what has run.

## Who does what

- **The model is a team of small specialists ("experts").** Each of its 42 expert layers has 288 of them, 12,096
  in all, and each word it writes needs only 8 per layer. So the graphics cards don't need all of them at once.
- **The graphics cards** do the part of the work that every word needs, and keep the experts that are asked for
  most often. They keep learning which ones those are while you use the model.
- **The RAM** holds every expert. When a word needs one the cards don't have, **the processor** works on it while
  the cards do their part.

The same in technical terms:

- **GPU (VRAM):** everything except the routed experts. That is the 34 KDA (linear attention) layers and the 11
  MLA (sparse attention) layers with their indexer, the mHC residual mixing, the routers, the dense layers 0-2,
  the shared experts and the output head. VRAM also holds the MLA cache and the KDA state. The layers are split
  over the cards in order, in proportion to each card's free VRAM at start or to `--split` (the run config's
  `"split"`), with the output head on the last card. What VRAM is left, less `--vram-reserve` (1 GB per card by
  default), holds an **expert cache** of the experts each card's layers use most.
- **The expert cache:** every expert has a use score that halves over about 700 tokens. A missed expert takes a
  free slot, or replaces a cached one whose score is clearly lower. At most 8 experts are copied in per token
  (`--max-swaps`), so a change of topic moves the cache over a few dozen tokens instead of stalling one.
  `--expert-profile FILE` saves the scores when the engine leaves its request loop (the server's `QUIT`, or the
  server's end of the pipe closing) and loads them at the next start, so the cache starts full. Every ordinary stop
  of the server (Ctrl+C, closing its terminal, SIGTERM) ends the engine with `QUIT`; an engine that is killed ends
  without saving them ([TROUBLESHOOTING.md](TROUBLESHOOTING.md) has the details). Setup puts one in the run config.
- **RAM:** every routed expert stays in the GGUF file, mapped into memory. At 3-bit that is 109.9 GB (counted in the
  real file), at 4-bit about 146 GB (derived from its download, not measured). The CPU computes the experts the cache
  doesn't hold, in place, while the GPU computes the shared expert. Before the server says it is ready, the engine
  reads the experts once so they are in RAM rather than on disk (`--no-warm` skips this).
- **Reading a prompt:** the prompt is read in chunks of 256 tokens (`--prefill-chunk`). Each expert that a chunk
  uses runs once on the GPU for all of the chunk's tokens. It comes from the expert cache when it is there.
  Otherwise it streams in from RAM through two buffers, so one expert copies in while the previous one computes.

## Long conversations

- **The KDA layers** keep a fixed-size state (64 heads of 128 x 128 per layer), however long the text is.
- **The MLA layers** keep a 512-wide latent per token (11 KB per token over the 11 layers) and their indexer's keys,
  pooled over 4 tokens (1.4 KB per token), in VRAM: 12.4 KB per token, 1.7 GB at 131,072 tokens and 13.3 GB at 1M
  (calculated). For each new token the indexer picks the 512 pools (2,048 tokens) that matter most, and the pool
  still being filled. So past about 2,000 tokens of context each new token attends to a fixed amount of the text,
  not all of it. Below that the indexer picks every pool and attention is plain causal.

The context (`--ctx`, up to 1,048,576 tokens) is the longest conversation the engine accepts. The engine sets its
MLA cache aside for the whole context when it starts, out of the VRAM that the expert cache would otherwise use.

## Several clients

Agents, a chat and a bot can share one engine. Requests wait in the server and run one at a time, in the order
they arrived. The KDA state is a running sum that cannot be rewound, so the engine keeps each conversation's state
instead:

- **Parking.** When a request starts a different conversation, the current one's whole state is copied to RAM. A
  later request that continues it copies it back and reads only the new tokens.
- **Checkpoints.** The server marks where each reply begins. When the client sends the reply back with its thinking,
  the model's template writes it back with that thinking, and the engine goes on from the end of the reply. When the
  next turn writes it otherwise (a client that does not send the thinking back), it differs from the engine's state
  from that point on. The engine then restarts from the checkpoint and reads the reply and the new message, not the
  whole conversation. A regenerated reply restarts there too.
- **Memory.** `--park-mib` caps the RAM that parked conversations and checkpoints use: 8192 MiB by default, 4096
  when setup finds less than 40 GiB of RAM left beside the experts (the 3-bit model on 128 GB, the 4-bit one on
  160 GB). The conversation used least recently goes first. Parking lives in the engine's RAM, so a restart of the
  engine starts with none.

None of this changes the answers: on the CPU engine, `tools/glm53_park_check.py` checks each resumed reply against
a fresh engine's, token for token.

## How it is checked

What has run without a GPU:

- **The CPU reference** (`strata-glm-ref`) runs the whole model on the CPU, one token at a time, with ggml's own
  dot products. On small random GLM-5.3-Flash models it matches llama.cpp to float rounding
  (`tools/glm53_selftest.sh`, which needs no GPU and no download; on napster-ai too: every model passes, KL 0).
- **Conversation parking** on the CPU engine: `tools/glm53_park_check.py` compares every resumed reply with a fresh
  engine's.
- **The server end to end** on the CPU engine with a small random model (`tools/glm53_server_check.sh`), about 750
  Python tests of the server (`python3 -m unittest discover -s serve -t . -p "test_*.py"`), setup's and the tools'
  tests (`tools/test_*.py`) and the engine's C++ tests (`ctest`).

What has run on napster-ai's three cards (2026-10-20), on the same small random models:

- **The GPU engine** (`strata-glm`) against the CPU reference layer by layer (`--check-cpu`) and against llama.cpp
  (`tools/glm53_gpu_check.sh`): its decode and its prompt path matched both, within the rounding of the GPU's 8-bit
  activations that llama.cpp's own GPU shows too, on one card and with the layers split over two of the three.
- **Conversation parking** on the GPU engine, with the layers split over two cards: passed.

What waits for more RAM on napster-ai:

- **The real model:** its logits compared with llama.cpp's (`tools/glm53_compare.py`), then its speed, load time
  and memory.
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
