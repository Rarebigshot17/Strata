# AGENTS.md

Strata runs one model, GLM-5.3-Flash, on Linux with NVIDIA graphics cards and system RAM: every routed expert stays
in RAM, the most used ones are cached in VRAM, and the CPU computes the rest. It has a C++/CUDA engine (`strata-glm`;
the CPU reference `strata-glm-ref`), a Python server with OpenAI-, Anthropic- and Responses-compatible APIs and a
web app (`serve/`), and an installer (`setup.py`, started by `./setup.sh`). It is a fork of
[Niko1221/Strata](https://github.com/Niko1221/Strata), which ran Qwen3.8-Flash-Next. The Qwen engine, AMD (HIP),
the Windows launchers, vision, batching, replicas, MTP drafting, the speed projection and slot save/restore are gone.

**The real model has not run yet (it waits for more RAM on napster-ai), so there is no speed, load time or memory
measurement.** What has run, on napster-ai on 2026-10-20 (RTX 3090 Ti 24 GB + RTX 3080 Ti 12 GB + RTX 3080 10 GB,
sm_86, driver 580.95.05, CUDA 13.2, gcc 14.2): the builds; the CPU reference against llama.cpp on small random models (all PASS, KL 0); the GPU check on
the same models (decode matched the CPU reference 40/40 and llama.cpp with KL 0, on one card and split over the
three; conversation parking passed; the prefill check failed on a bug in the check, fixed since, not yet rerun); and
the inspector and the template probe on the real file. Do not write a speed, load time or memory figure as measured.
A figure calculated from the model's shapes says that it is calculated; one derived from another says so.

## Installing Strata for a user

Follow **[docs/AI_SETUP.md](docs/AI_SETUP.md)**: check the PC, pick the size by RAM, run setup without questions,
start and check the server, and connect the user's apps. Never let the server listen beyond `127.0.0.1` without an
API key (`--api-key`). The server's port is 8101.

## Where things are

| Path | What |
| --- | --- |
| `src/glm/`, `include/strata/glm/` | The engine. `glm_main.cpp` is `strata-glm` (CUDA: `engine*.cu`, `gpu_*.cu`, `expert_cache.cu`); `ref_main.cpp` and `cpu_ref.cpp` are `strata-glm-ref`, the whole model on the CPU. `serve_loop.cpp` speaks the engine protocol (parking, checkpoints); `sampler.cpp` samples. |
| `src/kernels/`, `include/strata/kernels/`, `include/strata/artifact/` | The quantized GEMV kernels and the GGUF reader the engine shares. |
| `tests/glm/` | The engine's C++ unit tests (sampler, serve loop, the GPU checks' pass rule): no GPU, no model file. |
| `serve/` | The server: `server.py` (HTTP, the engine driver, the queue), `frontend.py` (chat template, output parser for thinking and tool calls), `responses.py`, `mcp.py`, `structured.py`, `runconfig.py`, `prometheus.py`, `web/` (the web app), and the tests `test_*.py`. |
| `setup.py`, `setup.sh`, `update.sh` | The installer, its launcher and the updater. |
| `tools/` | The GLM checks (`glm53_*`), the tests of setup and of the tools (`test_*.py`), `strata_bench.py` (`./bench.sh`), and helpers. |
| `docs/` | The user docs, and the port's record: [GLM53_PORT.md](docs/GLM53_PORT.md) (the engine, and the steps to run on the owner's machine) and [GLM53_SERVER_PORT.md](docs/GLM53_SERVER_PORT.md) (the server's decisions, the engine protocol, the plan's state). |

How it works: [docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md). The API and every setting:
[docs/DETAILS.md](docs/DETAILS.md#the-api).

## Building and testing

None of these needs a GPU or a model download, except where it says so.

- **Python tests** (the server and its modules; about 750, a few minutes):

  ```bash
  python3 -m unittest discover -s serve -t . -p "test_*.py"
  ```

- **Setup's and the tools' tests:** each file on its own, for example `python3 tools/test_setup_glm.py`:

  ```bash
  for t in tools/test_*.py; do python3 "$t" || echo "FAILED: $t"; done
  ```

- **The engine and its C++ tests.** Without CUDA, CMake builds `strata-glm-ref` and the unit tests. ggml comes from
  a llama.cpp checkout (`-DSTRATA_GGML_DIR=...`) or is fetched at the pinned commit.

  ```bash
  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j
  ctest --test-dir build
  ```

  With CUDA (needs the CUDA toolkit; napster-ai's cards are compute capability 8.6):

  ```bash
  cmake -B build -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j --target strata-glm strata-glm-ref
  ```

- **The GLM checks** compare Strata with llama.cpp on small random GLM-5.3-Flash models. They need llama.cpp built
  by `tools/glm53_ref/build.sh` (`NO_CUDA=1` builds it without CUDA) and `PYTHONPATH=<llama.cpp>/gguf-py`. Each
  script's header says what it takes.
  - `tools/glm53_selftest.sh LLAMA_BUILD_DIR STRATA_BUILD_DIR`: the CPU reference against llama.cpp.
  - `tools/glm53_park_check.py ENGINE -m MODEL.gguf`: conversation parking and checkpoints against fresh runs.
  - `tools/glm53_server_check.sh LLAMA_DIR ENGINE`: `serve/server.py` end to end on the engine.
  - `tools/glm53_gpu_check.sh LLAMA_BUILD_DIR STRATA_BUILD_DIR`: the GPU engine. **Needs an NVIDIA GPU.** It ran
    on napster-ai on 2026-10-20: decode and parking passed; the prefill part failed on a bug in the check, fixed
    since and not yet rerun.
- **The server without an engine:** `python3 serve/server.py` starts it with the mock engine (a fixed answer) on
  port 8101. `--engine strata --config strata-glm-<size>.json` runs the real one.

Keep the engine protocol section of [docs/GLM53_SERVER_PORT.md](docs/GLM53_SERVER_PORT.md) in step with
`src/glm/serve_loop.cpp` and `serve/server.py` when you change either.

## The docs' style

Plain words and short sentences. A measured number comes with what it was measured on. A calculated number says
that it is calculated. No claim without a measurement or the code behind it. Markdown, lines of about 120
characters.

## Changing the code

This is the owner's personal project: [Rarebigshot17/Strata](https://github.com/Rarebigshot17/Strata), branch
`glm53-flash`.

- Run the Python tests, the tools' tests and `ctest` after a change. A change to the engine also builds `strata-glm`
  with CUDA, and runs the GLM checks it touches.
- Change a default only where you measured it better, and give the numbers with what they were measured on.
- A result from a real run says what ran, on which machine, and what was not tested.
