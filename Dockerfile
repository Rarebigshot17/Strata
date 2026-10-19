# syntax=docker/dockerfile:1
#
# Strata: GLM-5.3-Flash on NVIDIA GPUs (compute capability 7.5 or newer: RTX 20 /
# 30 / 40 / 50) and system RAM. Every expert stays in RAM (~117 GB for the 3-bit
# UD-IQ3_XXS, ~162 GB for the 4-bit IQ4_XS); the GPUs hold the rest of the model
# and cache the most used experts. Several cards share one model.
#
# The engine (strata-glm, and the CPU reference strata-glm-ref) is compiled during
# docker build, so the first container start only downloads the model (~120 GB or
# ~170 GB) and starts the server. docker build has no GPU, so the CUDA
# architectures are fixed here instead of read from nvidia-smi: the engine is a fat
# binary with a cubin per listed arch, and the runtime picks the one matching your
# card. Narrow CUDA_ARCHITECTURES to your cards for a faster build (86 for RTX 30);
# a card outside the set is compiled for at its first start.
#
# Build:
#   docker build -t strata .
#   docker build -t strata --build-arg CUDA_ARCHITECTURES=86 .        # RTX 30 only
#
# Run (host needs an NVIDIA driver >= 580 and nvidia-container-toolkit):
#   docker run --rm --gpus all \
#     -p 8101:8101 \
#     --ulimit memlock=-1 \
#     -v strata-data:/data \
#     -e MODEL=UD-IQ3_XXS \
#     strata
#
# Setup choices are env vars, read by docker-entrypoint.sh: MODEL (UD-IQ3_XXS or
# IQ4_XS), CONTEXT (tokens; empty: setup's pick for the GPUs' VRAM), PARK_MIB,
# GPU (one card) or GPUS ("0,2" or "all"), HOST, PORT (8101), API_KEY, GGUF_DIR
# (GGUF files you already have), CONFIG (a config file to start with).
#
# Only the model files and the install config live in the /data volume (it needs
# room for 120-170 GB); the engine is part of the image. Add an API key before
# exposing the port to a network: -e API_KEY=<secret>. Pass -e REINSTALL=1 to
# change the model settings later.

FROM nvidia/cuda:13.0.0-devel-ubuntu24.04

# setup.py replaces itself with the server (setup.exec_server), so the server is PID 1
# and docker stop's SIGTERM reaches it.
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git libatomic1 libgomp1 \
        python3 python3-pip python3-venv unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/strata
COPY . .

# RTX 20 (75), RTX 30 (86), RTX 40 (89), RTX 50 (120), plus 80 for A-series. CMakeLists
# refuses anything below 75.
ARG CUDA_ARCHITECTURES=75;80;86;89;120

RUN python3 -m venv .venv \
    && .venv/bin/pip install --no-cache-dir --upgrade pip \
    && .venv/bin/pip install --no-cache-dir -r requirements.txt \
    && chmod +x setup.sh docker-entrypoint.sh

# llama.cpp at the pinned commit, then the engine, built the way setup.py builds it
# into build/, except that the CPU code is for any AVX2 CPU (STRATA_PORTABLE) rather
# than for the one that builds the image. build/BUILD.json is what setup.py reads to
# decide whether the engine is current: a matching src hash and every GPU's arch in
# its list mean the first start uses it instead of compiling it again. The packages
# are recorded the way setup.py records them, so its first run installs none.
RUN PIP_NO_CACHE_DIR=1 .venv/bin/python - <<'PYEOF'
import os
import setup

setup.python_packages()
llama = setup.get_llama_cpp()
nvcc, _ = setup.find_nvcc()
text = os.environ.get("CUDA_ARCHITECTURES", "75;80;86;89;120").strip().strip('"').replace(",", ";")
archs = [int(a.split("-")[0]) for a in text.split(";") if a.split("-")[0].isdigit()]

setup.cmake_build(setup.ROOT, setup.BUILD_DIR, setup.TARGETS,
                  setup.build_defs(archs, nvcc, llama) + ["-DSTRATA_PORTABLE=ON"])
setup.write_build_stamp(archs, nvcc)
PYEOF

# the cmake tree is build-time only: the two executables and BUILD.json are what the container needs
RUN find build -mindepth 1 -maxdepth 1 ! -name strata-glm ! -name strata-glm-ref ! -name BUILD.json \
        -exec rm -rf {} + \
    && test -x build/strata-glm

VOLUME ["/data"]
EXPOSE 8101

# /health is answered before the API key gate, so it works with or without one.
# The port only opens after the model loads (minutes: 120-170 GB are read into RAM,
# longer on a first run, which downloads them), so the start period is generous: a
# too short one marks a still-loading container unhealthy and a restart policy would
# kill it mid-download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=1800s --retries=3 \
  CMD curl -fs "http://127.0.0.1:${PORT:-8101}/health" || exit 1

ENTRYPOINT ["./docker-entrypoint.sh"]
