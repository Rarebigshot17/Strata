#!/bin/sh
# Entry point for the Strata container. The engine is compiled during docker
# build and lives in the image, so the first start only downloads the model.
# The install config is kept on the /data volume so a recreated container skips
# the setup pass and goes straight to serving.
set -e
cd /opt/strata || exit 1

# #1584: the engine in the image is built with AVX2 CPU kernels. A host CPU without AVX2 (Sandy / Ivy Bridge Xeons,
# Bulldozer) runs it into an illegal instruction. A warning, not a stop: recommend, never force (the flags file
# can be wrong in a VM, and a build made on that host may be fine). STRATA_CPUINFO is for the test.
cpuinfo="${STRATA_CPUINFO:-/proc/cpuinfo}"
if [ -r "$cpuinfo" ] && grep -q '^flags' "$cpuinfo" && ! grep -m1 '^flags' "$cpuinfo" | grep -qw avx2; then
  echo "WARNING: this host CPU has no AVX2. The engine in this image is built for CPUs with AVX2, so it may stop" >&2
  echo "  with 'illegal instruction' (exit 132). Build the image on this machine (docker build) with" >&2
  echo "  STRATA_PORTABLE off in the Dockerfile, which compiles the engine for the CPU it runs on." >&2
fi

STRATA_DATA="${STRATA_DATA:-/data}"
MODEL="${MODEL:-UD-IQ3_XXS}"    # UD-IQ3_XXS (3-bit, ~120 GB) | IQ4_XS (4-bit, ~170 GB)
CONTEXT="${CONTEXT:-}"          # tokens; empty: setup.py's pick for the GPUs' VRAM
PARK_MIB="${PARK_MIB:-}"        # RAM for parked conversations; empty: setup.py's pick for the RAM left
HOST="${HOST:-0.0.0.0}"          # every interface of the container: what a published port (-p) reaches
PORT="${PORT:-8101}"
API_KEY="${API_KEY:-}"          # passed on at every start, so a key added later counts without REINSTALL
GPUS="${GPUS:-}"                # "0,2" or "all": the cards the model's layers are shared across
GPU="${GPU:-}"                  # one card, numbered as nvidia-smi numbers them
GGUF_DIR="${GGUF_DIR:-}"        # a mounted folder with GGUF files you already have: no download
CONFIG="${CONFIG:-}"            # a config file to start with (wins over MODEL's /data/config/strata-glm-<model>.json)

case "$MODEL" in
  UD-IQ3_XXS) size="~120 GB" ;;
  IQ4_XS) size="~170 GB" ;;
  *) echo "MODEL=$MODEL: use UD-IQ3_XXS (3-bit) or IQ4_XS (4-bit)." >&2; exit 1 ;;
esac

# setup.py starts the strata-glm-*.json it finds (and asks which one when it finds
# several), so link in exactly the one this model was set up with. The config is the recorded output of that setup (the
# shards, the context, the parking RAM), not settings the entry point could rebuild
# from env vars. Its name is setup.py's: strata-glm-<model in lower case>.json.
tag="glm-$(printf '%s' "$MODEL" | tr 'A-Z' 'a-z')"
cfg="$STRATA_DATA/config/strata-$tag.json"
mkdir -p "$STRATA_DATA/config"

# REINSTALL is only needed to change settings for a model that is already set up
# (context, parking RAM, host). Switching between models already on the volume
# needs no setup pass: their config is already there.
#
# CONTEXT / PARK_MIB / GPUS / GPU are passed only when set, so an unset one keeps
# setup.py's own pick. The browser is never opened from a container.
#
# Which config the server starts with (#1244): CONFIG when set; else a link in /opt/strata that already points
# into $STRATA_DATA/config/ (a pod command that picked one by linking); else MODEL's strata-glm-<model>.json.
link="/opt/strata/strata-$tag.json"
keep=""
if [ -n "$CONFIG" ]; then
  [ -f "$CONFIG" ] || { echo "CONFIG=$CONFIG does not exist." >&2; exit 1; }
elif [ "${REINSTALL:-0}" != "1" ] && [ -L "$link" ] && [ -f "$link" ]; then
  case "$(readlink "$link")" in "$STRATA_DATA"/config/*) keep=1 ;; esac
fi

if [ -n "$CONFIG" ]; then
  ln -sfn "$CONFIG" "$link"
  echo "Config: $CONFIG (from CONFIG)"
elif [ -n "$keep" ]; then
  echo "Config: $(readlink "$link") (existing link kept)"
elif [ "${REINSTALL:-0}" = "1" ] || [ ! -f "$cfg" ]; then
  if [ -n "$GGUF_DIR" ]; then
    echo "Setting up $tag from the GGUF files in $GGUF_DIR (the engine is already in the image)."
  else
    echo "Setting up $tag: downloading the model ($size; the engine is already in the image)."
  fi
  set -- --model "$MODEL" --data-dir "$STRATA_DATA" --host "$HOST" --api-key "$API_KEY" \
    --port "$PORT" --no-browser --no-start
  if [ -n "$CONTEXT" ]; then set -- "$@" --context "$CONTEXT"; fi
  if [ -n "$PARK_MIB" ]; then set -- "$@" --park-mib "$PARK_MIB"; fi
  if [ -n "$GPUS" ]; then set -- "$@" --gpus "$GPUS"; fi
  if [ -n "$GPU" ]; then set -- "$@" --gpu "$GPU"; fi
  if [ -n "$GGUF_DIR" ]; then set -- "$@" --gguf-dir "$GGUF_DIR"; fi
  .venv/bin/python setup.py --setup --yes "$@"
  # setup writes through a link left by an earlier start (REINSTALL=1), else a file of its own here: the volume
  # gets that file, and the start below goes through the link to it
  [ -e "$link" ] && { cmp -s "$link" "$cfg" || cp -f "$link" "$cfg"; }
  ln -sfn "$cfg" "$link"
  echo "Config: $cfg (from MODEL $MODEL, just set up)"
else
  # #1244: the copy on the volume is the one that counts, so a regular file left in /opt/strata by an earlier setup
  # (or by an image built with one) must not stand in for it: edits to /data/config would be ignored
  ln -sfn "$cfg" "$link"
  echo "Config: $cfg (from MODEL $MODEL)"
fi

# No API key while the server listens beyond the container's loopback: whoever reaches the published port can use
# the model. -p 8101:8101 publishes it on every interface of the host (Docker's rules pass by ufw), and --network
# host listens on them directly. Not a stop (a port published on 127.0.0.1 only is safe without a key, and the
# container cannot see how it is published), but said at every start. The config decides: its "host" and "api_key".
open_without_key() {
  python3 - "$1" <<'PY'
import json, sys
try:
    cfg = json.load(open(sys.argv[1], encoding="utf-8-sig"))
except (OSError, ValueError):
    sys.exit(0)
key, host = cfg.get("api_key"), str(cfg.get("host") or "127.0.0.1")
sys.exit(1 if (key.strip() if isinstance(key, str) else key) or host in ("127.0.0.1", "localhost", "::1") else 0)
PY
}
# STRATA_API_KEY is the server's own variable for the key (the server inherits this environment).
if [ -z "$API_KEY" ] && [ -z "${STRATA_API_KEY:-}" ] && open_without_key "$link"; then
  echo "WARNING: no API key. The server listens on every interface of the container, so anyone who reaches its" >&2
  echo "  published port can use the model. Publish it on this host only (-p 127.0.0.1:$PORT:$PORT), or set" >&2
  echo "  -e API_KEY=<secret> before publishing it to a network: -p $PORT:$PORT publishes it on every interface" >&2
  echo "  of the host, past ufw, and --network host listens on them directly." >&2
fi

# Later starts skip straight here: setup.py finds the installed config and
# launches serve/server.py (OpenAI- and Anthropic-compatible API on :8101).
# GPUS / GPU are repeated on purpose: given at the start they pin the cards
# (GPUS is saved in the model's config, GPU is for this start only). API_KEY too:
# a key given at a start is saved in the config, so one added later counts.
set -- --port "$PORT"
if [ -n "$GPUS" ]; then set -- "$@" --gpus "$GPUS"; fi
if [ -n "$GPU" ]; then set -- "$@" --gpu "$GPU"; fi
if [ -n "$API_KEY" ]; then set -- "$@" --api-key "$API_KEY"; fi
exec .venv/bin/python setup.py "$@"
