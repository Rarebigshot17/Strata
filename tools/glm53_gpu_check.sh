#!/usr/bin/env bash
# The GLM-5.3-Flash GPU engine against the CPU reference and against llama.cpp, on small random models.
# Needs an NVIDIA GPU with 2 GiB free on each card it runs on (it stops first when one has less); no download; about
# a minute.
#
#   tools/glm53_gpu_check.sh LLAMA_BUILD_DIR STRATA_BUILD_DIR
#
# LLAMA_BUILD_DIR: from tools/glm53_ref/build.sh (llama-strata-ref, llama-quantize, test-llama-archs).
# STRATA_BUILD_DIR: a CUDA build of this repo with strata-glm and strata-glm-ref.
# Needs gguf-py: PYTHONPATH=<llama.cpp>/gguf-py (or pip install gguf).
#
# For each model: strata-glm --check-cpu (GPU vs CPU reference, layer by layer), the prompt path (--prefill, in
# chunks) against the CPU reference, then the GPU's logits against llama.cpp's. The GPU quantizes activations to
# q8_1 like llama.cpp's CUDA backend, so on quantized weights the numbers differ from the CPU's by more than float
# rounding; the thresholds allow for that. Last, conversation parking (tools/glm53_park_check.py).
set -uo pipefail

LB="${1:?llama.cpp build dir}"
SB="${2:?strata build dir}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

# The free memory of every card the check runs on, before anything starts: when another program holds a card, every
# engine below stops at its first CUDA call. The tiny models take about 1.2 GB per card (napster-ai, 2026-10-20:
# "GPU 0: ... 1.21 GB weights and buffers") and the CUDA context a few hundred MB more, so 2 GiB. nvidia-smi numbers
# the cards by PCI bus; when CUDA_VISIBLE_DEVICES names some, CUDA is told to number them the same way.
MIN_FREE_MIB=2048
smi=(--query-gpu=index,memory.free --format=csv,noheader,nounits)
if [ -n "${CUDA_VISIBLE_DEVICES+set}" ]; then
    [ -n "$CUDA_VISIBLE_DEVICES" ] || { echo "CUDA_VISIBLE_DEVICES is empty, so no card is visible: unset it"; exit 1; }
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    smi=(-i "$CUDA_VISIBLE_DEVICES" "${smi[@]}")
fi
if free=$(nvidia-smi "${smi[@]}" 2>&1); then
    full=0
    while IFS=', ' read -r idx mib; do
        [[ "$mib" =~ ^[0-9]+$ ]] || continue
        if (( mib < MIN_FREE_MIB )); then
            echo "GPU $idx has $mib MiB free, the check needs $MIN_FREE_MIB: stop the program that holds it" \
                "(nvidia-smi lists it), or leave the card out with CUDA_VISIBLE_DEVICES"
            full=1
        fi
    done <<< "$free"
    [ "$full" = 0 ] || exit 1
else
    echo "nvidia-smi cannot read the cards' free memory, checking anyway: $(echo "$free" | head -1)"
fi

"$LB/bin/test-llama-archs" -a glm5-next -o "$W/" > /dev/null 2>&1 || { echo "test-llama-archs failed"; exit 1; }
python3 "$HERE/glm53_tiny_variant.py" "$W/glm5-next-moe.gguf" "$W/v1.gguf" \
    glm5-next.attention.indexer.top_k=8 glm5-next.swiglu_clamp_exp=0.05 glm5-next.swiglu_clamp_shexp=0.03 \
    glm5-next.expert_weights_scale=2.5 glm5-next.expert_weights_norm=true \
    glm5-next.hyper_connection.sinkhorn_iterations=20 > /dev/null || exit 1
# v1 without the keys the released GGUF does not carry (glm53_selftest.sh): the engine must take llama.cpp's defaults
python3 "$HERE/glm53_tiny_variant.py" "$W/v1.gguf" "$W/v1-bare.gguf" \
    -glm5-next.attention.indexer.kpool_select_tail -glm5-next.attention.indexer.types > /dev/null || exit 1
"$LB/bin/llama-quantize" --allow-requantize "$W/v1.gguf" "$W/q4_0.gguf" Q4_0 > /dev/null 2>&1
"$LB/bin/llama-quantize" --allow-requantize "$W/v1.gguf" "$W/q8_0.gguf" Q8_0 > /dev/null 2>&1
python3 -c "import random; random.seed(1); print(','.join(str(random.randrange(128)) for _ in range(40)))" > "$W/ids.txt"

fail=0
for m in glm5-next-moe v1 v1-bare q4_0 q8_0; do
    echo "== $m"
    if ! "$SB/strata-glm" -m "$W/$m.gguf" --tokens-file "$W/ids.txt" --out "$W/gpu_$m.bin" --check-cpu > "$W/log_$m.txt" 2>&1; then
        fail=1
    fi
    grep -E "GPU |layer .* differs|check-cpu|strata-glm:|CUDA" "$W/log_$m.txt" | head -12
    # the prompt in chunks of 16 (crossing pool boundaries) against the CPU reference, then 4 decoded tokens
    if "$SB/strata-glm" -m "$W/$m.gguf" --tokens-file "$W/ids.txt" --prefill --prefill-chunk 16 --check-cpu \
            > "$W/pp_$m.txt" 2>&1; then
        echo "prefill: $(grep check-prefill "$W/pp_$m.txt")"
    else
        echo "prefill FAILED:"; tail -8 "$W/pp_$m.txt"; fail=1
    fi
    "$LB/bin/llama-strata-ref" -m "$W/$m.gguf" --tokens-file "$W/ids.txt" -c 256 -b 64 -ub 64 \
        --out "$W/ref_$m.bin" > "$W/ref_$m.txt" 2>&1
    if [ ! -s "$W/ref_$m.bin" ]; then
        echo "FAIL vs llama.cpp: llama-strata-ref wrote no logits:"; tail -4 "$W/ref_$m.txt"; fail=1
    elif [ ! -s "$W/gpu_$m.bin" ]; then
        echo "FAIL vs llama.cpp: strata-glm wrote no logits (its lines above)"; fail=1
    else
        res=$(python3 "$HERE/glm53_compare.py" "$W/ref_$m.bin" "$W/gpu_$m.bin") || fail=1
        echo "vs llama.cpp: $(echo "$res" | tail -2 | tr '\n' ' ')"
    fi
done
# conversations parked in host memory and restored (over every GPU the layers are split across), against fresh
# runs; one token at a time and without the expert cache, so that both are bit for bit the same computation
echo "== conversation parking"
if ! python3 "$HERE/glm53_park_check.py" "$SB/strata-glm" -m "$W/v1.gguf" --prefill-chunk 0 --no-expert-cache \
        > "$W/park.txt" 2>&1; then
    fail=1
    grep -E "FAIL|error|exited|died" "$W/park.txt" | head -12
fi
tail -1 "$W/park.txt"
exit $fail
