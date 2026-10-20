#!/usr/bin/env bash
# The GLM-5.3-Flash engine against llama.cpp on small random models: no download, no GPU, about a minute.
#
#   tools/glm53_selftest.sh LLAMA_BUILD_DIR STRATA_BUILD_DIR
#
# LLAMA_BUILD_DIR is a llama.cpp build with test-llama-archs, llama-quantize and llama-strata-ref
# (tools/glm53_ref/build.sh, then `cmake -B build -DLLAMA_BUILD_TESTS=ON && cmake --build build --target
# test-llama-archs`). STRATA_BUILD_DIR holds strata-glm-ref. Needs gguf-py: PYTHONPATH=<llama.cpp>/gguf-py.
#
# Models checked, each over 40 positions:
#   base     llama.cpp's random glm5-next model (MLA + dense FFN, KDA + MoE; indexer top-k beyond the prompt)
#   v1       + SwiGLU clamps, expert weight norm and x2.5 scale, 20 Sinkhorn iterations, indexer top-k 8
#            (from position 11 the indexer drops pools)
#   v2       v1 with the softplus decay gate instead of the lower-bound one
#   v1-bare  v1 without attention.indexer.kpool_select_tail and attention.indexer.types, as the released GGUF
#            ships: both engines must take llama.cpp's defaults (select the tail, every layer a full indexer)
#   Q4_0     v1 quantized: quantized activations through ggml-cpu's dot products
# llama.cpp runs with batch 1 and flash attention off, the engine's own order of operations.
set -euo pipefail
# the GPUs hidden: a CUDA build of llama.cpp would take them, and stop when another program has filled them
export CUDA_VISIBLE_DEVICES=

LB="${1:?llama.cpp build dir}"
SB="${2:?strata build dir}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT

"$LB/bin/test-llama-archs" -a glm5-next -o "$W/" > /dev/null
python3 "$HERE/glm53_tiny_variant.py" "$W/glm5-next-moe.gguf" "$W/v1.gguf" \
    glm5-next.attention.indexer.top_k=8 glm5-next.swiglu_clamp_exp=0.05 glm5-next.swiglu_clamp_shexp=0.03 \
    glm5-next.expert_weights_scale=2.5 glm5-next.expert_weights_norm=true \
    glm5-next.hyper_connection.sinkhorn_iterations=20 > /dev/null
python3 "$HERE/glm53_tiny_variant.py" "$W/v1.gguf" "$W/v2.gguf" glm5-next.kda.gate_lower_bound=-inf > /dev/null
python3 "$HERE/glm53_tiny_variant.py" "$W/v1.gguf" "$W/v1-bare.gguf" \
    -glm5-next.attention.indexer.kpool_select_tail -glm5-next.attention.indexer.types > /dev/null
"$LB/bin/llama-quantize" --allow-requantize "$W/v1.gguf" "$W/q4_0.gguf" Q4_0 > /dev/null 2>&1
python3 -c "import random; random.seed(1); print(','.join(str(random.randrange(128)) for _ in range(40)))" > "$W/ids.txt"

fail=0
for m in glm5-next-moe v1 v2 v1-bare q4_0; do
    "$LB/bin/llama-strata-ref" -m "$W/$m.gguf" --tokens-file "$W/ids.txt" -c 256 -b 1 -ub 1 -fa off \
        --out "$W/ref_$m.bin" > /dev/null 2>&1
    "$SB/strata-glm-ref" -m "$W/$m.gguf" --tokens-file "$W/ids.txt" --out "$W/glm_$m.bin" -t 4 > /dev/null
    if out=$(python3 "$HERE/glm53_compare.py" "$W/ref_$m.bin" "$W/glm_$m.bin" --kl-max 1e-6 --top1-min 1.0); then
        echo "PASS $m: $(echo "$out" | tail -2 | head -1)"
    else
        echo "FAIL $m:"; echo "$out" | tail -12; fail=1
    fi
done
exit $fail
