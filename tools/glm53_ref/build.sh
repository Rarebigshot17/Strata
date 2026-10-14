#!/usr/bin/env bash
# Build llama.cpp (CUDA) plus the reference-logits tool for the GLM-5.3-Flash port.
#
#   tools/glm53_ref/build.sh [LLAMA_DIR]      default LLAMA_DIR: ../llama.cpp next to the Strata checkout
#
# llama.cpp is pinned to the commit the tool was written against, because its batch API is still changing.
# Set LLAMA_COMMIT to try another one, and NO_CUDA=1 for a CPU-only build.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="${1:-$HERE/../../../llama.cpp}"
LLAMA_COMMIT="${LLAMA_COMMIT:-a0e6dc7384edb8f5d927155e1d26c10b4a310e6d}"

if [ ! -d "$LLAMA_DIR/.git" ]; then
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
fi
cd "$LLAMA_DIR"
git fetch --depth 1 origin "$LLAMA_COMMIT" 2>/dev/null || git fetch origin
git checkout --quiet "$LLAMA_COMMIT"

mkdir -p examples/strata-ref
cp "$HERE/ref_logits.cpp" "$HERE/CMakeLists.txt" examples/strata-ref/
grep -q "add_subdirectory(strata-ref)" examples/CMakeLists.txt ||
    sed -i 's/^\( *\)add_subdirectory(debug)$/&\n\1add_subdirectory(strata-ref)/' examples/CMakeLists.txt
grep -q "add_subdirectory(strata-ref)" examples/CMakeLists.txt || { echo "could not register examples/strata-ref"; exit 1; }

CUDA_FLAG="-DGGML_CUDA=ON"
[ "${NO_CUDA:-0}" = "1" ] && CUDA_FLAG="-DGGML_CUDA=OFF"
cmake -B build -DCMAKE_BUILD_TYPE=Release $CUDA_FLAG
cmake --build build -j"$(nproc)" --target llama-strata-ref llama-cli llama-server llama-quantize

echo
echo "built: $LLAMA_DIR/build/bin/llama-strata-ref (llama.cpp $(git rev-parse --short HEAD))"
