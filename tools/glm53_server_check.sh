#!/usr/bin/env bash
# serve/server.py end to end on strata-glm, with a small random GLM-5.3-Flash that has a real GLM-style tokenizer and
# chat template.
#
#   tools/glm53_server_check.sh LLAMA_DIR ENGINE
#
# LLAMA_DIR: the llama.cpp checkout from tools/glm53_ref/build.sh (its build/ has test-llama-archs and
# llama-tokenize; its models/templates has GLM's template; its gguf-py is used). ENGINE: strata-glm (GPU) or
# strata-glm-ref (CPU). The server starts the engine from a run config written here (-m the synthetic model,
# --ctx 512), reads the tokenizer and the chat template from the model's GGUF, and is checked through its API: the
# tokenizer against llama.cpp's, then /health, /v1/models, a chat completion, a streamed one, a second turn that
# resumes from the first one's checkpoint (cp=), a completion, /metrics, a slot save (501) and a request that does
# not fit the context.
set -uo pipefail
LD="${1:?llama.cpp dir}"
ENGINE="$(cd "$(dirname "${2:?strata-glm or strata-glm-ref}")" && pwd)/$(basename "$2")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
W="$(mktemp -d)"
PORT=18199
export PYTHONPATH="$LD/gguf-py${PYTHONPATH:+:$PYTHONPATH}"
cleanup() { [ -n "${SRV:-}" ] && kill "$SRV" 2>/dev/null && wait "$SRV" 2>/dev/null; rm -rf "$W"; }
trap cleanup EXIT

"$LD/build/bin/test-llama-archs" -a glm5-next -o "$W/" > /dev/null 2>&1 || { echo "FAIL test-llama-archs"; exit 1; }
python3 "$HERE/glm53_synth_model.py" "$W/glm5-next-moe.gguf" "$LD/models/templates/GLM-4.7-Flash.jinja" \
    "$W/synth.gguf" > /dev/null || exit 1
echo "tokenizer: $(python3 "$HERE/glm53_tokcheck.py" "$W/synth.gguf" "$LD/build/bin/llama-tokenize" | tail -1)"

# --checkpoint-min 1: the engine keeps a checkpoint even this close to the start, so the short second turn below
# can resume from the first one's (the default, 256 tokens, is for real conversations)
python3 - "$ENGINE" "$W" > "$W/strata-check.json" <<'EOF'
import json, sys
exe, w = sys.argv[1], sys.argv[2]
print(json.dumps({"exe": exe, "args": ["-m", f"{w}/synth.gguf", "--ctx", "512", "--checkpoint-min", "1"],
                  "log": f"{w}/engine.log"}))
EOF

cd "$ROOT"
python3 -m serve.server --engine strata --config "$W/strata-check.json" --port $PORT > "$W/server.log" 2>&1 &
SRV=$!
for _ in $(seq 1 240); do
    curl -sf localhost:$PORT/health > /dev/null && break
    kill -0 "$SRV" 2>/dev/null || break
    sleep 0.5
done
fail=0
check() {   # name, command, python condition on the JSON (or the text when 'raw')
    local name="$1" out
    out=$(eval "$2")
    if python3 -c "import json,sys; d=sys.stdin.read(); d=d if '$4'=='raw' else json.loads(d); sys.exit(0 if ($3) else 1)" <<< "$out" 2> /dev/null; then
        echo "PASS $name"
    else
        echo "FAIL $name: ${out:0:300}"; fail=1
    fi
}
U=localhost:$PORT
J='-H Content-Type:application/json'
check health "curl -s $U/health" "d['status']=='ok' and d['max_context']==512" json
check models "curl -s $U/v1/models" "d['data'][0]['id']" json
check chat "curl -s $U/v1/chat/completions $J -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":8,\"temperature\":0}'" \
    "d['usage']['completion_tokens']==8 and d['choices'][0]['finish_reason']=='length'" json
check stream "curl -sN $U/v1/chat/completions $J -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":5,\"stream\":true}'" \
    "d.count('data: ')>=3 and d.rstrip().endswith('data: [DONE]')" raw
check turn1 "curl -s $U/v1/chat/completions $J -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}],\"max_tokens\":4}'" \
    "d['usage']['completion_tokens']>=1" json
check turn2-resumes "curl -s $U/v1/chat/completions $J -d '{\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"},{\"role\":\"assistant\",\"content\":\"Hi.\"},{\"role\":\"user\",\"content\":\"And again\"}],\"max_tokens\":4}'" \
    "d['usage']['prompt_tokens_details']['cached_tokens']>0" json
check completion "curl -s $U/v1/completions $J -d '{\"prompt\":\"The quick\",\"max_tokens\":4,\"temperature\":0}'" \
    "d['usage']['completion_tokens']==4 and d['object']=='text_completion'" json
check metrics "curl -s $U/metrics" \
    "d['engine']['backend']=='cpu-reference' or d['engine']['backend']=='cuda'" json
check slot-save "curl -s -o /dev/null -w '%{http_code}' $U/slots/0?action=save $J -d '{\"filename\":\"x\"}'" \
    "d.strip()=='501'" raw
check overflow "curl -s $U/v1/chat/completions $J -d '{\"messages\":[{\"role\":\"user\",\"content\":\"x\"}],\"max_tokens\":99999}'" \
    "'exceeds the context' in d['error']['message']" json
[ $fail -ne 0 ] && { echo "--- server log"; tail -20 "$W/server.log"; echo "--- engine log"; tail -20 "$W/engine.log"; }
exit $fail
