#!/bin/bash
# Records every setup compared in README.md into data/runs/, one fresh llama-server per file.
# Nothing is downloaded for you. Set:
#
#   LLAMA_SERVER   llama-server from the official b12244 macos-arm64 release
#   QWEN_Q4        qwen2.5-0.5b-instruct-q4_k_m.gguf  from Qwen/Qwen2.5-0.5B-Instruct-GGUF
#   QWEN_FP16      qwen2.5-0.5b-instruct-fp16.gguf    from the same repository
#   OLLAMA_Q4      the model blob of `ollama pull qwen2.5:0.5b`
#   OLLAMA_FP16    the model blob of `ollama pull qwen2.5:0.5b-instruct-fp16`
#   UNCLIPPED      a copy of QWEN_FP16 patched by `rescale_check.py ... --unclip`
#   OLLAMA_URL     a running Ollama 0.40.16 that has the two tags above and `qwen-gguf-q4`,
#                  created with:  printf 'FROM %s\n' "$QWEN_Q4" > Modelfile; ollama create qwen-gguf-q4 -f Modelfile
#   SAMEMODEL      the samemodel command (default: samemodel)
#
# The blobs are in Ollama's model directory; `ollama show --modelfile TAG` prints the path.
set -eu
SAMEMODEL=${SAMEMODEL:-samemodel}
OUT=$(cd "$(dirname "$0")" && pwd)/data/runs
PORT=${PORT:-8191}
mkdir -p "$OUT"

llamacpp() {   # llamacpp FILE LABEL
    "$LLAMA_SERVER" -m "$1" --host 127.0.0.1 --port "$PORT" -c 4096 -np 1 > "$OUT/$2.server.log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 60); do
        curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q ok && break
        sleep 1
    done
    "$SAMEMODEL" record --base-url "http://127.0.0.1:$PORT" -n 60 --max-tokens 128 \
        --extra-body '{"cache_prompt": false}' -o "$OUT/$2.jsonl.gz"
    kill "$pid"
    wait "$pid" 2>/dev/null || true
    rm -f "$OUT/$2.server.log"
}

ollama() {     # ollama TAG LABEL
    "$SAMEMODEL" record --base-url "$OLLAMA_URL" --model "$1" -n 60 --max-tokens 128 -o "$OUT/$2.jsonl.gz"
}

llamacpp "$OLLAMA_FP16" checkpoint-f16
llamacpp "$QWEN_FP16"   qwen-gguf-fp16
llamacpp "$UNCLIPPED"   qwen-gguf-fp16-unclipped
llamacpp "$QWEN_Q4"     qwen-gguf-q4_k_m
llamacpp "$OLLAMA_Q4"   ollama-q4_k_m
ollama qwen2.5:0.5b               ollama-q4_k_m.via-ollama
ollama qwen-gguf-q4               qwen-gguf-q4_k_m.via-ollama
ollama qwen2.5:0.5b-instruct-fp16 checkpoint-f16.via-ollama
