#!/usr/bin/env bash
# Reproduce the llama.cpp measurements in benchmarks/README.md with samemodel itself.
#
# For every quantisation it starts a fresh llama-server and
#   1. runs `samemodel selfcheck`  (cache reuse and co-batching on one server)
#   2. records a reference run with the prompt cache off, for comparing weights and
#      backends afterwards with `samemodel diff`
#
# Usage: llamacpp.sh <llama-server binary> <models dir> <output dir>
#
# Environment:
#   QUANTS        quantisation suffixes               (default: "q4_k_m q8_0 fp16")
#   SERVER_FLAGS  backend selection for llama-server  (default: "--device none -ngl 0", CPU only)
#   MODEL_PREFIX  GGUF file name prefix               (default: qwen2.5-0.5b-instruct)
#   SAMEMODEL     how to invoke the tool              (default: samemodel)
#   PROMPTS, THREADS, PORT
#
# Only the server PID started here is ever stopped.
set -euo pipefail

SRV="$1"; MODELS="$2"; OUT="$3"
PORT="${PORT:-18431}"
THREADS="${THREADS:-4}"
PROMPTS="${PROMPTS:-60}"
QUANTS="${QUANTS:-q4_k_m q8_0 fp16}"
SERVER_FLAGS="${SERVER_FLAGS:---device none -ngl 0}"
MODEL_PREFIX="${MODEL_PREFIX:-qwen2.5-0.5b-instruct}"
SAMEMODEL="${SAMEMODEL:-samemodel}"
URL="http://127.0.0.1:$PORT"
mkdir -p "$OUT"

for q in $QUANTS; do
  if [ -s "$OUT/$q.selfcheck.json" ] && [ -s "$OUT/$q.scratch.jsonl.gz" ]; then
    echo "skip $q (complete)"; continue
  fi
  # shellcheck disable=SC2086
  "$SRV" -m "$MODELS/$MODEL_PREFIX-$q.gguf" $SERVER_FLAGS -t "$THREADS" -c 16384 -np 4 \
    --host 127.0.0.1 --port "$PORT" --no-webui > "$OUT/$q.server.log" 2>&1 &
  pid=$!
  trap 'kill $pid 2>/dev/null || true' EXIT
  for _ in $(seq 1 240); do
    if curl -s -m 2 "$URL/health" | grep -q '"ok"'; then break; fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "server for $q exited early" >&2; exit 1; fi
    sleep 0.5
  done
  # shellcheck disable=SC2086
  $SAMEMODEL selfcheck --base-url "$URL" --model "$MODEL_PREFIX-$q" -n "$PROMPTS" --forks 3 \
    --json "$OUT/$q.selfcheck.json" > "$OUT/$q.selfcheck.txt"
  # shellcheck disable=SC2086
  $SAMEMODEL record --base-url "$URL" --model "$MODEL_PREFIX-$q" -n "$PROMPTS" \
    --extra-body '{"cache_prompt": false}' -o "$OUT/$q.scratch.jsonl.gz" > /dev/null
  kill "$pid"; wait "$pid" 2>/dev/null || true
  trap - EXIT
  echo "== $q"; sed -n 3,5p "$OUT/$q.selfcheck.txt"
done
echo "all done"
