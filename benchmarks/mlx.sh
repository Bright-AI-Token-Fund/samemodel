#!/usr/bin/env bash
# Reproduce the MLX measurements in benchmarks/README.md with samemodel itself.
#
# Usage: mlx.sh <python with mlx-lm installed> <output dir>
#
# Environment:
#   MODELS     Hugging Face repos   (default: the three Qwen2.5-0.5B-Instruct MLX builds)
#   SAMEMODEL  how to invoke the tool (default: samemodel)
#   PROMPTS, PORT
#
# mlx_lm.server always keeps a prompt cache and has no per-request switch for it, so
# the reference run is recorded with a tag instead: every conversation then starts
# with a unique system message and nothing can be reused.
#
# Only the server PID started here is ever stopped.
set -euo pipefail

PY="$1"; OUT="$2"
PORT="${PORT:-18432}"
PROMPTS="${PROMPTS:-60}"
MODELS="${MODELS:-mlx-community/Qwen2.5-0.5B-Instruct-4bit mlx-community/Qwen2.5-0.5B-Instruct-8bit mlx-community/Qwen2.5-0.5B-Instruct-bf16}"
SAMEMODEL="${SAMEMODEL:-samemodel}"
URL="http://127.0.0.1:$PORT"
mkdir -p "$OUT"

for model in $MODELS; do
  q="${model##*-}"            # 4bit / 8bit / bf16
  if [ -s "$OUT/$q.selfcheck.json" ] && [ -s "$OUT/$q.scratch.jsonl.gz" ]; then
    echo "skip $q (complete)"; continue
  fi
  "$PY" -m mlx_lm server --model "$model" --host 127.0.0.1 --port "$PORT" \
    --log-level WARNING > "$OUT/$q.server.log" 2>&1 &
  pid=$!
  trap 'kill $pid 2>/dev/null || true' EXIT
  for _ in $(seq 1 600); do
    if curl -s -m 2 "$URL/health" 2>/dev/null | grep -q ok; then break; fi
    if ! kill -0 "$pid" 2>/dev/null; then echo "server for $q exited early" >&2; exit 1; fi
    sleep 1
  done
  # shellcheck disable=SC2086
  $SAMEMODEL selfcheck --base-url "$URL" --model "$model" -n "$PROMPTS" --forks 3 \
    --json "$OUT/$q.selfcheck.json" > "$OUT/$q.selfcheck.txt"
  # shellcheck disable=SC2086
  $SAMEMODEL record --base-url "$URL" --model "$model" -n "$PROMPTS" --tag reference \
    -o "$OUT/$q.scratch.jsonl.gz" > /dev/null
  kill "$pid"; wait "$pid" 2>/dev/null || true
  trap - EXIT
  echo "== $q"; sed -n 3,5p "$OUT/$q.selfcheck.txt"
done
echo "all done"
