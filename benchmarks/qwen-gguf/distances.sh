#!/bin/bash
# Prints the distance tables of README.md from the recordings in data/runs/.
# Usage: distances.sh > data/distances.txt
set -eu
SAMEMODEL=${SAMEMODEL:-samemodel}
RUNS=$(cd "$(dirname "$0")" && pwd)/data/runs

row() {
    printf "%-26s -> %-30s" "$1" "$2"
    "$SAMEMODEL" diff "$RUNS/$1.jsonl.gz" "$RUNS/$2.jsonl.gz" | sed -n '4p' | cut -c8-
}

echo "                                                              same   argmax  drift50  drift99     KL99  forks  verdict"
echo "# the same file in llama.cpp b12244 and in Ollama 0.40.16"
row checkpoint-f16   checkpoint-f16.via-ollama
row qwen-gguf-q4_k_m qwen-gguf-q4_k_m.via-ollama
row ollama-q4_k_m    ollama-q4_k_m.via-ollama
echo "# different files in llama.cpp b12244"
row checkpoint-f16 qwen-gguf-fp16
row checkpoint-f16 qwen-gguf-fp16-unclipped
row checkpoint-f16 qwen-gguf-q4_k_m
row checkpoint-f16 ollama-q4_k_m
row qwen-gguf-fp16 qwen-gguf-q4_k_m
row qwen-gguf-fp16 ollama-q4_k_m
row qwen-gguf-q4_k_m ollama-q4_k_m
