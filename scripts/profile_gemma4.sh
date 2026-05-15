#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-/opt/ai-lab/models/mudler/gemma-4-26B-A4B-it-APEX-GGUF/gemma-4-26B-A4B-APEX-I-Mini.gguf}
PROMPT=${PROMPT:-Briefly explain the full forward pass of a MoE model}
THREADS=${THREADS:-12}
TOKENS=${TOKENS:-16}
TEMP=${TEMP:-0.1}
TOP_K=${TOP_K:-40}
TOP_P=${TOP_P:-0.9}
SEED=${SEED:-42}
GPU=${GPU:-0}
MODE=${1:-stat}

cmd=(
  ./zig-out/bin/llmtoy generate
  "$MODEL"
  "$PROMPT"
  --chat
  --max-tokens "$TOKENS"
  --threads "$THREADS"
  --temperature "$TEMP"
  --top-k "$TOP_K"
  --top-p "$TOP_P"
  --seed "$SEED"
)
[[ "$GPU" == "1" ]] && cmd+=(--gpu)

# GPU runs need a memory cap to prevent system OOM.
runner=()
[[ "$GPU" == "1" ]] && runner=(systemd-run --scope -p MemoryMax=40G)

nix develop --command zig build -Doptimize=ReleaseFast

case "$MODE" in
  stat)
    "${runner[@]}" nix develop --command perf stat -d -d -d -- "${cmd[@]}"
    ;;
  record)
    mkdir -p profiles
    stamp=$(date +%Y%m%d-%H%M%S)
    suffix=$([[ "$GPU" == "1" ]] && echo "-gpu" || echo "-cpu")
    out="profiles/gemma4${suffix}-${stamp}"
    "${runner[@]}" nix develop --command perf record -F 99 -g -o "${out}.data" -- "${cmd[@]}"
    nix develop --command perf report -i "${out}.data" --stdio --no-children --sort comm,dso,symbol > "${out}.report.txt"
    echo "wrote ${out}.data"
    echo "wrote ${out}.report.txt"
    ;;
  *)
    echo "usage: GPU=1 $0 [stat|record]" >&2
    exit 2
    ;;
esac
