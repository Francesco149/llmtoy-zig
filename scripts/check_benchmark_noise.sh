#!/usr/bin/env bash
set -euo pipefail

lines="${1:-25}"

echo "== Top CPU processes =="
ps -eo pid,comm,pcpu,pmem,args --sort=-pcpu | head -n "$lines"

echo
echo "== Inference/profile processes =="
pgrep -af 'llama-cli|llmtoy generate|regression_compare.py|profile_gemma4.sh|perf (record|stat)' || true

echo
echo "Before recording benchmark numbers, stop or wait for unrelated high-CPU"
echo "processes. Idle llama-server processes are usually fine for CPU timing, but"
echo "large resident models can still affect memory pressure."
