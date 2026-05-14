#!/usr/bin/env bash
set -euo pipefail

# Stray inference/profile processes
stray=$(pgrep -af 'llama-cli|llmtoy generate|regression_compare.py|profile_gemma4.sh|perf (record|stat)' 2>/dev/null || true)
if [[ -n "$stray" ]]; then
    echo "STRAY (stop before benchmarking):"
    echo "$stray"
else
    echo "Stray processes: none"
fi

# CPU load vs core count
read -r load1 load5 _ < /proc/loadavg
ncpu=$(nproc)
echo "CPU load (1m/5m): $load1 / $load5  ($ncpu cores)"

# Memory
free -h | awk 'NR==2 {printf "Memory: %s total, %s avail\n", $2, $7}'
