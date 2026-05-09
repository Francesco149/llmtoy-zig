#!/usr/bin/env bash
# Convenience wrapper — runs from project root inside `nix develop`
set -euo pipefail
cd "$(dirname "$0")/.."
exec zig build "$@"
