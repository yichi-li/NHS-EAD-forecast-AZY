#!/usr/bin/env bash
# Tiny wrapper that sets DYLD_LIBRARY_PATH so LightGBM finds libomp on macOS,
# then invokes uv run with whatever args were passed.
# Usage:  ./run.sh python scripts/01_long_to_wide.py
set -euo pipefail
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/libomp/lib:${DYLD_LIBRARY_PATH:-}"
cd "$(dirname "$0")"
exec uv run "$@"
