#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG=configs/bevdet/bevdet-r50.py
ENGINE=work_dirs/bevdet_int8_fuse.engine

PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
    python tools/analysis_tools/benchmark_trt.py "$CONFIG" "$ENGINE" --samples 80 "$@"
