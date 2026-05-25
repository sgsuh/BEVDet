#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG=configs/bevdet/bevdet-r50.py
CHECKPOINT=ckpts/bevdet-r50.pth
WORK_DIR=work_dirs/

PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
    python tools/convert_bevdet_to_TRT.py "$CONFIG" "$CHECKPOINT" "$WORK_DIR" --fuse-conv-bn --fp16 --int8 "$@"
