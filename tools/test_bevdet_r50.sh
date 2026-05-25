#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG=configs/bevdet/bevdet-r50.py
CHECKPOINT=ckpts/bevdet-r50.pth

PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
    python tools/test.py "$CONFIG" "$CHECKPOINT" --eval mAP "$@"
