#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

CONFIG=configs/bevdet/bevdet-r50.py
CHECKPOINT=ckpts/bevdet-r50.pth
SAVEPATH=work_dirs/bevdet-r50-vis

mkdir -p "$SAVEPATH"

PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
    python tools/test.py "$CONFIG" "$CHECKPOINT" \
        --format-only \
        --eval-options "jsonfile_prefix=$SAVEPATH"

PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" \
    python tools/analysis_tools/vis.py "$SAVEPATH/pts_bbox/results_nusc.json" \
        --save_path "$SAVEPATH" \
        "$@"
