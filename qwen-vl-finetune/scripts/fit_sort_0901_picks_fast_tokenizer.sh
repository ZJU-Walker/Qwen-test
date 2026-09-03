#!/bin/bash
# Fit FAST + normalization over 506 unique action-bearing episodes:
# 371 complete prompted episodes + 46 0827 ball picks + 89 clean 0901 picks.
# Robot-QA records are language-only virtual twins and never enter these statistics.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/sort_0901_picks_data.sh

OUTPUT_DIR="${OUTPUT_DIR:-$SORT_0901_PICKS_FAST_DEFAULT}"
PYTHON_BIN="${PYTHON_BIN:-/iris/projects/humanoid/miniconda3/envs/qwen3vl/bin/python}"
if [ -e "$OUTPUT_DIR" ] && [ "${ALLOW_OVERWRITE:-False}" != "True" ]; then
    echo "ERROR: FAST artifact already exists: $OUTPUT_DIR" >&2
    echo "Use a new OUTPUT_DIR, or set ALLOW_OVERWRITE=True only intentionally." >&2
    exit 1
fi

"$PYTHON_BIN" -m scripts.train_fast_tokenizer \
    --robot_data_dirs "$SORT_0901_PICKS_ROBOT_DIRS" \
    --active_dims "7:14" \
    --delta_mask "9,-1" \
    --use_delta_actions True \
    --action_space ee6d \
    --action_horizon 50 \
    --train_split 1.0 \
    --min_episode_len 50 \
    --output_dir "$OUTPUT_DIR"
