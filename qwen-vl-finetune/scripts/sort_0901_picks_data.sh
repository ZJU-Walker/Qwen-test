#!/bin/bash
# Shared source list for the cleaned 0824 + selected 0827 + standalone 0901 picks.
# Source this file; it does not train or fit anything.

source "$(dirname "${BASH_SOURCE[0]}")/sort_0827_ball_data.sh"

# All three 0901 roots contain one full-episode pickup segment and intentionally have
# no paired human demonstration.  The 0901ball dataset-local exclusion sidecar removes
# episodes 8 and 11 (failed/regrasp demonstrations) in both training and FAST fitting.
SORT_0901_ROOT="${SORT_0901_ROOT:-/iris/projects/humanoid/trossen_data}"

SORT_0901_PICKS_ROBOT_DIRS="$SORT_0827_BALL_ROBOT_DIRS"
SORT_0901_PICKS_UNPROMPTED_DIRS="$SORT_0827_BALL_UNPROMPTED_DIRS"
for sort_dataset in 0901ball 0901green 0901grey; do
    SORT_0901_PICKS_ROBOT_DIRS="$SORT_0901_PICKS_ROBOT_DIRS,$SORT_0901_ROOT/$sort_dataset"
    SORT_0901_PICKS_UNPROMPTED_DIRS="$SORT_0901_PICKS_UNPROMPTED_DIRS,$SORT_0901_ROOT/$sort_dataset"
done
unset sort_dataset

SORT_0901_PICKS_HUMAN_DIRS="$SORT_0827_BALL_HUMAN_DIRS"
SORT_0901_PICKS_FAST_DEFAULT="${SORT_0901_PICKS_FAST_DEFAULT:-/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0824sort_0827ball_0901picks_ee6d}"
