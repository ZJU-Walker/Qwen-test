#!/bin/bash
# Shared, ordered source list for the 0824-clean + 0827-ball-QA experiment.
# Source this file; it intentionally performs no training or fitting itself.

# Relocatable overrides are used by the portable GCS bootstrap.  Defaults preserve the
# original /iris recipes byte-for-byte for local launches.
SORT_0824_ROOT="${SORT_0824_ROOT:-/iris/projects/humanoid/trossen_data/0824_prompting}"
SORT_0827_ROOT="${SORT_0827_ROOT:-/iris/projects/humanoid/trossen_data/0827_prompting_playdata/data_0827_prompting_playdata}"

SORT_0827_BALL_ROBOT_DIRS=""
for sort_dataset in \
    111 112 113 121 122 123 131 132 133 \
    311 312 313 321 322 323 331 332 333 \
    cup green_block grey_pepper_box tape; do
    SORT_0827_BALL_ROBOT_DIRS="${SORT_0827_BALL_ROBOT_DIRS:+$SORT_0827_BALL_ROBOT_DIRS,}$SORT_0824_ROOT/$sort_dataset"
done
for sort_dataset in green grey tape ball; do
    SORT_0827_BALL_ROBOT_DIRS="$SORT_0827_BALL_ROBOT_DIRS,$SORT_0827_ROOT/$sort_dataset"
done
unset sort_dataset

SORT_0827_BALL_HUMAN_DIRS="$SORT_0824_ROOT/human_demo"
SORT_0827_BALL_UNPROMPTED_DIRS="$SORT_0827_ROOT/ball"
SORT_0827_BALL_FAST_DEFAULT="${SORT_0827_BALL_FAST_DEFAULT:-/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0824sort_0827ball_ee6d}"
