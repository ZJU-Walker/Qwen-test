#!/bin/bash
# Fresh dense-20 prompted run with standalone pickup QA on four 80-GB H100s.
set -euo pipefail
cd "$(dirname "$0")"

if [ -n "${INIT_FROM-}" ]; then
    echo "ERROR: fresh 0901 pickup-QA training refuses nonempty INIT_FROM=${INIT_FROM}" >&2
    echo "Unset INIT_FROM (or set it empty) to initialize a new policy from Qwen base." >&2
    exit 1
fi

export DATA_TAG=0824sort_0827ball_0901picksqa_dense20s3_fresh_4h100
export INIT_FROM=
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/iris/u/kewalk/qwen_checkpoints}"

# Human prompts: 30-fps source, 10-Hz candidate cadence, endpoint preserving, <=20
# frames. 4.6M is a whole-video budget and preserves the reviewed per-frame grid.
export HUMAN_PROMPT_STRIDE=3
export HUMAN_PROMPT_MAX_FRAMES=20
export EXPLICIT_VIDEO_TIMESTAMPS=True
export VIDEO_MAX_PIXELS=4600000
export HISTORY_MAX_PIXELS=65536
export IMAGE_HISTORY=False
export STATE_HISTORY=False
export CURRENT_STATE_MASK_PROB=0.0

# Each standalone root contributes one action record and one robot visual-QA record.
export STANDALONE_ROBOT_QA_ENABLED=True
export ROBOT_QA_STRIDE=10
export ROBOT_QA_MAX_FRAMES=12

# 4 GPUs x microbatch 1 x accumulation 16 = global batch 64.
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
export TORCH_COMPILE=False
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

if [ "$NPROC_PER_NODE" -ne 4 ]; then
    echo "ERROR: 0901 dense20 recipe requires exactly 4 GPUs, got $NPROC_PER_NODE" >&2
    exit 1
fi
if [ "$PER_DEVICE_BATCH" -ne 1 ]; then
    echo "ERROR: reviewed four-H100 microbatch is 1/GPU, got $PER_DEVICE_BATCH" >&2
    exit 1
fi

exec bash train_action_expert_4b_sort_0901_picks.sh
