#!/bin/bash
# Fresh dense-prompt 0824-sort + 0827-ball-action run for four 80-GB H100s.
#
# This is intentionally NOT layered on the two-H200 dense20 wrapper: that recipe has
# hard 2-GPU/microbatch-2 guards and warm-starts the previous policy.  Here we retain
# the same input/data contract while using 4 GPUs x microbatch 1 x accumulation 16 =
# global batch 64. Ball action/FAST/flow records stay enabled; their paired standalone
# robot-video QA records are disabled.
set -euo pipefail
cd "$(dirname "$0")"

# Fail closed if a stale shell environment would turn this into weights-only policy
# initialization. The unique output tag makes the first launch a step-zero run from
# Qwen/Qwen3-VL-4B-Instruct; later invocations may exact-resume this same fresh run.
if [ -n "${INIT_FROM-}" ]; then
    echo "ERROR: fresh no-robot-QA training refuses nonempty INIT_FROM=${INIT_FROM}" >&2
    echo "Unset INIT_FROM (or set it to an empty string) to train from the Qwen base." >&2
    exit 1
fi

export DATA_TAG=0824sort_0827ballaction_dense20s3_fresh_norobotqa_4h100
export INIT_FROM=
export CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/iris/u/kewalk/qwen_checkpoints}"

# Dense human-prompt contract (30-fps sources -> 10-Hz candidates, <=20 frames).
export HUMAN_PROMPT_STRIDE=3
export HUMAN_PROMPT_MAX_FRAMES=20
export EXPLICIT_VIDEO_TIMESTAMPS=True
export VIDEO_MAX_PIXELS=4600000
export HISTORY_MAX_PIXELS=65536
export STATE_HISTORY=False
export IMAGE_HISTORY=False

# Keep standalone ball ACTION examples while omitting only their paired video-QA
# records. The called ball wrapper supplies the exact ball root allowlist.
export STANDALONE_ROBOT_QA_ENABLED=False

# The live allocation is four 80-GB H100s. The common launcher derives accumulation
# from its fixed target batch of 64: 64 / (4 * 1) = 16.
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-1}"
export TORCH_COMPILE=False
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

if [ "$NPROC_PER_NODE" -ne 4 ]; then
    echo "ERROR: fresh no-robot-QA recipe requires exactly 4 GPUs, got $NPROC_PER_NODE" >&2
    exit 1
fi
if [ "$PER_DEVICE_BATCH" -ne 1 ]; then
    echo "ERROR: four-H100 reviewed microbatch is 1/GPU, got $PER_DEVICE_BATCH" >&2
    exit 1
fi

exec bash train_action_expert_4b_sort_0827_ball.sh
