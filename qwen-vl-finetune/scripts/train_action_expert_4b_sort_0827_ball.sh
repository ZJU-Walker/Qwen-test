#!/bin/bash
# 0824-clean + 0827 novel-ball composition run.
#
# Sampling records (463 total, before the virtual epoch multiplier):
#   371 complete robot episodes: matching 0824 human prompt, normal QA+FAST+flow
#    46 ball action records:     no human prompt, oracle `pick ball`, FAST+flow only
#    46 ball robot-QA records:   robot demonstration/current view, language only
# This is ~80/10/10 without duplicating ball in FAST/norm statistics. 0827 cup is
# deliberately omitted from the first novel-object experiment.
set -euo pipefail
cd "$(dirname "$0")"
source sort_0827_ball_data.sh

export DATA_TAG="${DATA_TAG:-0824sort_0827ballqa}"
export TORCHRUN="${TORCHRUN:-torchrun}"
export ROBOT_DATA_DIRS="$SORT_0827_BALL_ROBOT_DIRS"
export HUMAN_PROMPT_DIRS="$SORT_0827_BALL_HUMAN_DIRS"
export UNPROMPTED_PICK_ONLY_DIRS="$SORT_0827_BALL_UNPROMPTED_DIRS"
export ROBOT_QA_STRIDE="${ROBOT_QA_STRIDE:-10}"
export ROBOT_QA_MAX_FRAMES="${ROBOT_QA_MAX_FRAMES:-12}"
export FAST_TOKENIZER="${FAST_TOKENIZER:-$SORT_0827_BALL_FAST_DEFAULT}"

if [ ! -f "$FAST_TOKENIZER/norm_stats.json" ]; then
    echo "ERROR: missing combined FAST artifact: $FAST_TOKENIZER" >&2
    echo "Run: bash scripts/fit_sort_0827_ball_fast_tokenizer.sh" >&2
    exit 1
fi

export SUBTASK_TASK=sort
export ROBOT_SUBTASK_LABELS_FILE=subtask_labels.json
export SUBTASK_FORMAT_MIX="${SUBTASK_FORMAT_MIX:-phase:0.40,object:0.10,target:0.10,where:0.15,demo:0.15,remaining:0.10}"
export QA_WHERE_ABSENT_PROB="${QA_WHERE_ABSENT_PROB:-0.2}"
export ORDER_SAMPLE_PROB=0.0
export HUMAN_PROMPT_SEGMENTS=True
export HUMAN_PROMPT_FULL_EP_PROB=0.5
export HUMAN_PROMPT_HOLDOUT=1
export HUMAN_PROMPT_HOLDOUT_PER_KEY=True
export SUBTASK_QUESTION="What should be done now?"
export SKIP_LEADING_SUBTASK=

export STATE_HISTORY=False
export IMAGE_HISTORY=False
export HUMAN_PROMPT_STRIDE="${HUMAN_PROMPT_STRIDE:-10}"
export HUMAN_PROMPT_MAX_FRAMES="${HUMAN_PROMPT_MAX_FRAMES:-8}"
export EXPLICIT_VIDEO_TIMESTAMPS="${EXPLICIT_VIDEO_TIMESTAMPS:-True}"

export EXPERT_ATTENDS_SUBTASK=True
export RTC_MAX_DELAY=20
export LM_LOSS_PER_SAMPLE=True
export PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-4}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"

exec bash train_action_expert_4b_humanprompt.sh
