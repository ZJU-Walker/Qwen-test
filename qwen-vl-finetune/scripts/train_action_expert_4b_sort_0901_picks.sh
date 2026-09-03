#!/bin/bash
# Cleaned 0824 + selected 0827 + clean standalone 0901 pickup/robot-QA run.
#
# Sampling records (641 total before the virtual epoch multiplier):
#   371 complete prompted action records: normal sort QA + FAST + flow
#   135 standalone pickup action records: oracle phase context + FAST + flow
#   135 standalone robot-video QA records: destination-free text CE only
# The standalone set is 46 old 0827 ball + 28 clean 0901 ball + 30 green + 31 grey.
set -euo pipefail
cd "$(dirname "$0")"
source sort_0901_picks_data.sh

export DATA_TAG="${DATA_TAG:-0824sort_0827ball_0901picksqa}"
export TORCHRUN="${TORCHRUN:-torchrun}"
export ROBOT_DATA_DIRS="$SORT_0901_PICKS_ROBOT_DIRS"
export HUMAN_PROMPT_DIRS="$SORT_0901_PICKS_HUMAN_DIRS"
export UNPROMPTED_PICK_ONLY_DIRS="$SORT_0901_PICKS_UNPROMPTED_DIRS"
export STANDALONE_ROBOT_QA_ENABLED="${STANDALONE_ROBOT_QA_ENABLED:-True}"
export ROBOT_QA_STRIDE="${ROBOT_QA_STRIDE:-10}"
export ROBOT_QA_MAX_FRAMES="${ROBOT_QA_MAX_FRAMES:-12}"
export FAST_TOKENIZER="${FAST_TOKENIZER:-$SORT_0901_PICKS_FAST_DEFAULT}"

if [ ! -f "$FAST_TOKENIZER/norm_stats.json" ]; then
    echo "ERROR: missing 0901 FAST artifact: $FAST_TOKENIZER" >&2
    echo "Run: bash scripts/fit_sort_0901_picks_fast_tokenizer.sh" >&2
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
