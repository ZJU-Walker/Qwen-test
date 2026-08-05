#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit

# ======================================================================================
# CONSTANT-LR run ("Choice 2", 2026-07-28): openpi's pi05_libero philosophy on our model.
#
#   * lr schedule: 300-step warmup -> CONSTANT forever (no anneal). Nothing ever calms
#     the weights down, so the smoothing is done entirely by...
#   * EMA 0.999 (~1000-step average; openpi pairs constant lr with exactly this). The
#     served/eval weights are the EMA copy (automatic via ema_expert.pt). NOTE: the EMA
#     average only becomes meaningful after ~2-3k steps -- do not judge early checkpoints.
#   * learning rates NOT copied from openpi (their 5e-5 was for a PREtrained expert):
#     ours stay 1e-4 from-scratch expert / 1e-5 VLM, as in the cosine run.
#   * --max_steps 40000 is a CEILING, not a plan: stop on plateau. Every checkpoint is
#     servable (that is the point of this recipe). Kill + relaunch auto-resumes.
#   * single-GPU node: grad_accum 8 -> effective batch 4 x 8 x 1 = 32, matching the
#     batch the rest of the recipe was designed around.
#
# Stopping rule: every ~2k steps run the seed-variance probe + offline MSE on the newest
# checkpoint; stop when both plateau. Compare against the 10k-cosine run.
# ======================================================================================

CUSTOM_CACHE_DIR="/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache"
mkdir -p "$CUSTOM_CACHE_DIR/huggingface" "$CUSTOM_CACHE_DIR/triton"
export HF_HOME="$CUSTOM_CACHE_DIR/huggingface"
export TRITON_CACHE_DIR="$CUSTOM_CACHE_DIR/triton"

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
# Overridable so multiple runs can share a node (NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 ...).
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}

MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
OUTPUT_DIR="/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/qwen3_4b_ae_hist_subpred_0717merged"
CACHE_DIR="$CUSTOM_CACHE_DIR/huggingface"
ROBOT_DATA_DIRS="/iris/projects/humanoid/trossen_data/0717_green_yellow_block_mem_merged"
RUN_NAME="qwen3vl_4b_ae_hist_subpred_0717merged_bs32"

RTC_MAX_DELAY=20
if [ "$RTC_MAX_DELAY" -gt 0 ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_rtc${RTC_MAX_DELAY}"
    RUN_NAME="${RUN_NAME}_rtc${RTC_MAX_DELAY}"
fi

EXPERT_ATTENDS_SUBTASK=False
if [ "$EXPERT_ATTENDS_SUBTASK" != "True" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_subinsul"
    RUN_NAME="${RUN_NAME}_subinsul"
fi

EXPERT_VLM_LAYERS=18
if [ "$EXPERT_VLM_LAYERS" -gt 0 ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_L${EXPERT_VLM_LAYERS}"
    RUN_NAME="${RUN_NAME}_L${EXPERT_VLM_LAYERS}"
fi

# Visual token budget (identical to the cosine run; see train_action_expert_4b_2gpu.sh).
VIDEO_MAX_PIXELS=1600000
WRIST_MAX_PIXELS=131072
MAX_PIXELS=131072
OUTPUT_DIR="${OUTPUT_DIR}_vis$((VIDEO_MAX_PIXELS / 100000))"
RUN_NAME="${RUN_NAME}_vis$((VIDEO_MAX_PIXELS / 100000))"

# Constant-lr recipe marker (vs the cosine run's _opt2).
EMA_DECAY=0.999
OUTPUT_DIR="${OUTPUT_DIR}_constlr"
RUN_NAME="${RUN_NAME}_constlr"

export QWEN_DUMP_MODEL_INPUTS="${OUTPUT_DIR}/input_dumps"
export QWEN_DUMP_MODEL_INPUTS_N=2

args=(
    --deepspeed scripts/zero2_action_expert.json
    --model_name_or_path "$MODEL_PATH"
    --output_dir "$OUTPUT_DIR"
    --cache_dir "$CACHE_DIR"
    # ---- robot data (identical to the cosine run) ----
    --robot_data_dirs "$ROBOT_DATA_DIRS"
    --camera cam_high
    --num_frames 10
    --frame_stride 10
    --active_dims "7:14"
    --action_dim 7
    --delta_mask "6,-1"
    --action_horizon 50
    --use_delta_actions True
    --wrist_cameras "cam_right_wrist"
    --wrist_max_pixels "$WRIST_MAX_PIXELS"
    --default_prompt "waiting"
    --train_split 1.0
    # ---- knowledge insulation + FAST ----
    --train_vlm True
    --predict_subtask True
    --use_fast_tokens True
    --fast_tokenizer_path /iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717merged
    --fast_loss_weight 1.0
    # ---- visual token budget ----
    --video_max_pixels "$VIDEO_MAX_PIXELS"
    --max_pixels "$MAX_PIXELS"
    --min_pixels 784
    # ---- training-time RTC ----
    --rtc_prefix_max_length "$RTC_MAX_DELAY"
    # ---- subtask insulation ----
    --expert_attends_subtask "$EXPERT_ATTENDS_SUBTASK"
    # ---- SmolVLA layer skipping ----
    --expert_num_layers "$EXPERT_VLM_LAYERS"
    # ---- training: CONSTANT LR + EMA 0.999 (see header) ----
    --bf16
    --adam_beta2 0.95
    --ema_decay "$EMA_DECAY"
    --lr_scheduler_type constant_with_warmup
    --warmup_steps 300
    --max_steps 40000
    --per_device_train_batch_size 4
    --gradient_accumulation_steps 8
    --eval_strategy "no"
    --save_strategy "steps"
    --save_steps 500
    --save_total_limit 3
    --save_safetensors False
    --learning_rate 1e-4
    --vlm_learning_rate 1e-5
    --weight_decay 0
    --max_grad_norm 1
    --logging_steps 1
    --model_max_length 8192
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --dataloader_persistent_workers True
    --run_name "$RUN_NAME"
    --report_to wandb
)

torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_action_expert.py "${args[@]}"
