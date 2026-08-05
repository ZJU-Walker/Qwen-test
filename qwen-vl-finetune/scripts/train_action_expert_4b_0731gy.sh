#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit

# ======================================================================================
# 0717 + 0731 run (2026-08-02): constant-lr recipe (train_action_expert_4b_constlr.sh) on
# the 0717 merged set PLUS the new 0731_green_yellow_merged set (448 episodes, ~97k
# frames, 50/50 episode draw). The 0730 DAgger sets are dropped (flawed data).
#
#   * MIXED SUBTASK SUPERVISION: 0731 has no subtask labels, so its episodes build NO
#     assistant turn (VLM supervised through FAST alone; the sequence = the exact
#     generation-prompt form insulated serving uses). 0717's labeled episodes keep the
#     subtask answer as before. Automatic per-episode in robot_data.py -- labeling 0731
#     later (streamlit) would be picked up with no config change.
#   * NEW FAST tokenizer artifact (fit on both dirs); its norm_stats.json is the frozen
#     stats for this run -- nothing is read from or written to dataset dirs.
#   * Everything else is byte-identical to the constlr run: warmup 300 -> constant lr,
#     EMA 0.999, bs 32, RTC 20, subtask insulation, L18, vis16 budget.
#   * --max_steps 40000 is a CEILING: stop on plateau (seed-variance probe + offline
#     MSE every ~2k steps). Every checkpoint is servable.
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
OUTPUT_DIR="/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/qwen3_4b_ae_hist_subpred_0717m_0731gy"
CACHE_DIR="$CUSTOM_CACHE_DIR/huggingface"
ROBOT_DATA_DIRS="/iris/projects/humanoid/trossen_data/0717_green_yellow_block_mem_merged,/iris/projects/humanoid/trossen_data/0731_green_yellow_merged"
FAST_TOKENIZER="/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717merged_0731gy"
RUN_NAME="qwen3vl_4b_ae_hist_subpred_0717m_0731gy_bs32"

# Max training-time RTC prefix length d ~ Uniform[0, max]. Reduced 20 -> 10 (2026-08-02):
# the deployed client runs delay 8-10, so 10 matches the serving distribution exactly.
RTC_MAX_DELAY=10
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

# Visual token budget (identical to the cosine/constlr runs).
VIDEO_MAX_PIXELS=1600000
WRIST_MAX_PIXELS=131072
MAX_PIXELS=131072
OUTPUT_DIR="${OUTPUT_DIR}_vis$((VIDEO_MAX_PIXELS / 100000))"
RUN_NAME="${RUN_NAME}_vis$((VIDEO_MAX_PIXELS / 100000))"

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
    # ---- robot data: 0717 merged (labeled) + 0731 green/yellow (unlabeled) ----
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
    # ---- knowledge insulation + FAST (new 3-dir artifact, carries norm_stats.json) ----
    --train_vlm True
    --predict_subtask True
    --use_fast_tokens True
    --fast_tokenizer_path "$FAST_TOKENIZER"
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
