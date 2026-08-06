#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit

# ======================================================================================
# HUMAN-VIDEO-PROMPT run (2026-08-06): the language instruction is replaced by a short
# human demonstration clip. Recipe = the ee6d constant-lr recipe on gated 0717 merged:
#
#   * DATA: 0717 merged (224 eps; human points -> robot picks), with the pointing phase
#     gated OUT via --skip_leading_subtask waiting: the "waiting" segment is never a
#     training timestep NOR history context (hist_min clamp -- unlike DAgger, history
#     must not reach into it: it shows the human pointing at the target in the robot's
#     own view and would leak the task past the prompt video). ep 149 auto-dropped
#     (mislabeled). Norm stats + FAST fit under the same gate (frozen artifact below).
#   * HUMAN PROMPT: per sample, a random same-color human demo clip (green pool 31-4,
#     yellow pool 32-4; last 4 of each held out for eval) is prepended as a SECOND video
#     ("Human demonstration:" <clip> "Robot view:" <history> ...). Fresh pairing every
#     draw. Clip sampling stride 10 (matches the history video's effective 3.3 fps ->
#     truthful shared M-RoPE timing), final frame always included, max 12 frames.
#   * SUBTASK: fixed question names no color; the color word appears ONLY in the
#     supervised assistant answer ("pick up green/yellow block") -- the VLM must read it
#     off the demo clip. Everything else (ee6d, both wrists, state history, RTC 10,
#     subtask insulation, L18 early exit, pi05 aug, constant LR + EMA) matches ee6d.
#   * --max_steps 30000 is a CEILING: stop on plateau. Every checkpoint is servable.
# ======================================================================================

CUSTOM_CACHE_DIR="/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache"
mkdir -p "$CUSTOM_CACHE_DIR/huggingface" "$CUSTOM_CACHE_DIR/triton"
export HF_HOME="$CUSTOM_CACHE_DIR/huggingface"
export TRITON_CACHE_DIR="$CUSTOM_CACHE_DIR/triton"
# wandb's service handshake writes a port file under TMPDIR; NFS home breaks it.
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
# DO NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here: with DeepSpeed
# overlap_comm (async side-stream gradient reduction) it silently corrupted parameter
# memory -- exactly 50 KiB of nan over embed_tokens rows 0-9, deterministically at
# optimizer step 3 (2026-08-06; see EmbedNanWatchCallback / QWEN_NAN_DEBUG).

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
# Pinned (not nvidia-smi): on a shared HGX node nvidia-smi ignores CUDA_VISIBLE_DEVICES
# and would fan out to all 8 GPUs. Override: NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 ...
NPROC_PER_NODE=${NPROC_PER_NODE:-2}

# Effective batch is FIXED at 64 regardless of GPU count (accumulation derived).
TARGET_BATCH=64
# 2 (not the ee6d recipe's 4): the human-prompt clip is a second video, sequences are
# ~1600 tokens vs ~1000, and 4-per-device OOMs an H200 by ~5 GB during backward.
PER_DEVICE_BATCH=2
if [ $((TARGET_BATCH % (PER_DEVICE_BATCH * NPROC_PER_NODE))) -ne 0 ]; then
    echo "ERROR: TARGET_BATCH=$TARGET_BATCH not divisible by $PER_DEVICE_BATCH x $NPROC_PER_NODE GPUs" >&2
    exit 1
fi
GRAD_ACCUM=$((TARGET_BATCH / (PER_DEVICE_BATCH * NPROC_PER_NODE)))

MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
OUTPUT_DIR="/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/qwen3_4b_ae_humanprompt_0717m_ee6d"
CACHE_DIR="$CUSTOM_CACHE_DIR/huggingface"
ROBOT_DATA_DIRS="/iris/projects/humanoid/trossen_data/0717_green_yellow_block_mem_merged"
HUMAN_PROMPT_DIRS="green=/iris/projects/humanoid/trossen_data/green_human_prompt,yellow=/iris/projects/humanoid/trossen_data/yellow_human_prompt"
FAST_TOKENIZER="/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717m_ee6d_gated"
RUN_NAME="qwen3vl_4b_ae_humanprompt_0717m_ee6d_bs64"

# Training-time RTC prefix d ~ Uniform[0, max]; 10 matches the deployed delay of 8-10.
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

# Visual token budget (identical to the ee6d run; the prompt clip draws the same
# per-video budget as the history video, so the sequence grows by roughly one video).
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
    # ---- robot data: gated 0717 merged + human demo pools ----
    --robot_data_dirs "$ROBOT_DATA_DIRS"
    --camera cam_high
    --num_frames 10
    --frame_stride 10
    --active_dims "7:14"
    --action_space ee6d
    --action_dim 10
    --delta_mask "9,-1"
    --action_horizon 50
    --use_delta_actions True
    --wrist_cameras "cam_right_wrist,cam_left_wrist"
    --wrist_max_pixels "$WRIST_MAX_PIXELS"
    --default_prompt "waiting"
    --train_split 1.0
    --min_episode_len 50
    --image_aug True
    --image_aug_prob 1.0
    --state_history True
    # ---- the human-video-prompt setup ----
    --skip_leading_subtask waiting
    --human_prompt_dirs "$HUMAN_PROMPT_DIRS"
    --human_prompt_stride 10
    --human_prompt_max_frames 12
    --human_prompt_holdout 4
    --subtask_question "which colored block did the human demonstrate picking up?"
    # ---- knowledge insulation + FAST (gated ee6d artifact, carries norm_stats.json) ----
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
    # ---- training: CONSTANT LR + EMA 0.999 ----
    --bf16
    --adam_beta2 0.95
    --ema_decay "$EMA_DECAY"
    --lr_scheduler_type constant_with_warmup
    --warmup_steps 300
    --max_steps 30000
    --per_device_train_batch_size "$PER_DEVICE_BATCH"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --eval_strategy "no"
    --save_strategy "steps"
    --save_steps 2000
    --save_total_limit 3
    --save_safetensors False
    --learning_rate 1e-4
    --vlm_learning_rate 1e-5
    --weight_decay 0
    --max_grad_norm 1
    --logging_steps 1
    --model_max_length 8192
    --gradient_checkpointing True
    --dataloader_num_workers "${NUM_WORKERS:-4}"
    # persistent_workers requires num_workers > 0 (NUM_WORKERS=0 is the nan-debug mode:
    # in-process data loading so the provenance ring buffer is visible to the trainer).
    --dataloader_persistent_workers "$([ "${NUM_WORKERS:-4}" -gt 0 ] && echo True || echo False)"
    # 223 samples/epoch with per-device 2 leaves a batch-of-1 tail each epoch; that
    # singleton hit a mixed-attention-mask edge case (fully-masked rows -> nan CE that
    # poisoned the weights at epoch 1.0). Dropping the tail costs <=1 random sample/epoch.
    --dataloader_drop_last True
    --run_name "$RUN_NAME"
    --report_to wandb
)

torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_action_expert.py "${args[@]}"
