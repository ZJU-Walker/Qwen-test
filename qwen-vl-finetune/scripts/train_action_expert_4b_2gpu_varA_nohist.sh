#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit

# ======================================================================================
# ABLATION VARIANT A: NO image history + subtask-as-input.  (2-GPU, from scratch.)
#
# Goal: check whether the action expert can learn a good policy at all with the SAME setup
# that works on pi05 -- i.e. rule out the long image-history context as the culprit.
#
#   * --image_history False   -> the ONLY top-down image is the CURRENT cam_high still
#                                (no history video). Input = 2 images: cam_high(t) + wrist(t).
#   * --predict_subtask False -> the SUBTASK ("pick up yellow block" / "pick up green block")
#                                is the language INPUT (prompt). The VLM does NOT predict the
#                                subtask; it is supervised ONLY through FAST. At inference the
#                                subtask is hard-coded by the caller.
#   * Robot joint state is still an input (discretized in the prompt, pi0.5-style).
#   * Knowledge insulation / gradient clamping + FAST are UNCHANGED (train_vlm True, FAST on).
#
# Variant B (train_action_expert_4b_2gpu_varB_hist.sh) is identical but --image_history True.
#
# New empty OUTPUT_DIR -> nothing to auto-resume -> trains FROM SCRATCH.
# ======================================================================================

CUSTOM_CACHE_DIR="/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache"
mkdir -p "$CUSTOM_CACHE_DIR/huggingface" "$CUSTOM_CACHE_DIR/triton"
export HF_HOME="$CUSTOM_CACHE_DIR/huggingface"
export TRITON_CACHE_DIR="$CUSTOM_CACHE_DIR/triton"

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
# Default: use every visible GPU (2 in a 2-GPU cgroup). To run this variant on ONE GPU while
# another variant trains on the other, launch with:
#   CUDA_VISIBLE_DEVICES=0 NPROC_PER_NODE=1 MASTER_PORT=21001 ./this.sh
# NOTE: nvidia-smi ignores CUDA_VISIBLE_DEVICES, so you MUST pass NPROC_PER_NODE=1 explicitly
# when pinning to a single GPU (otherwise it counts all GPUs -> 2 ranks -> crash).
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}

MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
# NEW empty output dir -> nothing to auto-resume -> trains FROM SCRATCH. Keep it empty.
OUTPUT_DIR="/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/qwen3_4b_ae_nohist_subin"
CACHE_DIR="$CUSTOM_CACHE_DIR/huggingface"
ROBOT_DATA_DIRS="/iris/projects/humanoid/trossen_data/0528_green_yellow_block_mem_merged_bk"
RUN_NAME="qwen3vl_4b_ae_nohist_subin_bs32"

# Training-time RTC (arXiv:2512.05964): set RTC_MAX_DELAY below to train with
# action-prefix conditioning, d ~ Uniform[0, RTC_MAX_DELAY] control ticks. Pick it >= the real
# deploy latency in ticks (latency_s * 30 Hz; ~0.6 s expert-only => ~20) and <= H - exec_horizon.
# 0 = off (vanilla training, bit-identical to before). RTC runs get their own
# OUTPUT_DIR / RUN_NAME suffix so they never mix with (or auto-resume from) the vanilla run.
RTC_MAX_DELAY=25   # <-- edit me: 0 = RTC off (vanilla); >0 trains with d ~ Uniform[0, this]
if [ "$RTC_MAX_DELAY" -gt 0 ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_rtc${RTC_MAX_DELAY}"
    RUN_NAME="${RUN_NAME}_rtc${RTC_MAX_DELAY}"
fi

args=(
    --deepspeed scripts/zero2_action_expert.json
    --model_name_or_path "$MODEL_PATH"
    --output_dir "$OUTPUT_DIR"
    --cache_dir "$CACHE_DIR"
    # ---- robot data ----
    --robot_data_dirs "$ROBOT_DATA_DIRS"
    --camera cam_high
    --num_frames 10
    --frame_stride 10
    --image_history False          # <-- VARIANT A: current cam_high still only, no history video
    --active_dims "7:14"
    --action_dim 7
    --delta_mask "6,-1"
    --action_horizon 50
    --use_delta_actions True
    --wrist_cameras "cam_right_wrist"
    --wrist_max_pixels 50176
    --default_prompt "waiting"
    --train_split 1.0
    # ---- knowledge insulation + FAST (UNCHANGED) ----
    --train_vlm True
    --predict_subtask False        # <-- subtask is the INPUT prompt; VLM supervised only via FAST
    --use_fast_tokens True
    --fast_tokenizer_path /iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen
    --fast_loss_weight 1.0
    --max_pixels 50176
    --min_pixels 784
    # ---- training-time RTC (0 = off; see RTC_MAX_DELAY above) ----
    --rtc_prefix_max_length "$RTC_MAX_DELAY"
    # ---- training ----
    --bf16
    # effective batch = NPROC x per_device(4) x grad_accum(4): 2 GPUs -> 32, single GPU -> 16.
    # (A and B use the SAME value, so the ablation stays comparable either way.)
    --max_steps 40000
    --per_device_train_batch_size 4
    --gradient_accumulation_steps 4
    --eval_strategy "no"
    --save_strategy "steps"
    --save_steps 500
    --save_total_limit 3
    --save_safetensors False
    --learning_rate 1e-4
    --vlm_learning_rate 1e-5
    --weight_decay 0
    --warmup_ratio 0.03
    --max_grad_norm 1
    --lr_scheduler_type "cosine"
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
