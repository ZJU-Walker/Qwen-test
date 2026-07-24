#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit

# ======================
# Advanced Cache Routing  (ported from sft_3b.sh)
# ======================
# Define your high-capacity storage folder here
CUSTOM_CACHE_DIR="/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache"

# Create the folders if they don't exist yet
mkdir -p "$CUSTOM_CACHE_DIR/huggingface"
mkdir -p "$CUSTOM_CACHE_DIR/triton"

# Tell Hugging Face and Triton to use these new fast/large folders
export HF_HOME="$CUSTOM_CACHE_DIR/huggingface"
export TRITON_CACHE_DIR="$CUSTOM_CACHE_DIR/triton"

# ======================
# Distributed Configuration  (ported from sft_3b.sh)
# ======================
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

# ======================
# Path Configuration  (ported from sft_3b.sh)
# ======================
MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"   # HuggingFace model ID
OUTPUT_DIR="/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/qwen3_4b"
CACHE_DIR="$CUSTOM_CACHE_DIR/huggingface"

# ======================
# Dataset Configuration
# ======================
# Make sure to update this to your actual dataset name from the registry!
DATASETS="your_dataset%100"

RUN_NAME="qwen3vl_4b"

# ======================
# Training Hyperparameters Array
# (values kept identical to the original sft_qwen3_4b.sh defaults from Qwen)
# ======================
args=(
    --deepspeed scripts/zero3.json
    --model_name_or_path "$MODEL_PATH"
    --dataset_use "$DATASETS"
    --output_dir "$OUTPUT_DIR"
    --cache_dir "$CACHE_DIR"
    --data_flatten True
    --tune_mm_vision False
    --tune_mm_mlp True
    --tune_mm_llm True
    --bf16
    --num_train_epochs 2500
    --per_device_train_batch_size 4
    --per_device_eval_batch_size 8
    --gradient_accumulation_steps 4
    --max_pixels 50176
    --min_pixels 784
    --eval_strategy "no"
    --save_strategy "steps"
    --save_steps 500
    --save_total_limit 3
    --learning_rate 2e-7
    --weight_decay 0
    --warmup_ratio 0.03
    --max_grad_norm 1
    --lr_scheduler_type "cosine"
    --logging_steps 1
    --model_max_length 8192
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --run_name "$RUN_NAME"
    --report_to wandb
)

# Launch command
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_qwen.py "${args[@]}"
