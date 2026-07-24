#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit

# ======================
# Advanced Cache Routing
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
# Distributed Configuration
# ======================
MASTER_ADDR="127.0.0.1"
MASTER_PORT=$(shuf -i 20000-29999 -n 1)
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

# ======================
# Path Configuration
# ======================
# Changed from the fake local path to the official Hugging Face Hub ID
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR="/iris/projects/humanoid/ke/Qwen3-VL/checkpoints"
CACHE_DIR="$CUSTOM_CACHE_DIR/huggingface"

# ======================
# Model Configuration
# ======================
# Make sure to update this to your actual dataset name from the registry!
DATASETS="your_dataset%100" 


# ======================
# Training Hyperparameters Array
# ======================
args=(
    --model_name_or_path "$MODEL_PATH"
    --tune_mm_llm True
    --tune_mm_vision False
    --tune_mm_mlp False
    --dataset_use "$DATASETS"
    --output_dir "$OUTPUT_DIR"
    --cache_dir "$CACHE_DIR"
    --bf16
    --per_device_train_batch_size 4
    --gradient_accumulation_steps 4
    --learning_rate 2e-7
    --mm_projector_lr 1e-5
    --vision_tower_lr 1e-6
    --optim adamw_torch
    --model_max_length 4096
    --data_flatten True
    --data_packing True
    --max_pixels 451584
    --min_pixels 12544
    --video_fps 2
    --video_max_frames 8
    --video_min_frames 4
    --video_max_pixels 1304576
    --video_min_pixels 200704
    --num_train_epochs 3000
    --warmup_ratio 0.03
    --lr_scheduler_type "cosine"
    --weight_decay 0.01
    --logging_steps 10
    --save_steps 500
    --save_total_limit 3
    --lora_enable False
    --lora_r 8
    --lora_alpha 16
    --lora_dropout 0.0
    --deepspeed scripts/zero3.json
)

# Launch command
torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_qwen.py "${args[@]}"