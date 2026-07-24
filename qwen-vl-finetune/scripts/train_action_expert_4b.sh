#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit

# ======================
# Cache Routing (same as sft_qwen3_4b_bk.sh)
# ======================
CUSTOM_CACHE_DIR="/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache"
mkdir -p "$CUSTOM_CACHE_DIR/huggingface" "$CUSTOM_CACHE_DIR/triton"
export HF_HOME="$CUSTOM_CACHE_DIR/huggingface"
export TRITON_CACHE_DIR="$CUSTOM_CACHE_DIR/triton"

# ======================
# Distributed Configuration
# ======================
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

# ======================
# Paths
# ======================
MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
# NOTE: use a fresh OUTPUT_DIR whenever the model shape changes (e.g. enabling FAST
# resizes embeddings 151936->152695). The script auto-resumes from checkpoint-* in this
# dir, and resuming across a shape change fails with a state_dict size mismatch. The
# earlier non-FAST run lives in .../qwen3_4b_action_expert -- keep it.
OUTPUT_DIR="/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/qwen3_4b_action_expert_fast"
CACHE_DIR="$CUSTOM_CACHE_DIR/huggingface"

# Comma-separated LeRobot dataset roots. Same merged dataset as the openpi
# pi05_trossen_memory config (repo_id="trossen_data/0528_green_yellow_block_mem_merged_bk").
ROBOT_DATA_DIRS="/iris/projects/humanoid/trossen_data/0528_green_yellow_block_mem_merged_bk"

RUN_NAME="qwen3vl_4b_action_expert"

args=(
    # zero2 with a concrete reduce_bucket_size: the "auto" value needs
    # model.config.hidden_size, which our plain nn.Module wrapper doesn't expose.
    --deepspeed scripts/zero2_action_expert.json
    --model_name_or_path "$MODEL_PATH"
    --output_dir "$OUTPUT_DIR"
    --cache_dir "$CACHE_DIR"
    # ---- robot data ----
    --robot_data_dirs "$ROBOT_DATA_DIRS"
    --camera cam_high
    --num_frames 10
    --frame_stride 10
    # Right arm only (dims 7..13): the left arm is stationary in this dataset.
    # action_dim must equal the number of selected dims; delta_mask applies to them.
    --active_dims "7:14"
    --action_dim 7
    --delta_mask "6,-1"
    --action_horizon 50
    --use_delta_actions True
    # Single current-timestep wrist still (no history) alongside the cam_high history.
    --wrist_cameras "cam_right_wrist"
    --wrist_max_pixels 50176
    --default_prompt "waiting"
    # openpi trains the policy on ALL demos (no holdout); set e.g. 0.75 to hold out
    # the last 25% of episodes like the subtask-prediction prototype did.
    --train_split 1.0
    # ---- knowledge insulation ----
    # train_vlm co-trains the VLM (expert flow-grads stay insulated via detached KV).
    # A VLM loss is required when train_vlm=True: subtask CE and/or FAST action CE.
    --train_vlm True
    --predict_subtask True
    # ---- FAST action tokens (arXiv:2501.09747) as VLM action supervision (KI recipe) ----
    # Fit the tokenizer first: python -m scripts.train_fast_tokenizer --output_dir <path>
    # (with the SAME active_dims/delta_mask/action_horizon as here). Set to False to disable.
    --use_fast_tokens True
    --fast_tokenizer_path /iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen
    --fast_loss_weight 1.0
    # ---- image/video processing (matches existing prototype) ----
    --max_pixels 50176
    --min_pixels 784
    # ---- training ----
    --bf16
    --num_train_epochs 20000
    --per_device_train_batch_size 4
    --gradient_accumulation_steps 4
    --eval_strategy "no"
    --save_strategy "steps"
    --save_steps 500
    --save_total_limit 3
    --save_safetensors False
    # learning_rate applies to the from-scratch expert; vlm_learning_rate applies to
    # the pretrained VLM when --train_vlm True (finetuning LRs must be much smaller).
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
    # Keep workers alive across epochs. With only ~61 episodes an "epoch" is ~4 optimizer
    # steps, so without this, workers respawn (re-pickling the dataset + FAST tokenizer)
    # every ~4 steps -> a ~170s stall each time. Persistent workers avoid the respawn.
    --dataloader_persistent_workers True
    --run_name "$RUN_NAME"
    --report_to wandb
)

torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_action_expert.py "${args[@]}"
