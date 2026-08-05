#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit

# ======================================================================================
# EE-6D COMBINED run (2026-08-02; +0804 DAgger 2026-08-04): constant-lr recipe on
# 0717 merged + 0731 + THREE 0804 DAgger sets (104 eps, recorded at 30 Hz with the ee6d
# 2-wrist client; is_intervention -> min_start gating trains ONLY the correction segments,
# with image history reaching back into the policy's own mistake), with THREE features:
#
#   * ACTION SPACE ee6d: states/actions converted joints -> [xyz, 6D rotation, jaw]
#     (10 dims) by FK at load; deltas = naive per-dim subtraction (delta_mask 9,-1),
#     measured valid and most noise-robust on this data (tests/probe_rotation_*.py).
#     Serving converts back via Gram-Schmidt + seeded DLS IK; robot i/o stays joints.
#     Full-pipeline round trip costs <1 micron (tests/smoke_test_ee6d_core.py).
#   * pi05-style IMAGE AUGMENTATION: per-clip crop 95% + rotate +-5 deg always on
#     (top camera); color jitter (brightness/contrast/hue) fires at p=0.5 per camera,
#     matching pi05's real augmax behavior; jitter-only wrists (scripts/viz_augmentation.py).
#   * BOTH wrist cameras (right + left current stills; wrist_cameras stamped in
#     visual_budget.json so serving auto-matches).
#
#   * Mixed subtask supervision unchanged: 0717 labeled, 0731 unlabeled (FAST-only).
#   * NEW FAST tokenizer artifact fast_tokenizer_trossen_0717m_0731gy_0804dag_ee6d
#     (10-dim chunks, all five dirs, max_fast_len 151) with its frozen norm_stats.json.
#   * --max_steps 40000 is a CEILING: stop on plateau. Every checkpoint is servable.
# ======================================================================================

CUSTOM_CACHE_DIR="/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache"
mkdir -p "$CUSTOM_CACHE_DIR/huggingface" "$CUSTOM_CACHE_DIR/triton"
export HF_HOME="$CUSTOM_CACHE_DIR/huggingface"
export TRITON_CACHE_DIR="$CUSTOM_CACHE_DIR/triton"

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
# Overridable so multiple runs can share a node (NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 ...).
NPROC_PER_NODE=${NPROC_PER_NODE:-$(nvidia-smi --list-gpus | wc -l)}

# Effective batch is FIXED at 64 regardless of GPU count: accumulation is derived so
# 1 GPU -> accum 16, 2 GPUs -> accum 8, etc. Same gradients either way; more GPUs just
# finish each optimizer step proportionally faster.
TARGET_BATCH=64
PER_DEVICE_BATCH=4
if [ $((TARGET_BATCH % (PER_DEVICE_BATCH * NPROC_PER_NODE))) -ne 0 ]; then
    echo "ERROR: TARGET_BATCH=$TARGET_BATCH not divisible by $PER_DEVICE_BATCH x $NPROC_PER_NODE GPUs" >&2
    exit 1
fi
GRAD_ACCUM=$((TARGET_BATCH / (PER_DEVICE_BATCH * NPROC_PER_NODE)))

MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
OUTPUT_DIR="/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/qwen3_4b_ae_hist_subpred_0717m_0731gy_0804dag_ee6d"
CACHE_DIR="$CUSTOM_CACHE_DIR/huggingface"
ROBOT_DATA_DIRS="/iris/projects/humanoid/trossen_data/0717_green_yellow_block_mem_merged,/iris/projects/humanoid/trossen_data/0731_green_yellow_merged,/iris/projects/humanoid/trossen_data/0804_green_dagger,/iris/projects/humanoid/trossen_data/0804_yellow_dagger_part1,/iris/projects/humanoid/trossen_data/0804_yellow_dagger_part2"
FAST_TOKENIZER="/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717m_0731gy_0804dag_ee6d"
RUN_NAME="qwen3vl_4b_ae_hist_subpred_0717m_0731gy_0804dag_ee6d_bs64"

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
    # Frame-aligned past states in the prompt (9 discretized states at the history-frame
    # timesteps, edge-clamped like the frames; ~+150 prompt tokens).
    --state_history True
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
    # Effective batch 64 (2026-08-02, user request), GPU-count-invariant: see the
    # TARGET_BATCH block above.
    --per_device_train_batch_size "$PER_DEVICE_BATCH"
    --gradient_accumulation_steps "$GRAD_ACCUM"
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
    --dataloader_num_workers "${NUM_WORKERS:-4}"
    --dataloader_persistent_workers True
    --run_name "$RUN_NAME"
    --report_to wandb
)

torchrun --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_action_expert.py "${args[@]}"
