#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit

# ======================================================================================
# 2-GPU FROM-SCRATCH action-expert training (base Qwen3-VL + a fresh action expert).
#
# NOTE: do NOT use train_action_expert_4b.sh -- its OUTPUT_DIR already has checkpoints, so it
# would auto-RESUME instead of starting fresh (and that resume breaks for ZeRO-2 across a
# GPU-count change). This points at a NEW empty OUTPUT_DIR, so there is nothing to resume and
# training starts from scratch. The original run is left untouched.
#
# 2 GPUs -> effective batch = per_device(4) x grad_accum(4) x 2 GPUs = 32 (was 16 on 1 GPU).
# Uses --max_steps (NOT --num_train_epochs) so #updates + the cosine schedule are fixed
# regardless of GPU count.
#
# To warm-start from a checkpoint instead, add:  --init_from <dir with pytorch_model.bin>
# 
# USAGE: ./scripts/train_action_expert_4b_2gpu.sh   (set RTC_MAX_DELAY in the config block below)
# ======================================================================================

CUSTOM_CACHE_DIR="/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache"
mkdir -p "$CUSTOM_CACHE_DIR/huggingface" "$CUSTOM_CACHE_DIR/triton"
export HF_HOME="$CUSTOM_CACHE_DIR/huggingface"
export TRITON_CACHE_DIR="$CUSTOM_CACHE_DIR/triton"

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
# Inside a 2-GPU Slurm cgroup, nvidia-smi sees exactly the allocated GPUs -> 2 ranks.
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

MODEL_PATH="Qwen/Qwen3-VL-4B-Instruct"
# NEW empty output dir -> nothing to auto-resume -> trains FROM SCRATCH. Keep it empty.
OUTPUT_DIR="/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/qwen3_4b_ae_hist_subpred_0717merged"
CACHE_DIR="$CUSTOM_CACHE_DIR/huggingface"
ROBOT_DATA_DIRS="/iris/projects/humanoid/trossen_data/0717_green_yellow_block_mem_merged"
RUN_NAME="qwen3vl_4b_ae_hist_subpred_0717merged_bs32"

# Training-time RTC (arXiv:2512.05964): set RTC_MAX_DELAY below to train with action-prefix
# conditioning, d ~ Uniform[0, RTC_MAX_DELAY] control ticks. Pick it >= the real deploy
# latency in ticks (latency_s * 30 Hz) and <= H - exec_horizon. NOTE: in this
# predict-subtask mode, per-chunk subtask generation adds ~0.3-0.9 s on top of the ~0.6 s
# expert pass -- likely too slow for RTC at 30 Hz; the subtask-input variants are the
# realistic RTC deployment. 0 = off (vanilla, bit-identical to before). RTC runs get
# their own OUTPUT_DIR / RUN_NAME suffix so they never mix with the vanilla run.
RTC_MAX_DELAY=20   # <-- edit me: 0 = RTC off (vanilla); >0 trains with d ~ Uniform[0, this]
if [ "$RTC_MAX_DELAY" -gt 0 ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_rtc${RTC_MAX_DELAY}"
    RUN_NAME="${RUN_NAME}_rtc${RTC_MAX_DELAY}"
fi

# Subtask insulation: True (default) = expert attends the subtask turn (pi0.5-style).
# False = the expert conditions on images+state ONLY; the subtask stays a VLM co-training
# signal. Serve such checkpoints with --insulated_subtask (+ optional --subtask_every N) --
# the expert then never waits for subtask generation (~130 ms/request saved). This is an
# ABLATION: explicit subtask conditioning may help task performance; A/B before adopting.
EXPERT_ATTENDS_SUBTASK=False   # <-- edit me: False = implicit HL (insulated); True = pi0.5-style
if [ "$EXPERT_ATTENDS_SUBTASK" != "True" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_subinsul"
    RUN_NAME="${RUN_NAME}_subinsul"
fi

# SmolVLA-style layer skipping (arXiv:2506.01844): the action expert uses only the first N
# VLM layers (expert layer i attends VLM layer i). SmolVLA's verdict is N = L/2 (half) --
# near-parity task quality at ~half the cost. For Qwen3-VL-4B (L=36) that is 18. 0 = use all
# 36 (current behavior). At serve time with implicit HL (no subtask decode) the VLM early-exits
# at layer N, so the two optimizations compound. Must match the value used at serve time.
EXPERT_VLM_LAYERS=18   # <-- edit me: 0 = all 36 layers; 18 = SmolVLA half-depth
if [ "$EXPERT_VLM_LAYERS" -gt 0 ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_L${EXPERT_VLM_LAYERS}"
    RUN_NAME="${RUN_NAME}_L${EXPERT_VLM_LAYERS}"
fi

# Visual token budget. These are PIXEL AREAS, not token counts (see argument.py:19-30).
# Conversion on Qwen3-VL: video 1 token = 32*32*2 = 2048 px, stills 1 token = 32*32 = 1024 px.
#   tokens per history frame ~= VIDEO_MAX_PIXELS / (num_frames * 2048), minus ~15% to rounding.
# The old default (1024*28*28 = 802,816) was Qwen2 arithmetic (28 = 14px patch * 2) and gave
# only 33 tok/frame -- HALF of the 64-token floor Qwen3-VL itself ships in its
# preprocessor_config.json. 1,600,000 puts history frames at 512x288 = 72 tok/frame.
# The wrist needs BOTH knobs raised: WRIST_MAX_PIXELS is our own pre-resize in
# robot_data.py:463, MAX_PIXELS is the processor's ceiling; the smaller one wins, so raising
# one alone does nothing. 131,072 -> 480x256 = 120 tokens (was 45).
# Verify any change with: python scripts/trace_image_pipeline.py
VIDEO_MAX_PIXELS=1600000   # <-- edit me: 802816 = old 33 tok/frame; 2097152 = 95 tok/frame
WRIST_MAX_PIXELS=131072
MAX_PIXELS=131072
OUTPUT_DIR="${OUTPUT_DIR}_vis$((VIDEO_MAX_PIXELS / 100000))"
RUN_NAME="${RUN_NAME}_vis$((VIDEO_MAX_PIXELS / 100000))"

# ---- Optimizer recipe v2 (2026-07-28 audit vs openpi/SmolVLA; see ACTION_EXPERT.md) ----
# * weight decay: --weight_decay 0 is now actually honored (a trainer bug let torch's
#   default 0.01 apply silently to both param groups; fixed in ActionExpertTrainer)
# * adam_beta2 0.95 (was HF default 0.999): matched memory lengths for Adam's two running
#   averages -- what openpi, SmolVLA, and LLM training use; prevents oversized steps for
#   ~hundreds of steps after gradient-regime shifts
# * EMA 0.99 on the expert+heads (ExpertEMACallback): the checkpoint carries ema_expert.pt
#   and inference AUTOMATICALLY serves the averaged weights, like openpi/Diffusion Policy.
#   The raw iterate is what flickers; the EMA copy is the deployable policy.
# * 10k steps (was 26k): 10k x batch 32 = 320k samples ~= 7 passes over the 43k-timestep
#   dataset; cosine anneals to 0 AT 10k, so let it FINISH -- a mid-schedule checkpoint is
#   a noisy iterate (measured: 5-10% of motion range was noise-seed variance at 15.5k/26k).
#   EMA makes mid-run checkpoints servable anyway if you must stop early.
EMA_DECAY=0.99
OUTPUT_DIR="${OUTPUT_DIR}_opt2"
RUN_NAME="${RUN_NAME}_opt2"

# Input-dump sanity check: at startup, rank 0 writes the first few FULLY-PROCESSED model
# inputs to <OUTPUT_DIR>/input_dumps/ -- PNGs reconstructed from pixel_values (exactly what
# the vision tower sees; *_compare.png = [native | model-view upscaled]) plus report.txt
# with the token/compression breakdown. Start training, wait for the "[input-dump]" console
# lines, inspect, and kill the run if you only wanted the check. Comment out to disable.
export QWEN_DUMP_MODEL_INPUTS="${OUTPUT_DIR}/input_dumps"
export QWEN_DUMP_MODEL_INPUTS_N=2   # items per dataloader worker process

args=(
    --deepspeed scripts/zero2_action_expert.json
    --model_name_or_path "$MODEL_PATH"
    --output_dir "$OUTPUT_DIR"
    --cache_dir "$CACHE_DIR"
    # ---- robot data (identical to the source run) ----
    --robot_data_dirs "$ROBOT_DATA_DIRS"
    --camera cam_high
    --num_frames 10 # TODO: change me as needed
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
    # ---- knowledge insulation + FAST (identical to the source run) ----
    --train_vlm True
    --predict_subtask True # TODO: Change as needed
    --use_fast_tokens True
    # Refit on the 0714 merged dataset (fresh norm stats); the old fast_tokenizer_trossen is
    # kept untouched for inference with pre-0714 checkpoints.
    --fast_tokenizer_path /iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717merged
    --fast_loss_weight 1.0
    # ---- visual token budget (see the block above) ----
    --video_max_pixels "$VIDEO_MAX_PIXELS"
    --max_pixels "$MAX_PIXELS"
    --min_pixels 784
    # ---- training-time RTC (0 = off; see RTC_MAX_DELAY above) ----
    --rtc_prefix_max_length "$RTC_MAX_DELAY"
    # ---- subtask insulation (see EXPERT_ATTENDS_SUBTASK above) ----
    --expert_attends_subtask "$EXPERT_ATTENDS_SUBTASK"
    # ---- SmolVLA layer skipping (see EXPERT_VLM_LAYERS above) ----
    --expert_num_layers "$EXPERT_VLM_LAYERS"
    # ---- training ----
    --bf16
    # optimizer recipe v2 (see block above)
    --adam_beta2 0.95
    --ema_decay "$EMA_DECAY"
    # 2 GPUs x per_device 4 x grad_accum 4 = effective batch 32. (grad_accum 2 -> 16.)
    --max_steps 10000
    --per_device_train_batch_size 4
    --gradient_accumulation_steps 4
    --eval_strategy "no"
    --save_strategy "steps"
    --save_steps 500
    --save_total_limit 3
    --save_safetensors False
    # Keep 1e-4/1e-5 (safe). Linear-scaling would allow up to ~2e-4 for 2x batch -- bump only
    # if it learns too slowly.
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
