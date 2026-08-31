#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit

# ======================================================================================
# HUMAN-VIDEO-PROMPT run (2026-08-06): the language instruction is replaced by a short
# human demonstration clip. Recipe = the ee6d constant-lr recipe on gated 0717 merged:
#
#   * DATA: 0717 merged (224 eps) + 0731 merged (224 eps) -- both are block-memory
#     format (human points -> robot picks), with the pointing phase gated OUT via
#     --skip_leading_subtask waiting: the "waiting" segment is never a training
#     timestep NOR history context (hist_min clamp -- unlike DAgger, history must not
#     reach into it: it shows the human pointing at the target in the robot's own view
#     and would leak the task past the prompt video). 0717 ep 149 auto-dropped
#     (mislabeled); seven unusable 0731 recordings are waiting-only (128/129/130/133/
#     150/152/153), leaving 217 usable and 440 total with 0717.
#     0731 labels are proprioception-generated (validated on
#     0717 human labels, colors cross-checked vs brian's independent run) -- install
#     them first: python tests/gen_0731_subtask_labels.py --install
#     Norm stats + FAST fit under the same gate on BOTH datasets (frozen artifact below).
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

CUSTOM_CACHE_DIR="${CUSTOM_CACHE_DIR:-/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache}"
mkdir -p "$CUSTOM_CACHE_DIR/huggingface" "$CUSTOM_CACHE_DIR/triton"
export HF_HOME="$CUSTOM_CACHE_DIR/huggingface"
export TRITON_CACHE_DIR="$CUSTOM_CACHE_DIR/triton"
# wandb's service handshake writes a port file under TMPDIR; NFS home breaks it.
export TMPDIR=/tmp
export PYTHONUNBUFFERED=1
# An uninstrumented FlashAttention run once died with a native SIGSEGV at step 267, so
# use PyTorch SDPA as the conservative default while that one-off remains unexplained.
# Later FA2 and SDPA reproductions were both caused by QWEN_STALL_DEBUG's old periodic
# all-thread dump, not by their attention kernels. Override for an explicit FA2 run with
# QWEN_ATTN_IMPL=flash_attention_2.
export QWEN_ATTN_IMPL="${QWEN_ATTN_IMPL:-sdpa}"
# Generous collective fuse for genuinely slow initialization/checkpoint I/O. The former
# step-1 timeout was not an I/O stall: QWEN_NAN_DEBUG called ZeRO-2 safe_get_full_grad(),
# whose rank-asymmetric failure path mismatched a 390,899,200-element all-reduce with the
# Trainer metric gather. That callback now skips the collective whenever world_size > 1.
# Keep both timeout knobs because the PG initialization paths do not share one default.
export DEEPSPEED_TIMEOUT=${DEEPSPEED_TIMEOUT:-120}   # minutes
# wandb appends its run files every logged step (logging_steps 1): keep that off NFS
# so it can never stall rank 0 (metrics still sync to the wandb cloud dashboard).
# Pre-set WANDB_DIR (e.g. Modal) wins.
export WANDB_DIR="${WANDB_DIR:-/tmp/qwen_wandb}"
mkdir -p "$WANDB_DIR"
# DO NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here: with DeepSpeed
# overlap_comm (async side-stream gradient reduction) it silently corrupted parameter
# memory -- exactly 50 KiB of nan over embed_tokens rows 0-9, deterministically at
# optimizer step 3 (2026-08-06; see EmbedNanWatchCallback / QWEN_NAN_DEBUG).

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
# Pinned (not nvidia-smi): on a shared HGX node nvidia-smi ignores CUDA_VISIBLE_DEVICES
# and would fan out to all 8 GPUs. Override: NPROC_PER_NODE=1 CUDA_VISIBLE_DEVICES=0 ...
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
TORCHRUN=${TORCHRUN:-torchrun}

# Effective batch is FIXED at 64 regardless of GPU count (accumulation derived).
TARGET_BATCH=64
# V1 default is 2: its prompt + robot-history two-video sequence OOMs an H200 at 4.
# V2 overrides this after removing robot history; an H200 production one-step sweep found
# batch 8 healthy and batch 16 OOM at 138.8/139.8 GiB.
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
if [ $((TARGET_BATCH % (PER_DEVICE_BATCH * NPROC_PER_NODE))) -ne 0 ]; then
    echo "ERROR: TARGET_BATCH=$TARGET_BATCH not divisible by $PER_DEVICE_BATCH x $NPROC_PER_NODE GPUs" >&2
    exit 1
fi
GRAD_ACCUM=$((TARGET_BATCH / (PER_DEVICE_BATCH * NPROC_PER_NODE)))

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-VL-4B-Instruct}"
# Keep dataset provenance in the artifact/run name. Variants may override DATA_TAG,
# while the default remains byte-for-byte compatible with the original 0717+0731 run.
DATA_TAG="${DATA_TAG:-0717m0731}"
# Keep the historical project checkpoint location as the default, while allowing a
# storage-constrained launch wrapper to redirect only its own durable artifacts.
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/iris/projects/humanoid/ke/Qwen3-VL/checkpoints}"
OUTPUT_DIR="${CHECKPOINT_ROOT%/}/qwen3_4b_ae_humanprompt_${DATA_TAG}_ee6d"
CACHE_DIR="$CUSTOM_CACHE_DIR/huggingface"
# Modal can stage read-heavy videos on container-local SSD and override only these
# inputs. Defaults preserve the exact cluster recipe and all durable outputs remain on
# /iris. The copied trees are byte-identical, so changing paths does not change samples.
ROBOT_DATA_DIRS="${ROBOT_DATA_DIRS:-/iris/projects/humanoid/trossen_data/0717_green_yellow_block_mem_merged,/iris/projects/humanoid/trossen_data/0731_green_yellow_merged}"
HUMAN_PROMPT_DIRS="${HUMAN_PROMPT_DIRS:-green=/iris/projects/humanoid/trossen_data/green_human_prompt,yellow=/iris/projects/humanoid/trossen_data/yellow_human_prompt}"
FAST_TOKENIZER="${FAST_TOKENIZER:-/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717m0731_ee6d_gated}"
RUN_NAME="qwen3vl_4b_ae_humanprompt_${DATA_TAG}_ee6d_bs64"
# Input-contract overrides used by isolated variants. Defaults preserve the original v1
# recipe exactly. A sparse prompt variant must attach explicit Qwen3 timestamp metadata.
STATE_HISTORY="${STATE_HISTORY:-True}"
IMAGE_HISTORY="${IMAGE_HISTORY:-True}"
CURRENT_STATE_MASK_PROB="${CURRENT_STATE_MASK_PROB:-0.0}"
HUMAN_PROMPT_STRIDE="${HUMAN_PROMPT_STRIDE:-10}"
HUMAN_PROMPT_MAX_FRAMES="${HUMAN_PROMPT_MAX_FRAMES:-12}"
EXPLICIT_VIDEO_TIMESTAMPS="${EXPLICIT_VIDEO_TIMESTAMPS:-False}"
# `${VAR-default}` (without the colon) intentionally permits an explicit empty value.
# Clean pick/place recordings have no leading human-pointing phase, so their wrapper
# sets this to empty; the original pointing datasets retain the `waiting` default.
SKIP_LEADING_SUBTASK="${SKIP_LEADING_SUBTASK-waiting}"
# Optional exact-root standalone pick stream (0827 ball): empty preserves every
# historical recipe.  A wrapper that enables it must also use explicit-HL
# (EXPERT_ATTENDS_SUBTASK=True); the trainer enforces that contract.
UNPROMPTED_PICK_ONLY_DIRS="${UNPROMPTED_PICK_ONLY_DIRS-}"
STANDALONE_ROBOT_QA_ENABLED="${STANDALONE_ROBOT_QA_ENABLED:-True}"
ROBOT_QA_STRIDE="${ROBOT_QA_STRIDE:-10}"
ROBOT_QA_MAX_FRAMES="${ROBOT_QA_MAX_FRAMES:-12}"

# Training-time RTC prefix d ~ Uniform[0, max]; 10 matches the deployed delay of 8-10.
# An expert-attends-subtask variant must raise this (bins uses 20; compressed ~5-token
# answers decode in far fewer ticks than the cup task's ~17-20): serving decodes the
# assistant answer serially before the expert runs. The value is stamped into
# visual_budget.json as rtc_prefix_max_length and the server REJECTS any /infer whose
# action_prefix exceeds it -- the client's DELAY_STEPS must stay <= this value.
RTC_MAX_DELAY="${RTC_MAX_DELAY:-10}"
if [ "$RTC_MAX_DELAY" -gt 0 ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_rtc${RTC_MAX_DELAY}"
    RUN_NAME="${RUN_NAME}_rtc${RTC_MAX_DELAY}"
fi

# False (default) = subtask insulation: the expert ignores the assistant answer turn and
# serving never decodes it (VLM early-exits at L18). True = explicit-HL conditioning
# (pi0.5 Fig-13): the expert ATTENDS the answer; serving decodes it per request via the
# single-prefill generate_subtask_cached path and the full 36-layer VLM runs. Requires
# fully-labeled data (the trainer refuses attends=True with unlabeled samples) and the
# RTC_MAX_DELAY bump above. The server auto-applies the stamped mode.
EXPERT_ATTENDS_SUBTASK="${EXPERT_ATTENDS_SUBTASK:-False}"
if [ "$EXPERT_ATTENDS_SUBTASK" != "True" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_subinsul"
    RUN_NAME="${RUN_NAME}_subinsul"
else
    OUTPUT_DIR="${OUTPUT_DIR}_subattend"
    RUN_NAME="${RUN_NAME}_subattend"
fi

# Per-sample subtask-text CE averaging (each question counts equally regardless of its
# answer length). Turn on together with mixed-length QA formats.
LM_LOSS_PER_SAMPLE="${LM_LOSS_PER_SAMPLE:-False}"

EXPERT_VLM_LAYERS=18
if [ "$EXPERT_VLM_LAYERS" -gt 0 ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_L${EXPERT_VLM_LAYERS}"
    RUN_NAME="${RUN_NAME}_L${EXPERT_VLM_LAYERS}"
fi

# Visual token budget. The default remains identical to the ee6d run; an isolated
# prompt-density variant may override it and receives a distinct `_visN` artifact name.
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-1600000}"
WRIST_MAX_PIXELS=131072
MAX_PIXELS=131072
OUTPUT_DIR="${OUTPUT_DIR}_vis$((VIDEO_MAX_PIXELS / 100000))"
RUN_NAME="${RUN_NAME}_vis$((VIDEO_MAX_PIXELS / 100000))"

EMA_DECAY=0.999
OUTPUT_DIR="${OUTPUT_DIR}_constlr"
RUN_NAME="${RUN_NAME}_constlr"

# RUN_TAG: suffix for concurrent variants (e.g. RUN_TAG=1gpu alongside the 2-GPU run
# on another allocation) -- separate checkpoints dir + wandb run, identical recipe.
if [ -n "${RUN_TAG:-}" ]; then
    OUTPUT_DIR="${OUTPUT_DIR}_${RUN_TAG}"
    RUN_NAME="${RUN_NAME}_${RUN_TAG}"
fi

# Overridable incl. explicit-empty (QWEN_DUMP_MODEL_INPUTS= disables the dumps --
# on Modal they'd write PNGs to the volume, whose periodic commits can stall writers).
export QWEN_DUMP_MODEL_INPUTS="${QWEN_DUMP_MODEL_INPUTS-${OUTPUT_DIR}/input_dumps}"
export QWEN_DUMP_MODEL_INPUTS_N=2

args=(
    --deepspeed scripts/zero2_action_expert.json
    --model_name_or_path "$MODEL_PATH"
    --output_dir "$OUTPUT_DIR"
    --cache_dir "$CACHE_DIR"
    # ---- robot data: gated 0717 merged + human demo pools ----
    --robot_data_dirs "$ROBOT_DATA_DIRS"
    --camera cam_high
    # History-video geometry (read only when IMAGE_HISTORY=True; V3 = 6 frames at
    # stride 15 -> 2 frames/s over 2.5 s at the 30 fps datasets). Defaults match the
    # pre-V3 hardcoded values.
    --num_frames "${NUM_FRAMES:-10}"
    --frame_stride "${FRAME_STRIDE:-10}"
    --history_max_pixels "${HISTORY_MAX_PIXELS:-65536}"
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
    --image_history "$IMAGE_HISTORY"
    --state_history "$STATE_HISTORY"
    --current_state_mask_prob "$CURRENT_STATE_MASK_PROB"
    # ---- the human-video-prompt setup ----
    --skip_leading_subtask "$SKIP_LEADING_SUBTASK"
    # 440-sample "epochs" are ~7 optimizer steps; the dataloader pipeline restart at
    # every boundary starves the GPU. Virtual epochs make boundaries ~50x rarer.
    --dataset_epoch_multiplier 50
    --human_prompt_dirs "$HUMAN_PROMPT_DIRS"
    --human_prompt_stride "$HUMAN_PROMPT_STRIDE"
    --human_prompt_max_frames "$HUMAN_PROMPT_MAX_FRAMES"
    --human_prompt_source_fps 30
    --explicit_video_timestamps "$EXPLICIT_VIDEO_TIMESTAMPS"
    --human_prompt_holdout "${HUMAN_PROMPT_HOLDOUT:-4}"
    --subtask_question "${SUBTASK_QUESTION:-which colored block did the human demonstrate picking up?}"
    # ---- segment-level prompts + bins QA (bins_task_plan.md; defaults preserve the
    # original color-pool recipe byte-for-byte) ----
    --human_prompt_segments "${HUMAN_PROMPT_SEGMENTS:-False}"
    --human_prompt_full_episode_prob "${HUMAN_PROMPT_FULL_EP_PROB:-0.0}"
    --subtask_format_mix "${SUBTASK_FORMAT_MIX-}"
    # Which QA module the mix/answers/pool keys come from: "bins" (0817/0820 bins task)
    # or "sort" (0824 three-tray sorting). Checkpoint-stamped; serving reads it back.
    --subtask_task "${SUBTASK_TASK:-bins}"
    # 'where' format only (sort): share of questions naming an object the drawn demo
    # never showed, answered with the abstention "not shown".
    --qa_where_absent_prob "${QA_WHERE_ABSENT_PROB:-0.2}"
    # Hold out human demos per POOL KEY instead of per dataset (needed when a single
    # human dataset holds every configuration, grouped on disk).
    --human_prompt_holdout_per_key "${HUMAN_PROMPT_HOLDOUT_PER_KEY:-False}"
    # Robot labels file (subtask_labels_4phase.json = pick/place split, 'phase' QA).
    --robot_subtask_labels_file "${ROBOT_SUBTASK_LABELS_FILE:-subtask_labels.json}"
    # Track-A order samples (language-only; requires 4-phase labels + a QA mix).
    --order_sample_prob "${ORDER_SAMPLE_PROB:-0.0}"
    # Exact opted-in one-segment pick roots yield paired action + robot-QA records.
    --unprompted_pick_only_dirs "$UNPROMPTED_PICK_ONLY_DIRS"
    --standalone_robot_qa_enabled "$STANDALONE_ROBOT_QA_ENABLED"
    --robot_qa_stride "$ROBOT_QA_STRIDE"
    --robot_qa_max_frames "$ROBOT_QA_MAX_FRAMES"
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
    --lm_loss_per_sample "$LM_LOSS_PER_SAMPLE"
    # ---- SmolVLA layer skipping ----
    --expert_num_layers "$EXPERT_VLM_LAYERS"
    # ---- training: CONSTANT LR + EMA 0.999 ----
    --bf16
    --adam_beta2 0.95
    --ema_decay "$EMA_DECAY"
    --lr_scheduler_type constant_with_warmup
    --warmup_steps 300
    # Overridable so a pre-launch dry run can execute the REAL recipe for a few steps
    # (MAX_STEPS=3) instead of a hand-built approximation of it.
    --max_steps "${MAX_STEPS:-30000}"
    --per_device_train_batch_size "$PER_DEVICE_BATCH"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --eval_strategy "no"
    --save_strategy "steps"
    --save_steps "${SAVE_STEPS:-2000}"
    --save_total_limit "${SAVE_TOTAL_LIMIT:-3}"
    --save_safetensors False
    --learning_rate 1e-4
    --vlm_learning_rate 1e-5
    --weight_decay 0
    --max_grad_norm 1
    --logging_steps 1
    --ddp_timeout 7200
    --model_max_length 8192
    --gradient_checkpointing True
    --torch_compile "${TORCH_COMPILE:-False}"
    --dataloader_num_workers "${NUM_WORKERS:-4}"
    --dataloader_prefetch_factor 4
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

# A new input contract may warm-start model weights while deliberately resetting the
# optimizer, scheduler, and step counter. Always forward this intent to Python: its
# validated checkpoint selector decides whether to exact-resume instead. A shell glob
# cannot distinguish a complete checkpoint from a directory left by a kill mid-save.
INIT_FROM="${INIT_FROM-}"
if [ -n "$INIT_FROM" ]; then
    args+=(--init_from "$INIT_FROM")
fi

"$TORCHRUN" --nproc_per_node=$NPROC_PER_NODE \
         --master_addr=$MASTER_ADDR \
         --master_port=$MASTER_PORT \
         qwenvl/train/train_action_expert.py "${args[@]}"
