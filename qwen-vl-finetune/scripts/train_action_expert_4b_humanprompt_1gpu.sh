#!/bin/bash
# Single-GPU variant of train_action_expert_4b_humanprompt.sh, safe to run CONCURRENTLY
# with the 2-GPU run on a different allocation: RUN_TAG=1gpu gives it its own output dir
# (..._constlr_1gpu) and wandb run name; everything else (recipe, effective batch 64 via
# derived accum 32) is inherited from the main script -- single source of truth.
#
#   NPROC_PER_NODE=1 pinned; pick the GPU with CUDA_VISIBLE_DEVICES (default: job's GPU 0)
#   NUM_WORKERS defaults to 12 (a lone rank can use the whole 16-cpu allocation)
set -euo pipefail
cd "$(dirname "$0")"
export NPROC_PER_NODE=1
export RUN_TAG="${RUN_TAG:-1gpu}"
export NUM_WORKERS="${NUM_WORKERS:-12}"
exec bash train_action_expert_4b_humanprompt.sh
