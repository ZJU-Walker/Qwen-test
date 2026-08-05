#!/bin/bash
# Detached launch of the ee6d run inside a 2-GPU slurm allocation.
#
#   ./scripts/launch_ee6d_2gpu.sh [jobid]     (default jobid 16452866)
#
# The training script itself is GPU-count-invariant (TARGET_BATCH=64 with derived
# accumulation: 2 GPUs -> accum 8 -> 4 x 8 x 2 = 64), so this wrapper only has to
# (1) attach to the right job, (2) refuse to double-launch, (3) survive shell exit.
set -euo pipefail

JOBID=${1:-16452993}
FT=/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune
LOG="$FT/train_ee6d_2gpu.log"

NGPU=$(srun --jobid "$JOBID" --overlap bash -c 'nvidia-smi --list-gpus | wc -l')
if [ "$NGPU" != "2" ]; then
    echo "ERROR: job $JOBID sees $NGPU GPU(s), expected 2" >&2
    exit 1
fi

# Never run two writers of the ee6d OUTPUT_DIR at once (checkpoint clobbering).
if srun --jobid "$JOBID" --overlap bash -c 'pgrep -af train_action_expert.py || true' \
        | grep -q '_ee6d'; then
    echo "ERROR: an ee6d training process is already alive on this node -- kill it first" >&2
    exit 1
fi

echo "Launching ee6d on job $JOBID: 2 GPUs, effective batch 64 (4 per-device x accum 8 x 2)"
# Detached srun gets a bare shell: put the qwen3vl env (torchrun etc.) on PATH explicitly.
# NUM_WORKERS=8 per rank assumes a >=24-CPU allocation (2 ranks x 9 procs on 24 cores);
# preprocessing (video decode + aug + FK) is the CPU-bound stage that starves the GPUs.
nohup srun --jobid "$JOBID" --overlap bash -c \
    "export PATH=/iris/projects/humanoid/miniconda3/envs/qwen3vl/bin:\$PATH && export NUM_WORKERS=${NUM_WORKERS:-8} && cd $FT && ./scripts/train_action_expert_4b_ee6d.sh" > "$LOG" 2>&1 &
echo "detached (srun PID $!) -- survives this shell; killing that PID stops training"
echo "monitor:  tail -f $LOG"
