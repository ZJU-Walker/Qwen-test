#!/bin/bash
#SBATCH --job-name=abc_convert
#SBATCH --account=iris
#SBATCH --partition=iris
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --output=/iris/projects/humanoid/abc_data/abc130k_annot20/logs/shard_%a.out
# Shards the ABC-130k conversion across nodes: shard i converts tasks[i::N_SHARDS].
# Each node gets its own IP (HF CDN throttles per IP) and node-local /tmp scratch.
# Run the final merge afterwards:
#   uv run scripts/abc_convert_lerobot.py --manifest .../manifest.json --out ... --skip-convert
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_DISABLE_XET=1

N_SHARDS=8
OUT=/iris/projects/humanoid/abc_data/abc130k_annot20
MANIFEST=$OUT/manifest.json
cd /iris/projects/humanoid/brian_qwen_test/Qwen3-VL/qwen-vl-finetune

TASKS=$(python3 - "$SLURM_ARRAY_TASK_ID" "$N_SHARDS" <<'EOF'
import json, sys
shard, n = int(sys.argv[1]), int(sys.argv[2])
tasks = [t["task"] for t in json.load(open("/iris/projects/humanoid/abc_data/abc130k_annot20/manifest.json"))["tasks"]]
print(" ".join(tasks[shard::n]))
EOF
)
echo "shard $SLURM_ARRAY_TASK_ID on $(hostname): $(echo $TASKS | wc -w) tasks"

uv run scripts/abc_convert_lerobot.py \
    --manifest "$MANIFEST" \
    --out "$OUT" \
    --workers 8 \
    --only-tasks $TASKS \
    --report-name "conversion_report_shard${SLURM_ARRAY_TASK_ID}.json" \
    --no-finalize \
    --scratch "/tmp/abc_convert_${UID}_s${SLURM_ARRAY_TASK_ID}"
