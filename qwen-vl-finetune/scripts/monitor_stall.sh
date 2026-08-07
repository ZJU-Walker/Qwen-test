#!/bin/bash
# Sidecar stall monitor: every 20s, append GPU utilization + training-process states
# to /tmp/monitor_stall.log. Run in the background next to a training launch:
#     bash scripts/monitor_stall.sh &
# Kill it when done (kill %1 / its PID). Read together with the per-rank stack logs
# /tmp/qwen_stall_rank*.log written by QWEN_STALL_DEBUG=1.
OUT=/tmp/monitor_stall.log
echo "===== monitor start $(date) on $(hostname) =====" >> "$OUT"
while true; do
    {
        echo "--- $(date +%H:%M:%S) ---"
        nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null
        # STAT: R running, S sleeping, D uninterruptible IO (NFS hangs show D)
        ps -u "$USER" -o pid,stat,etime,rss,cmd --sort=pid 2>/dev/null \
            | grep -E "train_action_expert|torchrun" | grep -v grep
    } >> "$OUT" 2>&1
    sleep 20
done
