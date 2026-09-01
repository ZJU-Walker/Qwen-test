#!/usr/bin/env python3
"""Offline contract checks for the one-command GCS/Slurm portable release."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PORTABLE = ROOT / "portable"
BOOTSTRAP = PORTABLE / "bootstrap_and_submit_4h100_noqa_resume3000.sh"
PUBLISHER = PORTABLE / "publish_gcs_release.sh"
DATA_SCRIPT = ROOT / "scripts" / "sort_0827_ball_data.sh"


def run(*argv: str, env: dict[str, str] | None = None) -> str:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.check_output(argv, cwd=ROOT, env=merged, text=True)


def main() -> None:
    subprocess.run(["bash", "-n", BOOTSTRAP], check=True)
    subprocess.run(["bash", "-n", PUBLISHER], check=True)

    bootstrap = BOOTSTRAP.read_text()
    publisher = PUBLISHER.read_text()
    requirements = (PORTABLE / "requirements-h100-cu124.txt").read_text()

    # User-facing one-script and secret contract.
    assert "WANDB_API_KEY_HERE" in bootstrap
    assert "wandb.login" in bootstrap and "verify=True" in bootstrap
    assert "unset WANDB_API_KEY" in bootstrap
    assert "prepare_wandb_run_id" in bootstrap
    assert 'id_file="$WANDB_DIR/portable-run-id"' in bootstrap
    assert "export WANDB_RUN_ID" in bootstrap
    assert "--inside-slurm" in bootstrap and "sbatch" in bootstrap
    assert "SLURM_GPU_ARGUMENT" in bootstrap
    assert "GOOGLE_APPLICATION_CREDENTIALS" in bootstrap
    assert 'GCS_PUBLIC_READ="${GCS_PUBLIC_READ:-True}"' in bootstrap
    assert "auth/disable_credentials True" in bootstrap
    assert "Anonymous public GCS access: PASS" in bootstrap
    assert "gs://qwenfiles/qwen-sort/dense20-noqa-resume3000/v1" in bootstrap

    # Exact four-GPU resume rather than a weights-only warm start.
    assert 'CHECKPOINT_NAME="checkpoint-3000"' in bootstrap
    assert "select_latest_complete_checkpoint" in bootstrap
    assert "expected_world_size=4" in bootstrap
    assert "expected_ema_decay=0.999" in bootstrap
    assert "export INIT_FROM=" in bootstrap
    assert "--init_from" not in bootstrap
    assert "export SAVE_STEPS=500" in bootstrap
    assert "export NUM_WORKERS=4" in bootstrap
    assert "export SAVE_TOTAL_LIMIT=2" in bootstrap
    assert "standalone_pick_action_only.py" in bootstrap
    assert "training_release_paths" in bootstrap
    assert 'diff --quiet "$release_commit" "$local_commit"' in bootstrap

    # Immutable GCS release is content-addressed and marked complete last.
    assert "SHA256SUMS.assets" in publisher
    assert "SHA256SUMS.checkpoint-3000" in publisher
    assert "SHA256SUMS.run-root" in publisher
    assert publisher.rfind("release-manifest.json") > publisher.rfind("SHA256SUMS.run-root")
    assert "standalone_robot_qa_enabled" in publisher
    assert '"standalone_robot_qa": 0' in publisher
    assert "gcloud storage cp --recursive" in publisher

    # Reviewed runtime pins: CUDA 12.4 PyTorch and the exact live package versions.
    assert "torch==2.6.0 torchvision==0.21.0" in bootstrap
    for pin in (
        "transformers==4.57.6",
        "deepspeed==0.17.1",
        "accelerate==1.7.0",
        "wandb==0.27.0",
    ):
        assert pin in requirements

    # The existing /iris recipe remains the default, while portable roots override it.
    env = {
        "SORT_0824_ROOT": "/portable/data/0824_prompting",
        "SORT_0827_ROOT": "/portable/data/0827/data_0827_prompting_playdata",
        "SORT_0827_BALL_FAST_DEFAULT": "/portable/artifacts/fast",
    }
    shell = f"""
set -euo pipefail
source {DATA_SCRIPT!s}
printf '%s\n' "$SORT_0827_BALL_HUMAN_DIRS"
printf '%s\n' "$SORT_0827_BALL_UNPROMPTED_DIRS"
printf '%s\n' "$SORT_0827_BALL_FAST_DEFAULT"
printf '%s\n' "$SORT_0827_BALL_ROBOT_DIRS"
"""
    lines = run("bash", "-c", shell, env=env).splitlines()
    assert lines[0] == "/portable/data/0824_prompting/human_demo"
    assert lines[1] == "/portable/data/0827/data_0827_prompting_playdata/ball"
    assert lines[2] == "/portable/artifacts/fast"
    assert "/portable/data/0824_prompting/green_block" in lines[3]
    assert "/portable/data/0827/data_0827_prompting_playdata/ball" in lines[3]

    # Committed files contain placeholders only, never an actual-looking W&B key.
    assert "WANDB_API_KEY_HERE" in bootstrap
    assert not any(
        token.startswith("wandb_api_") or token.startswith("wapi-")
        for token in bootstrap.split()
    )

    print("PASS: portable four-H100 GCS/W&B/Slurm release contract")


if __name__ == "__main__":
    main()
