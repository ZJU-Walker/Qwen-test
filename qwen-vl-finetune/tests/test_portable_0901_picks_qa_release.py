#!/usr/bin/env python3
"""Offline contract checks for the fresh 0901 pickup-QA portable release."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PORTABLE = ROOT / "portable"
BOOTSTRAP = PORTABLE / "bootstrap_and_submit_4h100_0901_picks_qa.sh"
PUBLISHER = PORTABLE / "publish_gcs_0901_picks_qa_release.sh"
README = PORTABLE / "RUN_0901_PICKS_QA_ON_NEW_4H100_CLUSTER.md"


def main() -> None:
    for script in (BOOTSTRAP, PUBLISHER):
        assert script.is_file(), script
        subprocess.run(["bash", "-n", script], check=True)

    bootstrap = BOOTSTRAP.read_text()
    publisher = PUBLISHER.read_text()
    readme = README.read_text()

    # Four-H100 dense20 + QA training, fresh on first launch.
    assert "train_action_expert_4b_sort_0901_picks_dense20_fresh_qa_4h100.sh" in bootstrap
    assert "expected 4 GPUs" in bootstrap and '"H100" not in p.name' in bootstrap
    assert "export NPROC_PER_NODE=4 PER_DEVICE_BATCH=1" in bootstrap
    assert "standalone_robot_qa" in publisher
    assert '"standalone_robot_qa": 135' in publisher
    assert '"normal_prompted": 371' in publisher
    assert '"total": 641' in publisher
    assert "checkpoint-3000" not in bootstrap
    assert "resume/" not in publisher
    assert "No prior checkpoint" in bootstrap
    assert '"git", "diff", "--quiet"' in bootstrap
    assert "DENSE20_PROCESSOR_PATH" in bootstrap
    assert "preflight_dense20_prompt_contract_cpu.py" in bootstrap

    # Exact data/artifact release, including the dataset-local QC exclusion.
    for name in ("0901ball", "0901green", "0901grey"):
        assert name in publisher
    assert "sort0901-picks.tar.zst" in bootstrap
    assert "sort0901-picks.tar.zst" in publisher
    assert "fast_tokenizer_trossen_0824sort_0827ball_0901picks_ee6d" in bootstrap
    assert "0901ball_train_exclude_episodes.json" in publisher
    assert "SHA256SUMS.assets" in bootstrap and "SHA256SUMS.assets" in publisher
    assert publisher.rfind("release-manifest.json") > publisher.rfind("SHA256SUMS.assets")

    # Secrets stay interactive; the one-command Slurm handoff preserves the W&B ID.
    assert 'read -r -s -p "W&B API key: "' in bootstrap
    assert "unset WANDB_API_KEY" in bootstrap
    assert "prepare_wandb_run_id" in bootstrap
    assert "sbatch" in bootstrap and "--inside-slurm" in bootstrap
    assert "WANDB_API_KEY_HERE" not in bootstrap

    # User guide covers preparation, publication, launch, and restart.
    for command in (
        "fit_sort_0901_picks_fast_tokenizer.sh",
        "publish_gcs_0901_picks_qa_release.sh",
        "bootstrap_and_submit_4h100_0901_picks_qa.sh",
        "train_action_expert_4b_sort_0901_picks_dense20_fresh_qa_4h100.sh",
    ):
        assert command in readme
    assert "episodes 8 and 11" in readme
    assert "641 sampling records" in readme

    print("PASS: portable fresh 0901 pickup-QA four-H100 release contract")


if __name__ == "__main__":
    main()
