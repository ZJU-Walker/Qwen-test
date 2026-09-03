#!/usr/bin/env python3
"""Shell-contract test for fresh dense20 0901 pickup-QA training."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
WRAPPER = SCRIPTS / "train_action_expert_4b_sort_0901_picks_dense20_fresh_qa_4h100.sh"
BASE = SCRIPTS / "train_action_expert_4b_humanprompt.sh"


def _value(lines: list[str], flag: str) -> str:
    return lines[lines.index(flag) + 1]


def main() -> None:
    for script in (
        SCRIPTS / "sort_0901_picks_data.sh",
        SCRIPTS / "fit_sort_0901_picks_fast_tokenizer.sh",
        SCRIPTS / "train_action_expert_4b_sort_0901_picks.sh",
        WRAPPER,
    ):
        subprocess.run(["bash", "-n", script], check=True)

    with tempfile.TemporaryDirectory() as tmp:
        temp = Path(tmp)
        fake_torchrun = temp / "torchrun"
        fake_torchrun.write_text("#!/bin/bash\nprintf '%s\\n' \"$@\"\n")
        fake_torchrun.chmod(0o755)
        fast = temp / "fast"
        fast.mkdir()
        (fast / "norm_stats.json").write_text("{}\n")
        checkpoints = temp / "checkpoints"

        env = dict(
            os.environ,
            TORCHRUN=str(fake_torchrun),
            FAST_TOKENIZER=str(fast),
            CHECKPOINT_ROOT=str(checkpoints),
            NPROC_PER_NODE="4",
            PER_DEVICE_BATCH="1",
            NUM_WORKERS="0",
            RUN_TAG="contract",
        )
        for key in (
            "INIT_FROM",
            "DATA_TAG",
            "HUMAN_PROMPT_STRIDE",
            "HUMAN_PROMPT_MAX_FRAMES",
            "VIDEO_MAX_PIXELS",
            "STANDALONE_ROBOT_QA_ENABLED",
            "CURRENT_STATE_MASK_PROB",
        ):
            env.pop(key, None)
        lines = subprocess.check_output(["bash", WRAPPER], env=env, text=True).splitlines()

        rejected = subprocess.run(
            ["bash", WRAPPER],
            env=dict(env, INIT_FROM="/tmp/forbidden"),
            text=True,
            capture_output=True,
            check=False,
        )

    assert "--nproc_per_node=4" in lines
    assert _value(lines, "--per_device_train_batch_size") == "1"
    assert _value(lines, "--gradient_accumulation_steps") == "16"
    assert _value(lines, "--human_prompt_stride") == "3"
    assert _value(lines, "--human_prompt_max_frames") == "20"
    assert _value(lines, "--video_max_pixels") == "4600000"
    assert _value(lines, "--standalone_robot_qa_enabled") == "True"
    assert _value(lines, "--robot_qa_stride") == "10"
    assert _value(lines, "--robot_qa_max_frames") == "12"
    assert _value(lines, "--image_history") == "False"
    assert _value(lines, "--state_history") == "False"
    assert _value(lines, "--current_state_mask_prob") == "0.0"
    assert _value(lines, "--expert_attends_subtask") == "True"
    assert _value(lines, "--lm_loss_per_sample") == "True"
    assert "--init_from" not in lines
    assert "0901ball" in _value(lines, "--robot_data_dirs")
    assert "0901green" in _value(lines, "--unprompted_pick_only_dirs")
    assert "0901grey" in _value(lines, "--unprompted_pick_only_dirs")
    assert "0901picksqa_dense20s3_fresh_4h100" in _value(lines, "--output_dir")
    assert rejected.returncode != 0
    assert "refuses nonempty INIT_FROM" in rejected.stderr
    assert 4 * 1 * 16 == 64

    print("PASS: fresh dense20 0901 pickup-QA four-H100 shell contract")


if __name__ == "__main__":
    main()
