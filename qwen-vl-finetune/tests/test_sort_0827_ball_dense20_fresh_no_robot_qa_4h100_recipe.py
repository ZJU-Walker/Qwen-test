#!/usr/bin/env python3
"""CPU/shell contract gate for fresh dense20 ball-action training on 4 H100s."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


FINETUNE = Path(__file__).resolve().parents[1]
SCRIPTS = FINETUNE / "scripts"
BASE = SCRIPTS / "train_action_expert_4b_humanprompt.sh"
BALL = SCRIPTS / "train_action_expert_4b_sort_0827_ball.sh"
RECIPE = (
    SCRIPTS
    / "train_action_expert_4b_sort_0827_ball_dense20_fresh_no_robot_qa_4h100.sh"
)
FRESH_ROOT = Path("/iris/u/kewalk/qwen_checkpoints")
BALL_ROOT = Path(
    "/iris/projects/humanoid/trossen_data/0827_prompting_playdata/"
    "data_0827_prompting_playdata/ball"
)
FRESH_TAG = "0824sort_0827ballaction_dense20s3_fresh_norobotqa_4h100"


def _value(argv: list[str], flag: str) -> str:
    index = argv.index(flag)
    return argv[index + 1]


def _clean_env(capture: Path) -> dict[str, str]:
    env = dict(
        os.environ,
        TORCHRUN=str(capture),
        NUM_WORKERS="0",
        WANDB_DIR=str(capture.parent / "wandb_fresh_norobotqa_contract"),
        RUN_TAG="cpu_contract",
    )
    for inherited in (
        "DATA_TAG",
        "INIT_FROM",
        "CHECKPOINT_ROOT",
        "HUMAN_PROMPT_STRIDE",
        "HUMAN_PROMPT_MAX_FRAMES",
        "EXPLICIT_VIDEO_TIMESTAMPS",
        "VIDEO_MAX_PIXELS",
        "HISTORY_MAX_PIXELS",
        "STATE_HISTORY",
        "IMAGE_HISTORY",
        "STANDALONE_ROBOT_QA_ENABLED",
        "NPROC_PER_NODE",
        "PER_DEVICE_BATCH",
        "TORCH_COMPILE",
        "SAVE_TOTAL_LIMIT",
    ):
        env.pop(inherited, None)
    return env


def main() -> None:
    for script in (BASE, BALL, RECIPE):
        assert script.is_file(), script
        subprocess.run(["bash", "-n", str(script)], check=True)

    recipe_text = RECIPE.read_text()
    base_text = BASE.read_text()
    assert f"export DATA_TAG={FRESH_TAG}" in recipe_text
    assert 'if [ -n "${INIT_FROM-}" ]' in recipe_text
    assert "export INIT_FROM=" in recipe_text
    assert "export STANDALONE_ROBOT_QA_ENABLED=False" in recipe_text
    assert "train_action_expert_4b_sort_0827_ball_dense20.sh" not in recipe_text
    assert "exec bash train_action_expert_4b_sort_0827_ball.sh" in recipe_text
    assert 'STANDALONE_ROBOT_QA_ENABLED="${STANDALONE_ROBOT_QA_ENABLED:-True}"' in base_text
    assert '--standalone_robot_qa_enabled "$STANDALONE_ROBOT_QA_ENABLED"' in base_text

    # Expand the entire real shell stack through a fake torchrun. This checks the
    # actual Trainer CLI without importing Transformers or touching a GPU.
    with tempfile.TemporaryDirectory() as tmp:
        capture = Path(tmp) / "torchrun"
        capture.write_text("#!/bin/bash\nprintf '%s\\n' \"$@\"\n")
        capture.chmod(0o755)
        env = _clean_env(capture)
        argv = subprocess.check_output(
            ["bash", str(RECIPE)], env=env, text=True
        ).splitlines()

        inherited_init = _clean_env(capture)
        inherited_init["INIT_FROM"] = "/tmp/forbidden-prior-policy"
        rejected_init = subprocess.run(
            ["bash", str(RECIPE)],
            env=inherited_init,
            text=True,
            capture_output=True,
            check=False,
        )

        wrong_geometry = _clean_env(capture)
        wrong_geometry["NPROC_PER_NODE"] = "2"
        rejected_geometry = subprocess.run(
            ["bash", str(RECIPE)],
            env=wrong_geometry,
            text=True,
            capture_output=True,
            check=False,
        )

    assert "--nproc_per_node=4" in argv
    assert _value(argv, "--model_name_or_path") == "Qwen/Qwen3-VL-4B-Instruct"
    assert _value(argv, "--per_device_train_batch_size") == "1"
    assert _value(argv, "--gradient_accumulation_steps") == "16"
    assert 4 * 1 * 16 == 64

    assert _value(argv, "--human_prompt_stride") == "3"
    assert _value(argv, "--human_prompt_max_frames") == "20"
    assert _value(argv, "--human_prompt_source_fps") == "30"
    assert _value(argv, "--explicit_video_timestamps") == "True"
    assert _value(argv, "--video_max_pixels") == "4600000"
    assert _value(argv, "--history_max_pixels") == "65536"
    assert _value(argv, "--image_history") == "False"
    assert _value(argv, "--state_history") == "False"
    assert _value(argv, "--torch_compile") == "False"

    # The exact ball root remains opted into standalone action records; only its
    # paired language-only robot-video QA records are disabled.
    assert Path(_value(argv, "--unprompted_pick_only_dirs")) == BALL_ROOT
    assert _value(argv, "--standalone_robot_qa_enabled") == "False"

    assert "--init_from" not in argv
    output = Path(_value(argv, "--output_dir"))
    run_name = _value(argv, "--run_name")
    assert output.parent == FRESH_ROOT
    assert FRESH_TAG in output.name and "cpu_contract" in output.name
    assert FRESH_TAG in run_name and "cpu_contract" in run_name
    assert _value(argv, "--save_total_limit") == "2"

    assert rejected_init.returncode != 0
    assert "refuses nonempty INIT_FROM" in rejected_init.stderr
    assert rejected_geometry.returncode != 0
    assert "requires exactly 4 GPUs" in rejected_geometry.stderr

    print("PASS: fresh dense20 no-robot-QA four-H100 shell contract")
    print("  initialization: Qwen base, no --init_from, unique fresh output")
    print("  data: 0824 prompted actions + 0827 ball actions; robot-video QA disabled")
    print("  prompt: stride 3, <=20 frames, 4.6M pixels; robot history off")
    print("  optimizer: 4 H100s x microbatch 1 x accumulation 16 = global batch 64")


if __name__ == "__main__":
    main()
