"""Real-data gate for the standalone-ball action-only (no robot-QA) ablation.

This scans parquets/videos and validates human-prompt pool compatibility, but does not
load a processor, FAST tokenizer, or model::

    /iris/projects/humanoid/miniconda3/envs/qwen3vl/bin/python \
        tests/smoke_test_standalone_pick_action_only.py
"""

import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qwenvl.data.robot_data import (  # noqa: E402
    RobotDataArguments,
    RobotFlowMatchingDataset,
    _episodes_for_norm_stats,
    _parse_exact_root_allowlist,
)


ROOT_0824 = Path(os.environ.get(
    "SORT_0824_ROOT", "/iris/projects/humanoid/trossen_data/0824_prompting"
))
ROOT_0827 = Path(os.environ.get(
    "SORT_0827_ROOT",
    "/iris/projects/humanoid/trossen_data/0827_prompting_playdata/"
    "data_0827_prompting_playdata",
))
ROBOT_0824 = (
    "111", "112", "113", "121", "122", "123", "131", "132", "133",
    "311", "312", "313", "321", "322", "323", "331", "332", "333",
    "cup", "green_block", "grey_pepper_box", "tape",
)
ROBOT_DIRS = [*(ROOT_0824 / name for name in ROBOT_0824)] + [
    ROOT_0827 / name for name in ("green", "grey", "tape", "ball")
]
BALL_ROOT = ROOT_0827 / "ball"


def _scan_args():
    return SimpleNamespace(
        robot_data_dirs=",".join(map(str, ROBOT_DIRS)),
        camera="cam_high",
        robot_subtask_labels_file="subtask_labels.json",
        skip_leading_subtask="",
        min_episode_len=50,
        action_space="joint",
        train_split=1.0,
        human_prompt_dirs=str(ROOT_0824 / "human_demo"),
        human_prompt_source_fps=30.0,
        human_prompt_holdout=1,
        human_prompt_holdout_per_key=True,
        order_sample_prob=0.0,
        human_prompt_full_episode_prob=0.5,
    )


def main():
    # Backward compatibility is explicit: existing recipes that omit the flag retain QA.
    assert RobotDataArguments().standalone_robot_qa_enabled is True

    args = _scan_args()
    ds = object.__new__(RobotFlowMatchingDataset)
    ds._subtask_task = "sort"
    ds._qa_mix = [("phase", 1.0)]
    ds.active_dims = None
    ds.wrist_cameras = []
    ds._unprompted_pick_only_roots = _parse_exact_root_allowlist(
        str(BALL_ROOT), args.robot_data_dirs
    )
    ds._standalone_robot_qa_enabled = False
    ds.episodes = ds._scan_episodes(args)

    counts = Counter(episode["sample_mode"] for episode in ds.episodes)
    assert counts == {"normal": 371, "standalone_action": 46}, counts
    assert len(ds.episodes) == 417
    assert not any(
        episode["sample_mode"] == "standalone_robot_qa"
        for episode in ds.episodes
    )
    assert len({episode["video_path"] for episode in ds.episodes}) == 417

    # Every normal record still validates against a human prompt; only the exact opted-in
    # ball action records bypass destination/full-combo checks.
    pools = ds._scan_human_prompt_segments(args)
    assert pools

    # The action-only ablation and default action+QA run normalize over the same 417
    # physical/action-bearing episodes. Simulate the default duplicate and prove it is
    # removed by the exact helper used in _load_or_compute_norm_stats.
    norm_action_only = _episodes_for_norm_stats(ds.episodes)
    qa_twins = [
        {**episode, "sample_mode": "standalone_robot_qa"}
        for episode in ds.episodes
        if episode["sample_mode"] == "standalone_action"
    ]
    norm_default = _episodes_for_norm_stats([*ds.episodes, *qa_twins])
    assert len(norm_action_only) == len(norm_default) == 417
    assert [episode["video_path"] for episode in norm_action_only] == [
        episode["video_path"] for episode in norm_default
    ]

    # Checkpoint provenance must distinguish the ablation even though serving clears
    # the training-only root allowlist.
    train_source = (
        Path(__file__).parents[1] / "qwenvl" / "train" / "train_action_expert.py"
    ).read_text()
    assert '"standalone_robot_qa_enabled"' in train_source

    print("PASS: standalone ball action-only ablation")
    print(f"  records: {dict(counts)}; robot-QA=0; norm episodes={len(norm_default)}")


if __name__ == "__main__":
    main()
