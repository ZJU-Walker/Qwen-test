#!/usr/bin/env python3
"""Real loader-scan gate for dense20 + 0901 standalone pickup QA.

Item media/mask construction is covered by ``smoke_test_standalone_pick_training.py``
with a fake processor. Keeping this gate processor-free lets it validate every real
parquet/label/prompt-pool relationship without loading a multi-gigabyte Qwen tokenizer
on a constrained login node.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT.parent / "qwen_cache" / "huggingface"))

from qwenvl.data.robot_data import (  # noqa: E402
    RobotDataArguments,
    RobotFlowMatchingDataset,
    _parse_exact_root_allowlist,
    parse_active_dims,
)


DATA = Path(os.environ.get("SORT_0901_ROOT", "/iris/projects/humanoid/trossen_data"))
ROOT_0824 = Path(os.environ.get("SORT_0824_ROOT", str(DATA / "0824_prompting")))
ROOT_0827 = Path(
    os.environ.get(
        "SORT_0827_ROOT",
        str(DATA / "0827_prompting_playdata/data_0827_prompting_playdata"),
    )
)
ROOTS_0901 = [DATA / name for name in ("0901ball", "0901green", "0901grey")]
ROBOT_0824 = (
    "111", "112", "113", "121", "122", "123", "131", "132", "133",
    "311", "312", "313", "321", "322", "323", "331", "332", "333",
    "cup", "green_block", "grey_pepper_box", "tape",
)
ROBOT_ROOTS = [*(ROOT_0824 / name for name in ROBOT_0824)] + [
    ROOT_0827 / name for name in ("green", "grey", "tape", "ball")
] + ROOTS_0901
STANDALONE_ROOTS = [ROOT_0827 / "ball", *ROOTS_0901]
FAST = Path(
    os.environ.get(
        "SORT_0901_PICKS_FAST_DEFAULT",
        str(ROOT.parent / "checkpoints/fast_tokenizer_trossen_0824sort_0827ball_0901picks_ee6d"),
    )
)


def _args() -> RobotDataArguments:
    da = RobotDataArguments()
    da.robot_data_dirs = ",".join(map(str, ROBOT_ROOTS))
    da.camera = "cam_high"
    da.image_history = False
    da.state_history = False
    da.current_state_mask_prob = 0.0
    da.active_dims = "7:14"
    da.action_space = "ee6d"
    da.action_dim = 10
    da.delta_mask = "9,-1"
    da.action_horizon = 50
    da.use_delta_actions = True
    da.wrist_cameras = "cam_right_wrist,cam_left_wrist"
    da.wrist_max_pixels = 131072
    da.max_pixels = 131072
    da.history_max_pixels = 65536
    da.train_split = 1.0
    da.min_episode_len = 50
    da.skip_leading_subtask = ""
    da.image_aug = False
    da.predict_subtask = True
    da.subtask_task = "sort"
    da.subtask_question = "What should be done now?"
    da.subtask_format_mix = (
        "phase:0.40,object:0.10,target:0.10,where:0.15,demo:0.15,remaining:0.10"
    )
    da.qa_where_absent_prob = 0.2
    da.use_fast_tokens = True
    da.fast_tokenizer_path = str(FAST)
    da.human_prompt_dirs = str(ROOT_0824 / "human_demo")
    da.human_prompt_segments = True
    da.human_prompt_full_episode_prob = 0.5
    da.human_prompt_stride = 3
    da.human_prompt_max_frames = 20
    da.human_prompt_source_fps = 30.0
    da.human_prompt_holdout = 1
    da.human_prompt_holdout_per_key = True
    da.explicit_video_timestamps = True
    da.video_max_pixels = 4_600_000
    da.order_sample_prob = 0.0
    da.unprompted_pick_only_dirs = ",".join(map(str, STANDALONE_ROOTS))
    da.standalone_robot_qa_enabled = True
    da.robot_qa_stride = 10
    da.robot_qa_max_frames = 12
    return da


def main() -> None:
    assert (FAST / "norm_stats.json").is_file(), FAST
    da = _args()
    ds = object.__new__(RobotFlowMatchingDataset)
    ds.data_args = da
    ds._subtask_task = "sort"
    ds._unprompted_pick_only_roots = _parse_exact_root_allowlist(
        da.unprompted_pick_only_dirs, da.robot_data_dirs
    )
    ds._standalone_robot_qa_enabled = True
    ds.active_dims = parse_active_dims(da.active_dims)
    ds.wrist_cameras = [name.strip() for name in da.wrist_cameras.split(",") if name.strip()]
    ds._qa_mix = ds._fmt.parse_format_mix(da.subtask_format_mix)
    ds.episodes = ds._scan_episodes(da)
    ds.human_prompt_full_pools = None
    ds.human_prompt_pools = ds._scan_human_prompts(da)

    counts = Counter(ep["sample_mode"] for ep in ds.episodes)
    assert counts == {
        "normal": 371,
        "standalone_action": 135,
        "standalone_robot_qa": 135,
    }, counts
    assert len(ds.episodes) == 641
    assert len({ep["video_path"] for ep in ds.episodes}) == 506
    assert not any(
        ep["source_root"].endswith("/0901ball")
        and Path(ep["video_path"]).name in {"episode_000008.mp4", "episode_000011.mp4"}
        for ep in ds.episodes
    )

    expected = {
        "/0901ball": (28, "ball", "ball"),
        "/0901green": (30, "green", "green block"),
        "/0901grey": (31, "grey", "grey box"),
    }
    for suffix, (physical, symbol, full_name) in expected.items():
        for mode in ("standalone_action", "standalone_robot_qa"):
            selected = [
                ep for ep in ds.episodes
                if ep["sample_mode"] == mode and ep["source_root"].endswith(suffix)
            ]
            assert len(selected) == physical, (suffix, mode, len(selected))
            for episode in selected:
                tasks = [segment["task"] for segment in episode["subtasks"]]
                specs = ds._fmt.standalone_pick_qa_specs(tasks)
                assert [answer for _, _, answer in specs] == [
                    symbol, f"pick {symbol}", symbol, f"pick {symbol}"
                ]
                assert f"pick up the {full_name}" in " ".join(
                    question for _, question, _ in specs
                )

    print("PASS: real dense20 0901 pickup loader scan + prompt-pool validation")
    print(f"  records: {dict(counts)}; unique action episodes: 506")


if __name__ == "__main__":
    main()
