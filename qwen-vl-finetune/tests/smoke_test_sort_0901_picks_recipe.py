#!/usr/bin/env python3
"""Processor-free integrity gate for the 0901 standalone-pick QA extension."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwenvl.data.subtask_formats_sort import (  # noqa: E402
    is_standalone_pick,
    standalone_pick_answer,
    standalone_pick_qa_specs,
)


DATA_ROOT = Path(os.environ.get("SORT_0901_ROOT", "/iris/projects/humanoid/trossen_data"))
SPECS = {
    "0901ball": (30, {"episode_000008.mp4", "episode_000011.mp4"}, "ball"),
    "0901green": (30, set(), "green"),
    "0901grey": (31, set(), "grey"),
}
EXPECTED_NEW_CLEAN = 89
EXPECTED_NORMAL = 371
EXPECTED_OLD_STANDALONE = 46
EXPECTED_STANDALONE = EXPECTED_OLD_STANDALONE + EXPECTED_NEW_CLEAN
EXPECTED_ACTION_EPISODES = EXPECTED_NORMAL + EXPECTED_STANDALONE
EXPECTED_RECORDS = EXPECTED_NORMAL + 2 * EXPECTED_STANDALONE


def _labels(root: Path) -> dict[str, list[dict]]:
    return json.loads((root / "videos/chunk-000/subtask_labels.json").read_text())


def _source_recipe() -> tuple[list[str], list[str], str, str]:
    script = ROOT / "scripts" / "sort_0901_picks_data.sh"
    command = f"""
set -euo pipefail
source {script}
printf '%s\n' "$SORT_0901_PICKS_ROBOT_DIRS"
printf '%s\n' "$SORT_0901_PICKS_UNPROMPTED_DIRS"
printf '%s\n' "$SORT_0901_PICKS_HUMAN_DIRS"
printf '%s\n' "$SORT_0901_PICKS_FAST_DEFAULT"
"""
    env = dict(os.environ, SORT_0901_ROOT=str(DATA_ROOT))
    lines = subprocess.check_output(
        ["bash", "-c", command], cwd=ROOT, env=env, text=True
    ).splitlines()
    return lines[0].split(","), lines[1].split(","), lines[2], lines[3]


def main() -> None:
    new_clean = 0
    for name, (expected_total, expected_excluded, expected_object) in SPECS.items():
        root = DATA_ROOT / name
        labels = _labels(root)
        assert len(labels) == expected_total, (name, len(labels))
        exclusion_path = root / "meta" / "train_exclude_episodes.json"
        excluded = set()
        if exclusion_path.exists():
            payload = json.loads(exclusion_path.read_text())
            assert set(payload) == {"version", "reason", "episodes"}, payload
            assert payload["version"] == 1
            assert isinstance(payload["reason"], str) and payload["reason"].strip()
            excluded = set(payload["episodes"])
        assert excluded == expected_excluded, (name, excluded)

        for episode, segments in labels.items():
            tasks = [segment["task"] for segment in segments]
            assert is_standalone_pick(tasks), (name, episode, tasks)
            assert standalone_pick_answer("object", tasks) == expected_object
            assert standalone_pick_answer("phase", tasks) == f"pick {expected_object}"
            qa_specs = standalone_pick_qa_specs(tasks)
            assert len(qa_specs) == 4
            assert all(
                word not in " ".join(question for _, question, _ in qa_specs).lower()
                for word in ("left", "middle", "right", "tray")
            )
            assert segments[0]["start"] == 0
            parquet = root / "data/chunk-000" / episode.replace(".mp4", ".parquet")
            assert parquet.is_file(), parquet
            for camera in ("cam_high", "cam_left_wrist", "cam_right_wrist"):
                video = (
                    root / "videos/chunk-000" / f"observation.images.{camera}" / episode
                )
                assert video.is_file(), video
        new_clean += len(labels) - len(excluded)

    robot_dirs, standalone_dirs, human_dirs, fast_path = _source_recipe()
    expected_new_roots = {str(DATA_ROOT / name) for name in SPECS}
    assert expected_new_roots <= set(robot_dirs)
    assert expected_new_roots <= set(standalone_dirs)
    assert len(robot_dirs) == 29, len(robot_dirs)
    assert len(standalone_dirs) == 4, standalone_dirs
    assert human_dirs.endswith("/0824_prompting/human_demo")
    assert fast_path.endswith("fast_tokenizer_trossen_0824sort_0827ball_0901picks_ee6d")
    assert new_clean == EXPECTED_NEW_CLEAN
    assert EXPECTED_ACTION_EPISODES == 506
    assert EXPECTED_RECORDS == 641

    print("PASS: 0901 standalone-pick QA recipe integrity")
    print(f"  new clean standalone picks: {new_clean} (2 failed ball clips excluded)")
    print(f"  complete prompted records: {EXPECTED_NORMAL}")
    print(f"  standalone action records: {EXPECTED_STANDALONE}")
    print(f"  standalone robot-QA records: {EXPECTED_STANDALONE}")
    print(f"  total action-bearing episodes: {EXPECTED_ACTION_EPISODES}")
    print(f"  total sampling records: {EXPECTED_RECORDS}")


if __name__ == "__main__":
    main()
