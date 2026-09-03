"""Focused CPU smoke test for unprompted standalone-pick action + robot-QA records.

Run from qwen-vl-finetune with the qwen3vl environment::

    python tests/smoke_test_standalone_pick_training.py

No model/tokenizer or real video decode is needed; the fake processor preserves the
assistant-boundary token contract used by RobotFlowMatchingDataset.
"""

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qwenvl.data.robot_data import (
    IGNORE_INDEX,
    RobotFlowMatchingDataset,
    _parse_exact_root_allowlist,
)


class _Processor:
    class _VideoProcessor:
        temporal_patch_size = 2
        fps = 2

    video_processor = _VideoProcessor()

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        # Exact Qwen assistant header/end ids used by the loader's label masking.
        ids = (
            [10, 151644, 77091, 198, 50, 151645, 198]
            if len(messages) == 2
            else [10, 151644, 77091, 198]
        )
        return {"input_ids": torch.tensor([ids])}


def _append_fake_fast(self, input_ids, labels, _norm_actions):
    postfix = torch.tensor([[901, 902, 903]])
    input_ids = torch.cat([input_ids, postfix], dim=1)
    labels = torch.cat(
        [labels, torch.tensor([[IGNORE_INDEX, 902, 903]])], dim=1
    )
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    mask[:, -3:] = True
    return input_ids, labels, mask


def _fake_rope(_merge, input_ids, **_kwargs):
    return torch.zeros((3, 1, input_ids.shape[1]), dtype=torch.long), None


def _dataset(task="pick up the ball"):
    segment = [{"task": task, "start": 0, "end": 2}]
    base = {
        "states": np.zeros((3, 14), dtype=np.float32),
        "actions": np.zeros((3, 14), dtype=np.float32),
        "video_path": "/ball/episode_000000.mp4",
        "wrist_paths": [],
        "subtasks": segment,
        "instruction": None,
        "min_start": 0,
        "hist_min": 0,
        "source_fps": 30.0,
        "source_root": "/ball",
    }
    ds = object.__new__(RobotFlowMatchingDataset)
    ds.episodes = [
        {**base, "sample_mode": "standalone_action"},
        {**base, "sample_mode": "standalone_robot_qa"},
    ]
    ds._subtask_task = "sort"
    ds._qa_mix = [("phase", 1.0)]
    ds._current_state_mask_prob = 0.0
    ds.horizon = 2
    ds.num_frames = 2
    ds.stride = 1
    ds.delta_mask = None
    ds.fps = 30.0
    ds.use_fast = True
    ds.merge_size = 2
    ds.get_rope_index = _fake_rope
    ds._dump_left = 0
    ds.data_args = SimpleNamespace(
        state_history=False,
        image_aug=False,
        image_aug_prob=1.0,
        predict_subtask=True,
        subtask_question="What should be done now?",
        image_history=True,
        history_max_pixels=10_000,
        max_pixels=10_000,
        explicit_video_timestamps=True,
        human_prompt_source_fps=30.0,
        qa_where_absent_prob=0.0,
        robot_qa_stride=10,
        robot_qa_max_frames=12,
    )
    ds.norm_stats = {
        "state": {"q01": [-1] * 14, "q99": [1] * 14},
        "actions": {"q01": [-1] * 14, "q99": [1] * 14},
    }
    ds.processor = _Processor()
    ds._append_fast_tokens = MethodType(_append_fake_fast, ds)
    image = Image.new("RGB", (8, 8))
    ds._extract_frames = MethodType(lambda _self, _episode, _t: [image, image], ds)
    ds._extract_single_frame = MethodType(
        lambda _self, _path, _t, _budget: image, ds
    )
    ds._extract_wrist_images = MethodType(lambda _self, _episode, _t: [], ds)
    ds._extract_robot_qa_video = MethodType(
        lambda _self, _episode: ([image, image], [0, 2], 3), ds
    )
    # Proves standalone records do not accidentally draw from global prompt pools.
    ds.human_prompt_pools = {"green to left": [("must-not-draw", 0, 1)]}
    return ds


def test_exact_root_allowlist():
    with TemporaryDirectory() as temp:
        root = Path(temp) / "ball"
        other = Path(temp) / "other"
        root.mkdir()
        other.mkdir()
        assert _parse_exact_root_allowlist(str(root), str(root)) == {
            str(root.resolve())
        }
        try:
            _parse_exact_root_allowlist(str(other), str(root))
        except ValueError as exc:
            assert "exact subset" in str(exc)
        else:
            raise AssertionError("undeclared standalone root was accepted")


def test_action_oracle_role_and_loss_masks():
    ds = _dataset()
    item = ds._build_item(0)
    messages, kwargs = ds.processor.calls[-1]
    user_text = " ".join(
        entry.get("text", "") for entry in messages[0]["content"]
    )
    assert len(messages) == 2
    assert messages[1]["content"][0]["text"] == "pick ball"
    assert "What should be done now?" in user_text
    assert "pick ball" not in user_text
    assert "Human demonstration:" not in user_text
    assert item["action_loss_mask"].item() == 1.0
    assert item["fast_token_mask"].sum().item() == 3
    assert item["subtask_token_mask"].sum().item() > 0
    # Assistant/oracle remains in input_ids but has zero CE; only FAST payload/end
    # are language-supervised.
    assert (item["labels"][~item["fast_token_mask"]] != IGNORE_INDEX).sum() == 0
    assert (item["labels"][item["fast_token_mask"]] != IGNORE_INDEX).sum() == 2
    assert len(kwargs["video_metadata"]) == 1  # robot history only, no human demo


def test_current_state_mask_hides_only_model_visible_values():
    ds = _dataset()
    ds._current_state_mask_prob = 0.5

    with patch("random.random", return_value=0.49):
        ds._build_item(0)
    masked_messages, _ = ds.processor.calls[-1]
    masked_text = " ".join(
        entry.get("text", "") for entry in masked_messages[0]["content"]
    )
    assert "State: [MASKED]" in masked_text
    assert "State: 127" not in masked_text

    with patch("random.random", return_value=0.50):
        ds._build_item(0)
    visible_messages, _ = ds.processor.calls[-1]
    visible_text = " ".join(
        entry.get("text", "") for entry in visible_messages[0]["content"]
    )
    assert "State: [MASKED]" not in visible_text
    assert "State: 127" in visible_text


def test_robot_qa_media_and_loss_masks():
    ds = _dataset()
    expected = {
        0: ("available for the robot to pick up", "ball", False),
        1: ("what should the robot do next", "pick ball", False),
        2: ("did the robot pick up in this demonstration", "ball", True),
        3: ("What pickup skill did the robot demonstrate", "pick ball", True),
    }
    for choice, (question, answer, is_video) in expected.items():
        with patch("random.randrange", return_value=choice):
            item = ds._build_item(1)
        messages, kwargs = ds.processor.calls[-1]
        user_text = " ".join(
            entry.get("text", "") for entry in messages[0]["content"]
        )
        assert "Robot demonstration:" in user_text
        assert "Human demonstration:" not in user_text
        assert question.lower() in user_text.lower()
        assert messages[1]["content"][0]["text"] == answer
        assert item["action_loss_mask"].item() == 0.0
        assert item["fast_token_mask"].sum().item() == 0
        assert (item["labels"] != IGNORE_INDEX).sum() > 0
        assert ("video_metadata" in kwargs) is is_video


def test_robot_qa_uses_full_green_and_grey_object_names():
    for task, full_name, answer in (
        ("pick up the green block", "green block", "pick green"),
        ("pick up the grey box", "grey box", "pick grey"),
    ):
        ds = _dataset(task)
        with patch("random.randrange", return_value=1):
            item = ds._build_item(1)
        messages, _kwargs = ds.processor.calls[-1]
        user_text = " ".join(
            entry.get("text", "") for entry in messages[0]["content"]
        )
        assert f"pick up the {full_name}" in user_text
        assert messages[1]["content"][0]["text"] == answer
        assert item["action_loss_mask"].item() == 0.0
        assert item["fast_token_mask"].sum().item() == 0


def test_robot_qa_history_gate_is_invariant_to_dense_prompt_budget():
    ds = _dataset()
    ds.data_args.image_aug = True
    ds.data_args.history_max_pixels = 65_536

    # Deliberately make augmentation change the geometry. The existing per-frame
    # history gate runs afterward and already isolates robot-QA from the global
    # whole-video budget used by dense human prompts.
    decoded = Image.new("RGB", (640, 480))
    ds._extract_robot_qa_video = MethodType(
        lambda _self, _episode: (
            [decoded.copy() for _ in range(12)], list(range(12)), 12
        ),
        ds,
    )
    augmentation_calls = []

    def augment(_self, frames, geometric):
        augmentation_calls.append((len(frames), geometric, frames[0].size))
        return [Image.new("RGB", (960, 540)) for _ in frames]

    ds._augment = MethodType(augment, ds)
    with patch("random.randrange", return_value=2):
        ds._build_item(1)

    messages, _kwargs = ds.processor.calls[-1]
    video_entry = next(
        entry for entry in messages[0]["content"] if entry.get("type") == "video"
    )
    gated_frames = video_entry["video"]
    assert augmentation_calls == [(12, True, (640, 480))]
    assert {frame.size for frame in gated_frames} == {(341, 192)}

    # Qwen factor-rounds the 341x192 gate output to 352x192. Both the legacy
    # 1.6M budget and dense-prompt 4.6M budget keep that exact grid: 396 tokens
    # for 12 frames (T/2 * H/32 * W/32 = 6 * 6 * 11).
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize

    grids = {
        smart_resize(
            num_frames=12,
            height=gated_frames[0].height,
            width=gated_frames[0].width,
            temporal_factor=2,
            factor=32,
            min_pixels=200_704,
            max_pixels=budget,
        )
        for budget in (1_600_000, 4_600_000)
    }
    assert grids == {(192, 352)}
    height, width = grids.pop()
    assert (12 // 2) * (height // 32) * (width // 32) == 396


def test_special_retry_never_substitutes_another_episode():
    ds = object.__new__(RobotFlowMatchingDataset)
    ds.episodes = [
        {"sample_mode": "standalone_action"},
        {"sample_mode": "normal"},
    ]
    calls = []

    def build(_self, idx):
        calls.append(idx)
        if len(calls) < 4:
            raise RuntimeError("transient decode failure")
        return {"kept_index": idx}

    ds._build_item = MethodType(build, ds)
    assert ds[0] == {"kept_index": 0}
    assert calls == [0, 0, 0, 0]


if __name__ == "__main__":
    test_exact_root_allowlist()
    test_action_oracle_role_and_loss_masks()
    test_current_state_mask_hides_only_model_visible_values()
    test_robot_qa_media_and_loss_masks()
    test_robot_qa_uses_full_green_and_grey_object_names()
    test_robot_qa_history_gate_is_invariant_to_dense_prompt_budget()
    test_special_retry_never_substitutes_another_episode()
    print("standalone-pick action/robot-QA smoke test: OK")
