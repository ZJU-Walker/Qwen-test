#!/usr/bin/env python3
"""CPU-only acceptance gate for the fixed dense-20 human-prompt contract.

This loads only the Qwen3-VL processor (never model weights), constructs the largest
human-prompt input used by the recipe, and checks both the exact visual grid and the
8192-token context limit.  Run it before requesting/occupying H200s.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PIL import Image
from transformers import AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwenvl.action_expert.human_prompt import (  # noqa: E402
    sample_prompt_indices,
    sampled_video_metadata,
)


DEFAULT_CACHE = ROOT.parent / "qwen_cache" / "huggingface"
SOURCE_FPS = 30.0
PROMPT_STRIDE = 3
PROMPT_MAX_FRAMES = 20
VIDEO_MAX_PIXELS = 4_600_000
VIDEO_MIN_PIXELS = 200_704
IMAGE_MAX_PIXELS = 131_072
IMAGE_MIN_PIXELS = 784
MODEL_MAX_LENGTH = 8192
FAST_ACTION_TOKENS = 50


def _local_processor_path() -> Path:
    override = os.environ.get("DENSE20_PROCESSOR_PATH")
    if override:
        path = Path(override)
        if not path.is_dir():
            raise FileNotFoundError(f"DENSE20_PROCESSOR_PATH is not a directory: {path}")
        return path

    # Pass a concrete snapshot directory to Transformers.  Passing the hub model id
    # with a nonstandard HF_HOME can still issue network HEAD requests even with
    # local_files_only=True, which defeats a fail-fast allocation preflight.
    cache_roots = [Path(os.environ.get("HF_HOME", DEFAULT_CACHE)), DEFAULT_CACHE]
    checked: list[Path] = []
    for cache_root in dict.fromkeys(cache_roots):
        for prefix in (cache_root, cache_root / "hub"):
            snapshots = (
                prefix / "models--Qwen--Qwen3-VL-4B-Instruct" / "snapshots"
            )
            checked.append(snapshots)
            for path in sorted(snapshots.glob("*"), reverse=True):
                if (path / "preprocessor_config.json").is_file():
                    return path
    raise FileNotFoundError(
        "Qwen3-VL processor snapshot is not cached; checked: "
        + ", ".join(str(path) for path in checked)
    )


def _set_area_budget(processor, *, video_max_pixels: int) -> None:
    """Mirror update_processor_pixels without importing the training dataset."""
    ip = processor.image_processor
    ip.min_pixels = IMAGE_MIN_PIXELS
    ip.max_pixels = IMAGE_MAX_PIXELS
    ip.size["shortest_edge"] = IMAGE_MIN_PIXELS
    ip.size["longest_edge"] = IMAGE_MAX_PIXELS

    vp = processor.video_processor
    vp.min_pixels = VIDEO_MIN_PIXELS
    vp.max_pixels = video_max_pixels
    vp.size["shortest_edge"] = VIDEO_MIN_PIXELS
    vp.size["longest_edge"] = video_max_pixels


def _dense20_messages(frames: list[Image.Image]) -> list[dict]:
    robot = Image.new("RGB", (640, 360), "black")
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Human demonstration:"},
                {"type": "video", "video": frames},
                {"type": "text", "text": "Robot view:"},
                {"type": "image", "image": robot},
                {"type": "image", "image": robot},
                {"type": "image", "image": robot},
                {
                    "type": "text",
                    "text": (
                        "Task: State the action to perform now., State: "
                        "111 65 200 215 187 175 141 148 18 205"
                    ),
                },
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "pick green"}],
        },
    ]


def check_dense20_contract() -> dict[str, int | list[int]]:
    processor_path = _local_processor_path()
    processor = AutoProcessor.from_pretrained(
        str(processor_path),
        local_files_only=True,
    )
    _set_area_budget(processor, video_max_pixels=VIDEO_MAX_PIXELS)

    # 58 source frames sampled at 0,3,...,57 is the exact 20-frame maximum case.
    source_count = 58
    source_indices = sample_prompt_indices(
        source_count, PROMPT_STRIDE, PROMPT_MAX_FRAMES
    )
    assert source_indices == list(range(0, source_count, PROMPT_STRIDE))
    assert len(source_indices) == PROMPT_MAX_FRAMES
    frames = [Image.new("RGB", (640, 360), (i, i, i)) for i in source_indices]
    metadata = sampled_video_metadata(source_count, SOURCE_FPS, source_indices)

    native_aligned_volume = PROMPT_MAX_FRAMES * 640 * 352
    assert VIDEO_MAX_PIXELS >= native_aligned_volume, (
        f"video budget {VIDEO_MAX_PIXELS} cannot preserve the 20x640x352 grid "
        f"({native_aligned_volume} pixels)"
    )

    encoded = processor.apply_chat_template(
        _dense20_messages(frames),
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=False,
        do_sample_frames=False,
        video_metadata=[metadata],
    )

    # Twenty frames form ten two-frame temporal groups.  Every group remains at the
    # source camera's Qwen-aligned 640x352 grid:
    #   grid_h = 352/16 = 22, grid_w = 640/16 = 40.
    video_grid = encoded["video_grid_thw"].tolist()
    expected_video_grid = [[PROMPT_MAX_FRAMES // 2, 22, 40]]
    assert video_grid == expected_video_grid, (
        f"dense-20 grid changed: got {video_grid}, expected {expected_video_grid}"
    )
    merge = int(processor.video_processor.merge_size)
    prompt_visual_tokens = sum(
        t * (h // merge) * (w // merge) for t, h, w in video_grid
    )
    assert prompt_visual_tokens == 2200

    # The real loader appends 50 FAST action tokens after chat templating.  Include
    # them in this acceptance count even though registering the FAST tokenizer would
    # unnecessarily pull the full robot dataset into this CPU-only visual preflight.
    chat_tokens = int(encoded["input_ids"].shape[-1])
    accepted_length = chat_tokens + FAST_ACTION_TOKENS
    assert accepted_length < MODEL_MAX_LENGTH, (
        f"dense-20 input requires {accepted_length} tokens, exceeding "
        f"model_max_length={MODEL_MAX_LENGTH}"
    )
    return {
        "sampled_frames": len(source_indices),
        "video_grid": video_grid[0],
        "prompt_visual_tokens": prompt_visual_tokens,
        "chat_tokens": chat_tokens,
        "accepted_tokens_with_fast": accepted_length,
        "context_headroom": MODEL_MAX_LENGTH - accepted_length,
    }


def test_dense20_contract() -> None:
    check_dense20_contract()


def main() -> None:
    result = check_dense20_contract()
    print("PASS: fixed dense-20 prompt contract")
    for key, value in result.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
