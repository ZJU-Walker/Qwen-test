"""
Diagnostic: does the Qwen3-VL processor keep the exact frames the dataset hands it?

Why this exists
---------------
`PrototypeRobotDataset` pre-extracts a fixed number of frames (self.num_frames) and
passes them to the processor as a list of PIL images. The processor, however, will
*re-sample* that list using its own fps logic unless told not to -- and with no video
metadata it guesses fps=24 and silently collapses e.g. 10 frames -> ~4.

This script reconstructs the minimal slice of the training pipeline (real processor +
the exact message structure the dataset builds) and checks the *realized* frame count
against what we asked for. It is the "form a numeric expectation, then verify it"
habit turned into a runnable assertion.

Run:
    conda activate qwen3vl
    python scripts/check_video_frame_count.py
"""

import os
import numpy as np
from PIL import Image
from transformers import AutoProcessor

MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-VL-4B-Instruct")
N_FRAMES = 10          # mirror PrototypeRobotDataset.num_frames
QUESTION = "which colored block did the human hand point to?"


def effective_frame_count(video_grid_thw, temporal_patch_size):
    """video_grid_thw is [[t, h, w]]; each temporal step covers temporal_patch_size frames."""
    t = int(video_grid_thw[0][0])
    return t * temporal_patch_size


def build_and_process(processor, frames, **kwargs):
    """Mimic data_processor._build_messages -> apply_chat_template for one video sample."""
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": QUESTION},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": "green block"}]},
    ]
    out = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt", **kwargs
    )
    return out


def main():
    print(f"Loading processor: {MODEL}")
    processor = AutoProcessor.from_pretrained(MODEL)
    tps = processor.video_processor.temporal_patch_size

    # Synthetic frames are fine -- we only care about how many survive, not their content.
    frames = [
        Image.fromarray(np.full((224, 224, 3), i * 20, dtype=np.uint8))
        for i in range(N_FRAMES)
    ]

    # A) default behavior (the bug): processor re-samples
    default_out = build_and_process(processor, frames)
    default_frames = effective_frame_count(
        default_out["video_grid_thw"].tolist(), tps
    )

    # B) the fix: do_sample_frames=False keeps exactly what we extracted
    fixed_out = build_and_process(processor, frames, do_sample_frames=False)
    fixed_frames = effective_frame_count(fixed_out["video_grid_thw"].tolist(), tps)

    print(f"\nExtracted frames (asked for):        {N_FRAMES}")
    print(f"Default (sampling on)  -> realized:  {default_frames}"
          f"   ({default_out['input_ids'].shape[1]} tokens)")
    print(f"do_sample_frames=False -> realized:  {fixed_frames}"
          f"   ({fixed_out['input_ids'].shape[1]} tokens)")

    if default_frames != N_FRAMES:
        print(f"\n[!] Default path drops frames: {N_FRAMES} -> {default_frames}. "
              f"This is the silent resampling bug.")
    assert fixed_frames == N_FRAMES, (
        f"FAIL: even with do_sample_frames=False the model sees {fixed_frames} "
        f"frames, expected {N_FRAMES}. Investigate max_frames/min_frames clipping."
    )
    print("\nPASS: with do_sample_frames=False the model sees all "
          f"{N_FRAMES} extracted frames.")


if __name__ == "__main__":
    main()
