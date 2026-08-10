"""Split selected 0717/0731 episodes at the waiting->pick cut for manual checking.

For each selected episode, writes two H.264 mp4s (VS Code-previewable) into a
per-dataset subdir of the output dir (default /iris/projects/humanoid/ke/0731_label_check):
the frames BEFORE the cut (..._1_waiting_*.mp4) and the frames FROM the cut
(..._2_pick_up_<color>_*.mp4), each frame overlaid with segment name + global frame
index. Correct labels look like: part 1 = arm still (human pointing), part 2 = arm
moving, and the color in the filename matches the block actually picked.

Cut rule (both datasets, gen_0731_subtask_labels.py): after a startup guard and a real
quiet window, sustained-motion onset + 10 frames. So part 1 ("waiting") should contain
the human pointing and remain robot-still except its LAST ~10 frames, where the arm
visibly begins to move; part 2 should be fully in motion from frame one.

Selection per dataset: the cut extremes (earliest/latest pick start), a few
seeded-random episodes of each color, and for 0731 the green/yellow collection
boundary (episodes 108-111 -- 0731 colors come from the collection order, so this is
where a color split would break; 0717 colors are per-episode human labels).

Usage (CPU, ~2 min):
    cd /iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune
    python tests/gen_0731_split_check.py            # or: [--out DIR] [--extra N]
"""
import argparse
import json
import random
from pathlib import Path

import av
import cv2

DATA = Path("/iris/projects/humanoid/trossen_data")
DATASETS = {
    "0717": (DATA / "0717_green_yellow_block_mem_merged", False),
    "0731": (DATA / "0731_green_yellow_merged", True),      # True = check the color boundary
}
CAM = "observation.images.cam_high"


def pick_episodes(labels, extra_per_color, color_boundary):
    picks = {k: v for k, v in labels.items()
             if len(v) > 1 and v[0]["task"] == "waiting"}   # skip degenerate/mislabeled
    cuts = {k: v[1]["start"] for k, v in picks.items()}
    sel = {min(cuts, key=cuts.get), max(cuts, key=cuts.get)}          # cut extremes
    if color_boundary:
        sel |= {f"episode_{i:06d}.mp4" for i in (108, 109, 110, 111)}
    rng = random.Random(0)
    for color in ("green", "yellow"):
        pool = sorted(k for k, v in picks.items() if color in v[1]["task"] and k not in sel)
        sel |= set(rng.sample(pool, min(extra_per_color, len(pool))))
    return sorted(sel & picks.keys())


def split_episode(root, key, segs, out_dir):
    video = root / "videos" / "chunk-000" / CAM / key
    # AV1-encoded: cv2.VideoCapture cannot decode these; read via torchvision/PyAV
    # (software dav1d decoder -- the same path the training dataloader uses).
    from torchvision.io import read_video
    vid, _, info = read_video(str(video), pts_unit="sec", output_format="THWC")
    assert len(vid), f"no frames decoded from {video}"
    fps = float(info.get("video_fps") or 30.0)
    frames = [f.numpy().copy() for f in vid]               # RGB throughout

    cut = segs[1]["start"]
    task = segs[1]["task"].replace(" ", "_")
    stem = key.replace(".mp4", "")
    parts = [(f"{stem}_1_waiting_f0-{cut - 1}.mp4", 0, cut),
             (f"{stem}_2_{task}_f{cut}-{len(frames) - 1}.mp4", cut, len(frames))]
    h, w = frames[0].shape[:2]
    for name, lo, hi in parts:
        # H.264-in-mp4 via PyAV (bundles libx264) so VS Code / browsers can preview;
        # cv2's VideoWriter has no H.264 encoder in the pip build.
        container = av.open(str(out_dir / name), "w")
        stream = container.add_stream("h264", rate=round(fps))
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.options = {"crf": "23"}
        label = "waiting" if "_1_" in name else task
        for g in range(lo, hi):
            f = frames[g].copy()
            cv2.putText(f, f"{label}  frame {g}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)   # red (RGB)
            for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    print(f"  {key}: cut {cut} ({cut / fps:.1f}s), {segs[1]['task']} -> "
          f"{parts[0][0]} + {parts[1][0]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/iris/projects/humanoid/ke/0731_label_check")
    ap.add_argument("--extra", type=int, default=3, help="random episodes per color")
    args = ap.parse_args()

    for name, (root, color_boundary) in DATASETS.items():
        out_dir = Path(args.out) / name
        out_dir.mkdir(parents=True, exist_ok=True)
        labels = json.load(open(root / "videos" / "chunk-000" / "subtask_labels.json"))
        print(f"{name} ({root.name}):")
        for key in pick_episodes(labels, args.extra, color_boundary):
            split_episode(root, key, labels[key], out_dir)
    print(f"\nwrote splits to {args.out}/{{0717,0731}} -- check: part 1 still except "
          f"its last ~10 frames (the deliberate onset+10 offset), part 2 fully moving "
          f"from frame one, filename color = block actually picked")


if __name__ == "__main__":
    main()
