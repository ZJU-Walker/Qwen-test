"""Visual sanity check for pi05-style train-time augmentation.

For N random samples: the full 10-frame cam_high history strip ORIGINAL (top row) vs
AUGMENTED (bottom row) plus the wrist stills, one PNG per sample. Uses the dataset's OWN
extraction + augmentation methods, so what you see is exactly what training feeds the
processor. Frames within one sample must show the SAME crop/rotation/jitter (per-clip
sampling); different samples must differ.

Run:  python -m scripts.viz_augmentation --out <dir> [--n 8]
"""
import argparse
import os
import random
import sys

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune")
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

from pathlib import Path

from PIL import Image, ImageDraw

from qwenvl.action_expert.inference import build_dataset

DATA = "/iris/projects/humanoid/trossen_data"
MIX = f"{DATA}/0717_green_yellow_block_mem_merged,{DATA}/0731_green_yellow_merged"
TOK = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717merged_0731gy"


def strip(frames, w=200):
    h = int(frames[0].height * w / frames[0].width)
    out = Image.new("RGB", (len(frames) * (w + 2), h), "white")
    for i, f in enumerate(frames):
        out.paste(f.resize((w, h)), (i * (w + 2), 0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # wrist_cameras must be set BEFORE construction (wrist video paths resolve at scan
    # time); the budget overlay in build_data_args applies pre-scan.
    ds = build_dataset(fast_tok=TOK, data_dirs=MIX,
                       budget={"wrist_cameras": "cam_right_wrist,cam_left_wrist"})
    ds.data_args.image_aug = True
    random.seed(args.seed)

    for i in range(args.n):
        ep = ds.episodes[random.randrange(len(ds.episodes))]
        t = random.randrange(len(ep["states"]))
        frames = ds._extract_frames(ep, t)
        wrists = ds._extract_wrist_images(ep, t)
        aug_frames = ds._augment(frames, geometric=True)
        aug_wrists = [ds._augment([w], geometric=False)[0] for w in wrists]

        rows = [strip(frames), strip(aug_frames),
                strip(wrists + aug_wrists, w=300)]
        W = max(r.width for r in rows)
        H = sum(r.height + 22 for r in rows)
        sheet = Image.new("RGB", (W, H), "white")
        y = 0
        for label, r in zip(["ORIGINAL history (oldest -> newest)",
                             "AUGMENTED history (same crop/rotation/jitter on every frame)",
                             "wrists: original right, original left | augmented right, augmented left"],
                            rows):
            ImageDraw.Draw(sheet).text((4, y + 2), label, fill="black")
            sheet.paste(r, (0, y + 20))
            y += r.height + 22
        p = out_dir / f"aug_sample_{i:02d}.png"
        sheet.save(p)
        print(f"wrote {p.name}")
    print(f"DONE -> {out_dir}")


if __name__ == "__main__":
    main()
