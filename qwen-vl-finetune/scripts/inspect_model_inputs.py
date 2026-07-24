"""Sanity-check the image pipeline: what resolution does the model ACTUALLY see?

For a real episode timestep this script traces every stage of the input pipeline
(native video frame -> dataset pre-resize -> processor smart-resize -> patch grid ->
merged vision tokens), prints the numbers, and saves PNGs you can open:

  <out>/report.txt                        full breakdown
  <out>/hist_XX_compare.png               history frame: [native | model view (upscaled, NEAREST)]
  <out>/hist_XX_model.png                 the model-resolution image itself (small)
  <out>/wrist_compare.png / _model.png    same for the wrist still

The "model view" is the original frame resized to EXACTLY the resolution the processor
hands the vision tower -- (grid_h*patch) x (grid_w*patch) px (patch from the processor config), read from grid_thw,
which is authoritative. It is then upscaled back to native size with NEAREST so the
blockiness/information loss is visible instead of hidden by your image viewer.

Notes on the stages (what is and is not lossy):
  1. camera -> stored video (AV1) or JPEG q90 at deploy: lossy codec, resolution kept
  2. dataset/client pre-resize: wrist (and no-history top still) downscaled to
     --wrist_max_pixels / --max_pixels BEFORE the processor (the processor does not
     resize pre-loaded PIL images)
  3. processor smart-resize: the REAL resolution bottleneck; for videos a budget is
     spread across the whole clip, so per-frame pixels shrink as frames grow
  4. 14x14 patchify + 2-frame temporal pairing: lossless rearrangement
  5. 2x2 spatial merge -> 1 token per 28x28 px block: learned compression in the
     vision tower (representation, not pixel loss)

Run (CPU only, no GPU/model needed):
  python scripts/inspect_model_inputs.py --episode 0 --t 100
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

from qwenvl.action_expert.inference import DATA_DIRS, FAST_TOK, build_dataset, make_prompt, templatize


def model_view(img: Image.Image, grid_h: int, grid_w: int, patch: int) -> Image.Image:
    """The image at EXACTLY the resolution the vision tower receives (grid * patch px)."""
    return img.resize((grid_w * patch, grid_h * patch), Image.BICUBIC)


def compare_png(native: Image.Image, model_img: Image.Image, path: Path, label: str):
    """[native | model view upscaled with NEAREST to native size] side by side."""
    up = model_img.resize(native.size, Image.NEAREST)  # nearest exposes the true blockiness
    w, h = native.size
    canvas = Image.new("RGB", (w * 2 + 8, h), (255, 255, 255))
    canvas.paste(native, (0, 0))
    canvas.paste(up, (w + 8, 0))
    canvas.save(path)
    return f"{label}: native {native.size[0]}x{native.size[1]} | model {model_img.size[0]}x{model_img.size[1]} -> {path.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast_tok", default=FAST_TOK)
    ap.add_argument("--data_dirs", default=DATA_DIRS)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--t", type=int, default=-1, help="timestep (-1 = middle of the episode)")
    ap.add_argument("--no_image_history", action="store_true")
    ap.add_argument("--out", default="/iris/projects/humanoid/ke/Qwen3-VL/input_inspection")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    ds = build_dataset(fast_tok=args.fast_tok, data_dirs=args.data_dirs,
                       image_history=not args.no_image_history, predict_subtask=True)
    ep = ds.episodes[args.episode]
    t = args.t if args.t >= 0 else len(ep["states"]) // 2

    # ---- processor config that governs the resizing ----
    vp = ds.processor.video_processor
    ip = ds.processor.image_processor
    log(f"episode {args.episode} t={t}  video={Path(ep['video_path']).name}")
    log("\n=== processor config (the dials that control resolution) ===")
    for name, obj in (("video_processor", vp), ("image_processor", ip)):
        keys = [k for k in ("min_pixels", "max_pixels", "total_pixels", "min_frames", "max_frames",
                            "size", "patch_size", "temporal_patch_size", "merge_size", "fps")
                if getattr(obj, k, None) is not None]
        log(f"  {name}: " + "  ".join(f"{k}={getattr(obj, k)}" for k in keys))
    log(f"  data_args: max_pixels={ds.data_args.max_pixels}  min_pixels={ds.data_args.min_pixels}  "
        f"wrist_max_pixels={ds.data_args.wrist_max_pixels}  num_frames={ds.num_frames}  stride={ds.stride}")

    # ---- pull the exact inputs the server/dataset would build at this timestep ----
    # native = decoded straight from the stored video, NO budget resize (for comparison)
    native_full = ds._extract_single_frame(ep["video_path"], t, 10**9)
    if ds.data_args.image_history:
        frames = ds._extract_frames(ep, t)          # what templatize receives (raw frames)
    else:
        frames = [ds._extract_single_frame(ep["video_path"], t, ds.data_args.max_pixels)]
    wrist = ds._extract_wrist_images(ep, t)         # pre-resized to wrist_max_pixels
    wrist_native = None
    if ep.get("wrist_paths"):
        wrist_native = ds._extract_single_frame(ep["wrist_paths"][0], t, 10**9)

    prompt = make_prompt(ds, ep["states"][t])
    mm = templatize(ds, frames, wrist, prompt, None, "cpu")

    log(f"\n=== sequence ===\n  total prefix tokens: {mm['input_ids'].shape[1]}")

    # ---- video (history) stage-by-stage ----
    vgt = mm.get("video_grid_thw")
    if vgt is not None:
        tgrid, gh, gw = (int(x) for x in vgt[0])
        vpatch = vp.patch_size
        toks_per_group = (gh // vp.merge_size) * (gw // vp.merge_size)
        log("\n=== cam_high history (video path) ===")
        log(f"  native frame:        {native_full.size[0]}x{native_full.size[1]} "
            f"({native_full.size[0]*native_full.size[1]/1e3:.0f} kpx)")
        log(f"  passed to processor: {frames[0].size[0]}x{frames[0].size[1]} (no pre-resize in history mode)")
        log(f"  MODEL SEES:          {gw*vpatch}x{gh*vpatch} px per frame "
            f"({gw*vpatch*gh*vpatch/1e3:.1f} kpx = "
            f"{native_full.size[0]*native_full.size[1]/(gw*vpatch*gh*vpatch):.1f}x less area than native)")
        log(f"  grid: {tgrid} temporal groups (2 frames each) x {gh}x{gw} patches of {vpatch}px")
        log(f"  tokens: {toks_per_group}/group after 2x2 merge -> {tgrid*toks_per_group} video tokens total")
        for i, fr in enumerate(frames):
            mv = model_view(fr, gh, gw, vpatch)
            mv.save(out / f"hist_{i:02d}_model.png")
            log("  " + compare_png(fr, mv, out / f"hist_{i:02d}_compare.png",
                                   f"history frame {i} (t-{(len(frames)-1-i)*ds.stride})"))

    # ---- no-history still ----
    igt = mm.get("image_grid_thw")
    img_idx = 0
    if igt is not None and not ds.data_args.image_history:
        _, gh, gw = (int(x) for x in igt[img_idx])
        log("\n=== cam_high current still (no-history mode) ===")
        log(f"  native {native_full.size} -> pre-resized {frames[0].size} -> "
            f"MODEL SEES {gw*ip.patch_size}x{gh*ip.patch_size}")
        mv = model_view(frames[0], gh, gw, ip.patch_size)
        mv.save(out / "top_model.png")
        log("  " + compare_png(native_full, mv, out / "top_compare.png", "top still"))
        img_idx += 1

    # ---- wrist still ----
    if igt is not None and wrist:
        _, gh, gw = (int(x) for x in igt[img_idx])
        toks = (gh // ip.merge_size) * (gw // ip.merge_size)
        log("\n=== wrist still ===")
        if wrist_native is not None:
            log(f"  native:              {wrist_native.size[0]}x{wrist_native.size[1]} "
                f"({wrist_native.size[0]*wrist_native.size[1]/1e3:.0f} kpx)")
        log(f"  dataset pre-resize:  {wrist[0].size[0]}x{wrist[0].size[1]} (wrist_max_pixels={ds.data_args.wrist_max_pixels})")
        log(f"  MODEL SEES:          {gw*ip.patch_size}x{gh*ip.patch_size} px ({toks} tokens after merge)")
        mv = model_view(wrist[0], gh, gw, ip.patch_size)
        mv.save(out / "wrist_model.png")
        base = wrist_native if wrist_native is not None else wrist[0]
        log("  " + compare_png(base, mv, out / "wrist_compare.png", "wrist"))

    log(f"\nPNGs + this report saved to: {out}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
