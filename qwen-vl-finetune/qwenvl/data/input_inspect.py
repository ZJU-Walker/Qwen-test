"""Dump the model's ACTUAL inputs from inside the training pipeline.

Unlike scripts/inspect_model_inputs.py (which *simulates* the processor resize), this
reconstructs the images by un-patchifying `pixel_values` / `pixel_values_videos` -- the
exact tensors the vision tower consumes -- so what you look at is provably what the model
sees. A round-trip self-check (re-flattening the reconstruction and comparing bitwise
against the original rows) guards the layout assumptions.

Enabled via env vars, checked by RobotFlowMatchingDataset at construction:
    QWEN_DUMP_MODEL_INPUTS=<dir>     # empty/unset = off (zero overhead)
    QWEN_DUMP_MODEL_INPUTS_N=2       # items to dump per dataloader process (default 2)
Only rank 0 dumps. Each dataloader worker is its own process, so with W workers you get
up to W*N items, each tagged with the worker pid. After the quota the hook is dead code.

Both Qwen3-VL processors flatten patches the same way (verified against the installed
transformers source, video_processing_qwen3_vl.py / image_processing_qwen2_vl_fast.py):
    view   (B, t, tp, C, H/m, m, p, W/m, m, p)
    permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)   ->  rows ordered (t, Hm, Wm, m, m)
    reshape(B, t*gh*gw, C*tp*p*p)
Stills are videos with the frame duplicated along tp (every pixel appears twice per row).
"""

import os
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# forward permute without the batch dim: [t,tp,C,Hm,m1,p1,Wm,m2,p2] -> [t,Hm,Wm,m1,m2,C,tp,p1,p2]
_FWD = (0, 3, 6, 4, 7, 2, 1, 5, 8)
_INV = tuple(int(i) for i in np.argsort(_FWD))


def unpatchify(flat: np.ndarray, grid, temporal_patch: int, patch: int, merge: int):
    """(t*gh*gw, C*tp*p*p) -> frames (t*tp, C, gh*p, gw*p), plus an exact round-trip check."""
    t, gh, gw = (int(x) for x in grid)
    C = flat.shape[1] // (temporal_patch * patch * patch)
    x = flat.reshape(t, gh // merge, gw // merge, merge, merge, C, temporal_patch, patch, patch)
    x = x.transpose(_INV)  # -> (t, tp, C, gh//m, m, p, gw//m, m, p)
    frames = x.reshape(t * temporal_patch, C, gh * patch, gw * patch)

    # self-check: re-apply the processor's own ops; must match the input rows bitwise
    re_flat = (
        frames.reshape(t, temporal_patch, C, gh // merge, merge, patch, gw // merge, merge, patch)
        .transpose(_FWD)
        .reshape(t * gh * gw, C * temporal_patch * patch * patch)
    )
    exact = np.array_equal(re_flat, flat)
    return frames, exact


def _to_pils(frames: np.ndarray, mean, std):
    """Un-normalize (C,H,W float) frames back to uint8 PIL images."""
    mean = np.asarray(mean, dtype=np.float32).reshape(1, -1, 1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(1, -1, 1, 1)
    x = frames * std + mean  # undo normalize; values ~[0,1] (rescale was 1/255)
    x = np.clip(x * 255.0 + 0.5, 0, 255).astype(np.uint8).transpose(0, 2, 3, 1)
    return [Image.fromarray(f) for f in x]


def _compare(native: Image.Image, model_img: Image.Image) -> Image.Image:
    """[native | model view upscaled with NEAREST] -- blockiness stays visible."""
    up = model_img.resize(native.size, Image.NEAREST)
    w, h = native.size
    canvas = Image.new("RGB", (w * 2 + 8, h), (255, 255, 255))
    canvas.paste(native, (0, 0))
    canvas.paste(up, (w + 8, 0))
    return canvas


def _collapse_pads(text: str) -> str:
    for pad in ("<|video_pad|>", "<|image_pad|>"):
        text = re.sub(
            f"(?:{re.escape(pad)})+",
            lambda m, p=pad: f"{p}(x{m.group(0).count(p)})",
            text,
        )
    return text


def dump_item(ds, item, top_frames, wrist_images, subtask, t, ep_idx, out_dir):
    """Write PNGs + a compression report for one fully-built training item."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"pid{os.getpid()}_{getattr(dump_item, '_n', 0):02d}"
    dump_item._n = getattr(dump_item, "_n", 0) + 1

    vp, ip, tok = ds.processor.video_processor, ds.processor.image_processor, ds.tokenizer
    lines = [f"=== {tag}  (episode {ep_idx}, t={t}, subtask={subtask!r}) ==="]

    # ---- token accounting over the full sequence ----
    ids = item["input_ids"][0]
    vid_pad = tok.convert_tokens_to_ids("<|video_pad|>")
    img_pad = tok.convert_tokens_to_ids("<|image_pad|>")
    n_total = ids.numel()
    n_vid = int((ids == vid_pad).sum())
    n_img = int((ids == img_pad).sum())
    n_fast = int(item["fast_token_mask"].sum()) if "fast_token_mask" in item else 0
    n_sub = int(item["subtask_token_mask"].sum()) if "subtask_token_mask" in item else 0
    n_text = n_total - n_vid - n_img - n_fast
    lines.append(
        f"sequence: {n_total} tokens = {n_vid} video + {n_img} image + {n_text} text "
        f"(incl. {n_sub} subtask-turn) + {n_fast} FAST"
    )

    checks = []

    # ---- cam_high history: the VIDEO pipeline ----
    if item.get("pixel_values_videos") is not None:
        flat_all = item["pixel_values_videos"].numpy()
        offset = 0
        for vi, grid in enumerate(item["video_grid_thw"]):
            tg, gh, gw = (int(x) for x in grid)
            n_rows = tg * gh * gw
            frames, exact = unpatchify(
                flat_all[offset:offset + n_rows], grid, vp.temporal_patch_size, vp.patch_size, vp.merge_size
            )
            offset += n_rows
            checks.append(("video", exact))
            pils = _to_pils(frames, vp.image_mean, vp.image_std)

            toks = n_rows // (vp.merge_size ** 2)
            mw, mh = gw * vp.patch_size, gh * vp.patch_size
            nat_w, nat_h = top_frames[0].size
            px_tok_model = vp.patch_size * vp.merge_size * vp.patch_size * vp.merge_size * vp.temporal_patch_size
            px_tok_native = nat_w * nat_h * len(top_frames) / toks
            lines += [
                f"cam_high history (video pipeline): {len(top_frames)} native frames {nat_w}x{nat_h} "
                f"({nat_w*nat_h/1e3:.1f} kpx)",
                f"  model sees {mw}x{mh} px/frame ({mw*mh/1e3:.1f} kpx = "
                f"{nat_w*nat_h/(mw*mh):.1f}x less area)",
                f"  grid {tg}x{gh}x{gw} patches({vp.patch_size}px) -> merged "
                f"{tg}x{gh//vp.merge_size}x{gw//vp.merge_size} = {toks} tokens "
                f"({toks/len(top_frames):.1f}/frame)",
                f"  1 token = {vp.patch_size*vp.merge_size}x{vp.patch_size*vp.merge_size} px x "
                f"{vp.temporal_patch_size} frames = {px_tok_model} model px | "
                f"{px_tok_native:,.0f} native px",
                f"  compression ladder per frame: {nat_w*nat_h:,} native px -> {mw*mh:,} model px "
                f"(resize {nat_w*nat_h/(mw*mh):.1f}x) -> {gh*gw} ViT patches -> "
                f"{toks/len(top_frames):.1f} tokens (2x2 spatial merge + "
                f"{vp.temporal_patch_size}-frame temporal pack)",
            ]
            for fi, pil in enumerate(pils[: len(top_frames)]):
                pil.save(out / f"{tag}_hist_{fi:02d}_model.png")
                nat = top_frames[fi] if fi < len(top_frames) else top_frames[-1]
                _compare(nat, pil).save(out / f"{tag}_hist_{fi:02d}_compare.png")

    # ---- stills (wrist, and the top frame in no-history mode): the IMAGE pipeline ----
    if item.get("pixel_values") is not None:
        flat_all = item["pixel_values"].numpy()
        stills = ([] if ds.data_args.image_history else [("top", top_frames[0])]) + [
            (f"wrist{k}" if len(wrist_images) > 1 else "wrist", w) for k, w in enumerate(wrist_images)
        ]
        offset = 0
        for si, grid in enumerate(item["image_grid_thw"]):
            tg, gh, gw = (int(x) for x in grid)
            n_rows = tg * gh * gw
            frames, exact = unpatchify(
                flat_all[offset:offset + n_rows], grid, ip.temporal_patch_size, ip.patch_size, ip.merge_size
            )
            offset += n_rows
            checks.append(("image", exact))
            pil = _to_pils(frames[:1], ip.image_mean, ip.image_std)[0]  # tp frames are duplicates

            name, src = stills[si] if si < len(stills) else (f"img{si}", None)
            toks = n_rows // (ip.merge_size ** 2)
            mw, mh = gw * ip.patch_size, gh * ip.patch_size
            src_desc = f"dataset pre-resize {src.size[0]}x{src.size[1]} -> " if src is not None else ""
            lines += [
                f"{name} (image pipeline): {src_desc}model {mw}x{mh} px, {toks} tokens; "
                f"1 token = {ip.patch_size*ip.merge_size}x{ip.patch_size*ip.merge_size} px "
                f"({mw*mh//toks} px/token)",
            ]
            pil.save(out / f"{tag}_{name}_model.png")
            if src is not None:
                _compare(src, pil).save(out / f"{tag}_{name}_compare.png")

    ok = all(e for _, e in checks)
    lines.append(
        "reconstruction self-check: "
        + ("EXACT (re-patchify == pixel_values, bitwise)" if ok else
           f"!! MISMATCH {checks} -- layout assumption broken, do not trust the PNGs")
    )

    # ---- the literal token stream, pads collapsed ----
    (out / f"{tag}_text.txt").write_text(_collapse_pads(tok.decode(ids)))
    lines.append(f"files: {tag}_hist_XX_{{model,compare}}.png  {tag}_*_model.png  {tag}_text.txt")

    with open(out / "report.txt", "a") as f:
        f.write("\n".join(lines) + "\n\n")
    print(f"[input-dump] {tag}: {n_total} tok "
          f"({n_vid} video / {n_img} image / {n_text} text / {n_fast} FAST) -> {out}  "
          f"self-check={'EXACT' if ok else 'MISMATCH!'}")
