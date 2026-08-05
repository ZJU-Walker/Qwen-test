"""Contact sheets for human verification of the GENERATED 0731 subtask labels.

Per episode: three cam_high frames at [boundary-15, boundary, boundary+15] (0.5s either
side of the detected waiting->pick transition). If the detector is right, the LEFT frame
shows the arm still (human pointing done/retracting), the RIGHT frame shows the arm
clearly moving toward the labeled block. Header shows episode id, labeled color, and
boundary frame. 16 episodes per sheet -> 14 sheets.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from torchvision.io import read_video

D0731 = Path("/iris/projects/humanoid/trossen_data/0731_green_yellow_merged")
LABELS = Path("/iris/projects/humanoid/trossen_data/scripts_brian/0731_subtask_labels_generated.json")
OUT = Path("/iris/projects/humanoid/trossen_data/scripts_brian/0731_label_verification")
OUT.mkdir(exist_ok=True)
FPS, THUMB_W, PAD = 30, 320, 4

labels = json.load(open(LABELS))
rows = []
for key in sorted(labels):
    idx = int(key.split("_")[1].split(".")[0])
    segs = labels[key]
    video = D0731 / "videos" / "chunk-000" / "observation.images.cam_high" / key
    if len(segs) == 1:  # degenerate episode, labeled pure waiting
        rows.append((idx, "DEGENERATE (waiting only)", None, video))
        continue
    rows.append((idx, segs[1]["task"], segs[1]["start"], video))


def grab(video, t):
    lo = max(0, t)
    v, _, _ = read_video(str(video), start_pts=lo / FPS, end_pts=(lo + 0.9) / FPS,
                         pts_unit="sec", output_format="TCHW")
    if len(v) == 0:
        v, _, _ = read_video(str(video), pts_unit="sec", output_format="TCHW")
        lo = min(max(0, t), len(v) - 1)
        frame = v[lo]
    else:
        frame = v[0]
    img = Image.fromarray(frame.permute(1, 2, 0).numpy())
    return img.resize((THUMB_W, int(img.height * THUMB_W / img.width)))


thumb_h = None
per_sheet = 16
for sheet_i in range(0, len(rows), per_sheet):
    batch = rows[sheet_i : sheet_i + per_sheet]
    tiles = []
    for idx, task, boundary, video in batch:
        if boundary is None:
            imgs = [grab(video, 0)] * 3
            title = f"ep {idx:03d}  {task}"
        else:
            imgs = [grab(video, boundary - 15), grab(video, boundary), grab(video, boundary + 15)]
            title = f"ep {idx:03d}  '{task}'  boundary={boundary} ({boundary/FPS:.1f}s)"
        w, h = imgs[0].size
        row_img = Image.new("RGB", (3 * w + 2 * PAD, h + 18), "white")
        for j, im in enumerate(imgs):
            row_img.paste(im, (j * (w + PAD), 18))
        ImageDraw.Draw(row_img).text((4, 2), title + "   [-0.5s | boundary | +0.5s]", fill="black")
        tiles.append(row_img)
    W = max(t.width for t in tiles)
    sheet = Image.new("RGB", (W, sum(t.height + PAD for t in tiles)), "white")
    y = 0
    for t in tiles:
        sheet.paste(t, (0, y))
        y += t.height + PAD
    out = OUT / f"sheet_{sheet_i // per_sheet:02d}_eps{batch[0][0]:03d}-{batch[-1][0]:03d}.png"
    sheet.save(out)
    print(f"wrote {out.name}")
print(f"DONE: {len(rows)} episodes across {(len(rows) + per_sheet - 1) // per_sheet} sheets in {OUT}")
