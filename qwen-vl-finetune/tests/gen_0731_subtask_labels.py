"""Generate subtask labels for 0731_green_yellow_merged from proprioception.

The block-memory episodes are: human points (robot arm STILL) -> robot picks. The
waiting->pick boundary is therefore the first sustained motion of the right arm.
Detector: first frame where the right-arm joint-velocity norm exceeds THRESH for
SUSTAIN consecutive frames.

VALIDATION: 0717 has human-made labels ("waiting" [0,k], "pick up X" [k+1,...]) --
we run the detector on 0717 and report |detected - labeled| statistics before
trusting it on 0731. Output goes to a SCRATCH file (NOT into the 0731 dataset dir,
so the running Qwen 0731gy training and its unlabeled-episode premise are untouched);
the merge step splices it into the merged openpi dataset only.

Usage: python tests/gen_0731_subtask_labels.py <out_json>
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/iris/projects/humanoid/trossen_data")
D0717 = DATA / "0717_green_yellow_block_mem_merged"
D0731 = DATA / "0731_green_yellow_merged"
ARM = slice(7, 13)      # right-arm joints (excl. gripper) in the 14-dim state
THRESH = 0.005          # rad/frame, sustained
SUSTAIN = 5
# Human labelers mark the pick ~0.5s BEFORE physical arm motion (they key off the
# pointing hand retracting). Measured on 0717 ground truth: constant +15-frame bias of
# motion-onset vs human label, stable across thresholds (+13..+17). Subtract it.
LABEL_BIAS = 15


def detect_move_start(parquet_path):
    df = pd.read_parquet(parquet_path, columns=["observation.state"])
    s = np.stack(df["observation.state"].to_numpy()).astype(np.float32)[:, ARM]
    v = np.linalg.norm(np.diff(s, axis=0), axis=1)          # per-frame joint speed
    above = v > THRESH
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= SUSTAIN:
            return i - SUSTAIN + 2   # first frame of the sustained motion (+1 for diff offset)
    return 0


# ---- validate on 0717 (human labels = ground truth) ----
labels_0717 = json.load(open(D0717 / "videos" / "chunk-000" / "subtask_labels.json"))
errs = []
for key, segs in sorted(labels_0717.items()):
    waiting = [s for s in segs if s["task"] == "waiting"]
    picks = [s for s in segs if s["task"] != "waiting"]
    if len(waiting) != 1 or not picks or waiting[0]["start"] != 0:
        continue   # only episodes with the canonical waiting->pick structure validate cleanly
    gt = picks[0]["start"]
    pq = D0717 / "data" / "chunk-000" / (key.replace(".mp4", "") + ".parquet")
    det = max(1, detect_move_start(pq) - LABEL_BIAS)
    errs.append(det - gt)
errs = np.asarray(errs)
print(f"validation on 0717 ({len(errs)} canonical episodes): "
      f"median |err| {np.median(np.abs(errs)):.0f} frames, p90 {np.percentile(np.abs(errs), 90):.0f}, "
      f"max {np.abs(errs).max():.0f}  (bias {np.median(errs):+.0f}; 30 fps)")
assert np.median(np.abs(errs)) <= 8, "bias-corrected detector still disagrees with human labels -- do not trust"

# ---- generate for 0731 ----
out = {}
for i in range(224):
    key = f"episode_{i:06d}.mp4"
    pq = D0731 / "data" / "chunk-000" / f"episode_{i:06d}.parquet"
    n = len(pd.read_parquet(pq, columns=["observation.state"]))
    task = "pick up green block" if i <= 109 else "pick up yellow block"   # visual boundary, verified both sides
    if n < 10:
        # Degenerate aborted recording (0731 has three: eps 133/150/153 with 1-2 frames).
        # No behavior to imitate -- label as pure waiting so it can never teach a bogus pick.
        out[key] = [{"task": "waiting", "start": 0, "end": n - 1}]
        print(f"  degenerate episode {i}: {n} frames -> whole-episode 'waiting'")
        continue
    move = int(np.clip(detect_move_start(pq) - LABEL_BIAS, 1, n - 2))
    out[key] = [{"task": "waiting", "start": 0, "end": move - 1},
                {"task": task, "start": move, "end": n - 1}]
starts = [v[1]["start"] for v in out.values() if len(v) > 1]
print(f"0731: 224 episodes labeled; pick starts median {int(np.median(starts))} frames "
      f"({np.median(starts)/30:.1f}s), range [{min(starts)}, {max(starts)}]")

out_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/0731_subtask_labels.json"
json.dump(out, open(out_path, "w"), indent=2)
print(f"wrote {out_path}")
