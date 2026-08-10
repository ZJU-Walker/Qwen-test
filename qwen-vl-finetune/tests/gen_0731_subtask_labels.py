"""Generate subtask labels for 0717 AND 0731 from proprioception (name kept from the
original 0731-only version).

The block-memory episodes are: human points (robot arm STILL) -> robot picks.

CUT RULE (2026-08-10): ignore startup motion until the arm has been quiet for QUIET
consecutive frames at/after MIN_ONSET; onset is then the first SUSTAINED right-arm
motion (joint-velocity norm > THRESH for SUSTAIN consecutive frames), and
cut = onset + CUT_OFFSET. This rejects both brief noise and long startup repositioning
that otherwise put the cut before the human pointing phase. The "waiting" segment is
[0, cut-1], the pick segment [cut, end]. Recordings shorter than MIN_EPISODE_LEN or
without a clean quiet->motion transition are waiting-only and therefore dropped.

Color sources (never detected):
  - 0717: the pick task text of the episode's ORIGINAL human labels. Episodes without
    the canonical waiting->pick human structure (only ep 149) are copied through
    UNCHANGED, so the skip_leading_subtask gate still drops them.
  - 0731: the visually-verified collection-order boundary (eps 0-109 green, 110+
    yellow); degenerate/partial recordings get waiting-only labels.

VALIDATION before writing anything: the detector runs on 0717's human labels
(median onset-vs-human distance must stay ~15 frames, their known labeling bias) and
0731 colors are cross-checked against brian's independently generated file (must agree
exactly; boundary diffs vs his onset-15 cuts are ~25 frames by construction).

OUTPUT MODES:
  python tests/gen_0731_subtask_labels.py                 # dry run -> /tmp/{0717,0731}_subtask_labels.json
  python tests/gen_0731_subtask_labels.py --install --force   # write into BOTH datasets'
                                                          #   videos/chunk-000/subtask_labels.json

--install OVERWRITES 0717's human-made labels file; the original is backed up first to
subtask_labels_human_backup.json in the same dir (once -- an existing backup is never
touched, and it is also what validation reads after the first install). Every run that
reads either dataset sees the new labels from then on, including a rerun of the old
ee6d recipe (premise "0731 unlabeled", 0717 human cut points).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/iris/projects/humanoid/trossen_data")
D0717 = DATA / "0717_green_yellow_block_mem_merged"
D0731 = DATA / "0731_green_yellow_merged"
BRIAN_REF = DATA / "scripts_brian" / "0731_subtask_labels_generated.json"
ARM = slice(7, 13)      # right-arm joints (excl. gripper) in the 14-dim state
THRESH = 0.005          # rad/frame, sustained
SUSTAIN = 5
QUIET = 10              # require a real stationary window before accepting motion
MIN_ONSET = 40          # startup repositioning can last ~1 s; pointing lasts longer
CUT_OFFSET = 10         # frames after motion onset (see CUT RULE above)
MIN_EPISODE_LEN = 50    # matches the training recipe; shorter recordings are unusable
# Human labelers marked the pick ~0.5s BEFORE physical arm motion (they keyed off the
# pointing hand retracting): a constant +15-frame bias of motion-onset vs human label,
# stable across thresholds (+13..+17). Used ONLY to validate the detector against them.
LABEL_BIAS = 15


def labels_path(root):
    return root / "videos" / "chunk-000" / "subtask_labels.json"


def load_0717_human_labels():
    backup = labels_path(D0717).with_name("subtask_labels_human_backup.json")
    return json.load(open(backup if backup.exists() else labels_path(D0717)))


def detect_move_start(parquet_path):
    df = pd.read_parquet(parquet_path, columns=["observation.state"])
    s = np.stack(df["observation.state"].to_numpy()).astype(np.float32)[:, ARM]
    v = np.linalg.norm(np.diff(s, axis=0), axis=1)          # per-frame joint speed
    above = v > THRESH
    quiet_run = 0
    motion_run = 0
    armed = False
    for i, a in enumerate(above):
        frame = i + 1  # diff[i] is the transition into state frame i+1
        if not a:
            quiet_run += 1
            motion_run = 0
            if quiet_run >= QUIET:
                armed = True
        else:
            quiet_run = 0
            if armed:
                motion_run += 1
                if motion_run >= SUSTAIN:
                    onset = frame - SUSTAIN + 1
                    if onset >= MIN_ONSET:
                        return onset
                    # A real quiet window followed by motion before frame 40 is still
                    # startup repositioning (seen in both datasets). Require another
                    # quiet window before accepting a later motion candidate.
                    armed = False
                    motion_run = 0
    return None


def episode_len(root, key):
    pq = root / "data" / "chunk-000" / key.replace(".mp4", ".parquet")
    return len(pd.read_parquet(pq, columns=["observation.state"])), pq


def canonical(segs):
    return (len(segs) > 1 and segs[0]["task"] == "waiting" and segs[0]["start"] == 0
            and sum(s["task"] == "waiting" for s in segs) == 1)


def validate_on_0717(human_labels):
    errs = []
    for key, segs in sorted(human_labels.items()):
        if not canonical(segs):
            continue
        gt = segs[1]["start"]
        det = detect_move_start(D0717 / "data" / "chunk-000" / (key.replace(".mp4", "") + ".parquet"))
        assert det is not None, f"no clean motion onset in canonical 0717 episode {key}"
        errs.append(det - LABEL_BIAS - gt)
    errs = np.asarray(errs)
    print(f"detector validation on 0717 human labels ({len(errs)} canonical episodes): "
          f"median |err| {np.median(np.abs(errs)):.0f} frames, p90 {np.percentile(np.abs(errs), 90):.0f}, "
          f"max {np.abs(errs).max():.0f}  (bias-corrected; 30 fps)")
    assert np.median(np.abs(errs)) <= 8, "detector disagrees with human labels -- do not trust"


def relabel(n, onset, task):
    move = int(np.clip(onset + CUT_OFFSET, 1, n - 2))
    return [{"task": "waiting", "start": 0, "end": move - 1},
            {"task": task, "start": move, "end": n - 1}]


def generate_0717(human_labels):
    out = {}
    for key, segs in sorted(human_labels.items()):
        if not canonical(segs):
            out[key] = segs   # ep 149: copied unchanged -> still dropped by the gate
            print(f"  0717 {key}: non-canonical human labels copied unchanged")
            continue
        n, pq = episode_len(D0717, key)
        onset = detect_move_start(pq)
        assert onset is not None, f"no clean motion onset in canonical 0717 episode {key}"
        out[key] = relabel(n, onset, segs[1]["task"])   # color from the human label
    report("0717", out)
    return out


def generate_0731():
    out = {}
    for i in range(224):
        key = f"episode_{i:06d}.mp4"
        n, pq = episode_len(D0731, key)
        task = "pick up green block" if i <= 109 else "pick up yellow block"   # visual boundary, verified both sides
        onset = detect_move_start(pq) if n >= MIN_EPISODE_LEN else None
        if onset is None:
            # Includes five sub-50-frame recordings (128/129/133/150/153), episode 130
            # (robot already moving at the beginning), and episode 152 (starts mid-grasp).
            # No clean prompted pick to imitate: waiting-only makes the skip gate drop it.
            out[key] = [{"task": "waiting", "start": 0, "end": n - 1}]
            print(f"  0731 unusable episode {i}: {n} frames, no clean transition "
                  f"-> whole-episode 'waiting'")
            continue
        out[key] = relabel(n, onset, task)
    report("0731", out)
    return out


def report(name, out):
    starts = [v[1]["start"] for v in out.values() if len(v) > 1]
    print(f"{name}: {len(out)} episodes labeled ({len(starts)} with picks); cut median "
          f"{int(np.median(starts))} frames ({np.median(starts)/30:.1f}s), "
          f"range [{min(starts)}, {max(starts)}]")


def cross_check_vs_brian(out):
    """0731 color agreement vs brian's independent 2026-08-02 run (boundaries differ by
    ~CUT_OFFSET+LABEL_BIAS by construction -- his cuts were onset-15, ours onset+10)."""
    try:
        ref = json.load(open(BRIAN_REF))
    except OSError as e:
        print(f"cross-check SKIPPED (reference not readable: {e})")
        return
    color_mismatch, start_diffs, intentionally_dropped = [], [], []
    for key, segs in out.items():
        rsegs = ref.get(key)
        if rsegs is None:
            continue
        ours = [s["task"] for s in segs if s["task"] != "waiting"]
        theirs = [s["task"] for s in rsegs if s["task"] != "waiting"]
        if not ours:
            intentionally_dropped.append(key)
            continue
        if ours != theirs:
            color_mismatch.append(key)
        if len(segs) > 1 and len(rsegs) > 1:
            start_diffs.append(segs[1]["start"] - rsegs[1]["start"])
    d = np.asarray(start_diffs)
    print(f"0731 cross-check vs {BRIAN_REF.name}: {len(start_diffs)} episodes compared, "
          f"boundary diff median {np.median(d):+.0f} frames (expected ~+{CUT_OFFSET + LABEL_BIAS})")
    assert not color_mismatch, f"COLOR MISMATCH vs brian's labels: {color_mismatch[:5]} -- stop and inspect"
    print(f"intentionally dropped before color comparison: {intentionally_dropped}")
    print("colors agree on all compared episodes")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--install", action="store_true",
                    help="write into both datasets' videos/chunk-000/subtask_labels.json "
                         "(0717's human original is backed up first, once)")
    ap.add_argument("--force", action="store_true",
                    help="required to overwrite existing installed labels files")
    args = ap.parse_args()

    human_0717 = load_0717_human_labels()
    validate_on_0717(human_0717)
    generated = {D0717: generate_0717(human_0717), D0731: generate_0731()}
    cross_check_vs_brian(generated[D0731])

    for root, out in generated.items():
        if args.install:
            target = labels_path(root)
            if target.exists() and not args.force:
                raise SystemExit(f"{target} already exists; rerun with --force to overwrite")
            if root == D0717:
                backup = target.with_name("subtask_labels_human_backup.json")
                if not backup.exists():
                    shutil.copy2(target, backup)
                    print(f"backed up 0717 human labels -> {backup}")
        else:
            target = Path(f"/tmp/{root.name.split('_')[0]}_subtask_labels.json")
        json.dump(out, open(target, "w"), indent=2)
        print(f"wrote {target}" + ("" if args.install else "  (dry run -- use --install --force to write into the datasets)"))


if __name__ == "__main__":
    main()
