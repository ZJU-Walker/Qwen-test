"""EE-6D representation core gates (CPU, real data). Must pass BEFORE any training.

1. 6D encode/decode round trip on random rotations (incl. near-180 deg): exact.
2. IK(FK(q)) recovery on real dataset frames, seeded realistically (previous frame's
   joints): the branch the robot executes is recovered, not a mirror solution.
3. THE gate: full-pipeline chunk round trip on real episodes --
   joints -> EE10 -> delta (mask 9,-1) -> quantile normalize -> unnormalize -> undo
   -> Gram-Schmidt -> sequential IK -> joints.
   Reported in end-effector mm / deg and joint rad. This is the representation's total
   cost to precision; it must be far below the mm scale we are chasing.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune")

from qwenvl.data.ee_repr import ee_chunk_to_joints, gs_decode, joints_to_ee, sixd_from_matrix
from qwenvl.data.robot_data import (
    compute_norm_stats, make_action_chunk, make_delta_mask, quantile_normalize,
    quantile_unnormalize, undo_delta_actions,
)
from qwenvl.data.wxai_kinematics import fk_matrix, matrix_to_rotvec, rotvec_to_matrix

DATA = Path("/iris/projects/humanoid/trossen_data")
DIRS = [DATA / "0717_green_yellow_block_mem_merged", DATA / "0731_green_yellow_merged"]
H = 50
rng = np.random.default_rng(0)

# ---- 1. 6D round trip ----
max_err = 0.0
for _ in range(2000):
    rv = rng.standard_normal(3)
    rv = rv / np.linalg.norm(rv) * rng.uniform(0, np.pi - 1e-3)
    R = rotvec_to_matrix(rv)
    R2 = gs_decode(sixd_from_matrix(R))
    max_err = max(max_err, np.abs(R2 - R).max())
assert max_err < 1e-9, f"6D round trip broken: {max_err}"
print(f"1. 6D encode/decode round trip exact (max |dR| {max_err:.1e})  OK")

# ---- load real episodes (right arm), subsampled ----
eps = []
for root in DIRS:
    for pq in sorted((root / "data" / "chunk-000").glob("episode_*.parquet"))[::4]:
        df = pd.read_parquet(pq, columns=["observation.state", "action"])
        s = np.stack(df["observation.state"].to_numpy()).astype(np.float32)[:, 7:14]
        a = np.stack(df["action"].to_numpy()).astype(np.float32)[:, 7:14]
        if len(s) >= 60:
            eps.append({"states": s, "actions": a})
print(f"   loaded {len(eps)} episodes for real-data gates")

# ---- 2. IK(FK(q)) with realistic seeding ----
jerrs = []
for ep in eps[::6]:
    q = ep["actions"]
    for t in range(1, min(len(q), 200), 25):
        ee = joints_to_ee(q[t])
        rec, ok = __import__("qwenvl.data.ee_repr", fromlist=["ee_to_joints"]).ee_to_joints(ee, q[t - 1][:6])
        assert ok, f"IK failed to converge at a real dataset pose (t={t})"
        jerrs.append(np.abs(rec[:6] - q[t][:6]).max())
jerrs = np.array(jerrs)
print(f"2. IK(FK(q)) seeded from previous frame: n={len(jerrs)}  "
      f"max joint err {jerrs.max():.2e} rad  median {np.median(jerrs):.2e}  OK")
assert jerrs.max() < 1e-3, "IK recovers a different branch than the data executed"

# ---- 3. full-pipeline chunk round trip ----
ee_eps = [{"states": joints_to_ee(ep["states"]), "actions": joints_to_ee(ep["actions"]),
           "min_start": 0} for ep in eps]
delta_mask = make_delta_mask("9,-1")
stats = compute_norm_stats(ee_eps, H, delta_mask)
rngs = np.asarray(stats["actions"]["q99"]) - np.asarray(stats["actions"]["q01"])
print(f"3a. norm-stat sanity: per-dim q99-q01 ranges:\n    {np.array2string(rngs, precision=4)}")
assert rngs.min() > 1e-3, f"degenerate action dim (range {rngs.min():.1e}) -- quantile norm would amplify noise"

pos_errs, rot_errs, jaw_errs, joint_errs, n_bad = [], [], [], [], 0
for ep, eej in list(zip(eps, ee_eps))[::6]:
    for t in range(0, len(ep["states"]) - H, 40):
        state_ee = eej["states"][t]
        chunk = make_action_chunk(eej["actions"], state_ee, t, H, delta_mask)
        norm = quantile_normalize(chunk, stats["actions"])
        rt = quantile_unnormalize(norm, stats["actions"])
        abs_ee = undo_delta_actions(rt, state_ee, delta_mask)
        joints, bad = ee_chunk_to_joints(abs_ee, ep["states"][t][:6])
        n_bad += bad
        gt = ep["actions"][t : t + H]
        joints = joints[: len(gt)]
        joint_errs.append(np.abs(joints[:, :6] - gt[:, :6]).max())
        jaw_errs.append(np.abs(joints[:, 6] - gt[:, 6]).max())
        for j in range(0, len(gt), 10):
            T_gt = fk_matrix(gt[j][:6]); T_rt = fk_matrix(joints[j][:6])
            pos_errs.append(np.linalg.norm(T_gt[:3, 3] - T_rt[:3, 3]) * 1000.0)
            rot_errs.append(np.degrees(np.linalg.norm(matrix_to_rotvec(T_gt[:3, :3].T @ T_rt[:3, :3]))))
pos_errs, rot_errs = np.array(pos_errs), np.array(rot_errs)
joint_errs, jaw_errs = np.array(joint_errs), np.array(jaw_errs)
print(f"3b. full round trip over {len(joint_errs)} chunks ({len(pos_errs)} poses), IK non-converged steps: {n_bad}")
print(f"    EE position error:  max {pos_errs.max():.4f} mm   median {np.median(pos_errs):.5f} mm")
print(f"    EE rotation error:  max {rot_errs.max():.4f} deg  median {np.median(rot_errs):.5f} deg")
print(f"    joint error:        max {joint_errs.max():.2e} rad;  jaw max {jaw_errs.max():.2e}")
assert pos_errs.max() < 0.5, "representation costs >0.5mm -- unacceptable for a mm-precision hunt"
assert rot_errs.max() < 0.1, "representation costs >0.1deg"
assert n_bad == 0, "IK failed on real data chunks"
print("\nALL EE6D CORE GATES PASS")
