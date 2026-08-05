"""Measure, on the REAL 0717+0731 data, the quantities that decide the EE rotation
representation question:

1. Does the ABSOLUTE tool orientation ever approach the rotation-vector seam at 180 deg?
   (If yes, rotvec is unsafe for the STATE input; 6D is required there.)
2. How large are CHUNK-DELTA rotations (state_t vs commanded pose at t+k, k<=50)?
   (If small, delta-rotvec is smooth/seam-free and ideal as the ACTION target.)
3. How wrong is naive rotvec subtraction vs the proper relative rotation, on real chunks?
4. Scale check for naive 6D subtraction: embedding-difference norm vs true geodesic angle.
"""
import sys

import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL")
from wxai_kinematics import fk_matrix, matrix_to_rotvec

DATA = Path("/iris/projects/humanoid/trossen_data")
DIRS = [DATA / "0717_green_yellow_block_mem_merged", DATA / "0731_green_yellow_merged"]
H = 50

abs_angles, delta_angles, naive_err, six_ratio = [], [], [], []
n_eps = 0
for root in DIRS:
    for pq in sorted((root / "data" / "chunk-000").glob("episode_*.parquet")):
        df = pd.read_parquet(pq, columns=["observation.state", "action"])
        s = np.stack(df["observation.state"].to_numpy()).astype(np.float64)[:, 7:13]
        a = np.stack(df["action"].to_numpy()).astype(np.float64)[:, 7:13]
        if len(s) < 10:
            continue
        n_eps += 1
        idx = range(0, len(s), 10)
        Rs = {t: fk_matrix(s[t])[:3, :3] for t in idx}
        for t in idx:
            R_state = Rs[t]
            rv_abs = matrix_to_rotvec(R_state)
            abs_angles.append(np.linalg.norm(rv_abs))
            k = min(t + H - 1, len(a) - 1)          # farthest commanded pose in the chunk
            R_cmd = fk_matrix(a[k])[:3, :3]
            R_delta = R_state.T @ R_cmd
            rv_delta = matrix_to_rotvec(R_delta)
            delta_angles.append(np.linalg.norm(rv_delta))
            # naive rotvec subtraction vs proper delta (angle of the discrepancy rotation)
            rv_cmd = matrix_to_rotvec(R_cmd)
            naive = rv_cmd - rv_abs
            from wxai_kinematics import rotvec_to_matrix
            R_err = rotvec_to_matrix(naive).T @ R_delta      # hmm: compare in a common frame
            # simpler, unambiguous scalar: |angle(naive-as-rotation applied to state) vs cmd|
            R_from_naive = R_state @ rotvec_to_matrix(naive)
            naive_err.append(np.linalg.norm(matrix_to_rotvec(R_from_naive.T @ R_cmd)))
            # 6D: embedding-space difference norm per degree of true rotation
            d6 = np.linalg.norm((R_cmd[:, :2] - R_state[:, :2]).ravel())
            if np.linalg.norm(rv_delta) > 1e-4:
                six_ratio.append(d6 / np.linalg.norm(rv_delta))

abs_angles = np.array(abs_angles); delta_angles = np.array(delta_angles)
naive_err = np.array(naive_err); six_ratio = np.array(six_ratio)
deg = np.degrees
print(f"{n_eps} episodes, {len(abs_angles)} sampled timesteps\n")
print(f"1. ABSOLUTE tool orientation |rotvec| (deg): median {deg(np.median(abs_angles)):.1f}  "
      f"p99 {deg(np.percentile(abs_angles, 99)):.1f}  max {deg(abs_angles.max()):.1f}  "
      f"(seam at 180; margin {deg(np.pi - abs_angles.max()):.1f} deg)")
print(f"2. CHUNK-DELTA rotation over <=50 steps (deg): median {deg(np.median(delta_angles)):.2f}  "
      f"p99 {deg(np.percentile(delta_angles, 99)):.1f}  max {deg(delta_angles.max()):.1f}")
print(f"3. naive rotvec-subtraction error vs proper delta (deg): median {deg(np.median(naive_err)):.3f}  "
      f"p99 {deg(np.percentile(naive_err, 99)):.2f}  max {deg(naive_err.max()):.1f}")
print(f"4. 6D embedding-diff norm per rad of true rotation: median {np.median(six_ratio):.3f}  "
      f"spread p1-p99 [{np.percentile(six_ratio,1):.3f}, {np.percentile(six_ratio,99):.3f}] "
      f"(1.0 = isometric; wide spread = magnitude distorts by axis)")
print("DONE")
