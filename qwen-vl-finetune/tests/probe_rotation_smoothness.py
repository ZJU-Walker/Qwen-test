"""Zhou-et-al-style (arXiv:1812.07035) smoothness analysis of candidate rotation
representations, evaluated on OUR real chunk-delta distribution (no training needed).

Candidates for the ACTION orientation target:
  d-rv : delta rotation vector  rotvec(R_state^T R_k)          (3 dims)
  a-6d : absolute 6D            first two columns of R_k        (6 dims, GS decode)
  d-6d : naive 6D difference    6D(R_k) - 6D(R_state)           (6 dims, add-back + GS)

Tests (all reps compared in a scale-fair way: each dim standardized by its dataset std,
mirroring what quantile normalization does before the flow model sees anything):

A. CHART ROUGHNESS -- for consecutive chunk steps, ||rep_{k+1}-rep_k|| (normalized units)
   per degree of true geodesic rotation. A smooth chart has a tight, flat ratio; spikes
   mean small physical motions cause large target jumps (hard regression targets).

B. NOISE-ROBUSTNESS vs ANGLE (the paper's error-vs-angle curve, regression-free): add
   Gaussian noise of fixed normalized magnitude eps to the target rep -- simulating a
   fixed-size prediction error -- decode back to a rotation, measure geodesic error in
   degrees, binned by the chunk-delta angle theta. A well-conditioned rep has a flat
   curve; an ill-conditioned one amplifies errors at some angles.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL")
from wxai_kinematics import fk_matrix, matrix_to_rotvec, rotvec_to_matrix

DATA = Path("/iris/projects/humanoid/trossen_data")
DIRS = [DATA / "0717_green_yellow_block_mem_merged", DATA / "0731_green_yellow_merged"]
H, EPS, SEED = 50, 0.1, 0
rng = np.random.default_rng(SEED)


def sixd(R):
    return R[:, :2].T.ravel()          # row-major flatten of first two columns


def gs_decode(v):
    a1, a2 = v[:3], v[3:]
    b1 = a1 / max(np.linalg.norm(a1), 1e-9)
    a2p = a2 - np.dot(b1, a2) * b1
    b2 = a2p / max(np.linalg.norm(a2p), 1e-9)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def geo(Ra, Rb):
    return np.linalg.norm(matrix_to_rotvec(Ra.T @ Rb))


# ---- gather real (R_state, chunk R_k sequence) samples ----
samples = []       # (R_state, [R_k for k in ksteps], theta_end)
for root in DIRS:
    for pq in sorted((root / "data" / "chunk-000").glob("episode_*.parquet"))[::2]:
        df = pd.read_parquet(pq, columns=["observation.state", "action"])
        s = np.stack(df["observation.state"].to_numpy()).astype(np.float64)[:, 7:13]
        a = np.stack(df["action"].to_numpy()).astype(np.float64)[:, 7:13]
        if len(s) < 60:
            continue
        for t in range(0, len(s) - H, 30):
            R_state = fk_matrix(s[t])[:3, :3]
            Rks = [fk_matrix(a[t + k])[:3, :3] for k in range(0, H, 7)]
            samples.append((R_state, Rks))
print(f"{len(samples)} chunks sampled")

REPS = {
    "d-rv": lambda Rs, Rk: matrix_to_rotvec(Rs.T @ Rk),
    "a-6d": lambda Rs, Rk: sixd(Rk),
    "d-6d": lambda Rs, Rk: sixd(Rk) - sixd(Rs),
}
DECODE = {
    "d-rv": lambda v, Rs: Rs @ rotvec_to_matrix(v),
    "a-6d": lambda v, Rs: gs_decode(v),
    "d-6d": lambda v, Rs: gs_decode(sixd(Rs) + v),
}

# per-dim std over the dataset (scale-fair normalization)
stds = {}
for name, fn in REPS.items():
    vals = np.stack([fn(Rs, Rk) for Rs, Rks in samples for Rk in Rks])
    stds[name] = vals.std(axis=0) + 1e-9

# ---- A: chart roughness ----
print("\nA. chart roughness: normalized-units of target change per DEGREE of true rotation")
for name, fn in REPS.items():
    ratios = []
    for Rs, Rks in samples:
        for i in range(len(Rks) - 1):
            g = np.degrees(geo(Rks[i], Rks[i + 1]))
            if g < 0.05:
                continue
            d = np.linalg.norm((fn(Rs, Rks[i + 1]) - fn(Rs, Rks[i])) / stds[name])
            ratios.append(d / g)
    r = np.array(ratios)
    print(f"  {name}: median {np.median(r):.4f}  p99 {np.percentile(r, 99):.4f}  "
          f"max {r.max():.4f}  (max/median {r.max()/np.median(r):.1f}x)")

# ---- B: noise-robustness vs delta angle (Zhou-style curve) ----
print(f"\nB. decode error (deg) after adding eps={EPS} normalized-unit noise to the target,")
print("   binned by chunk-delta angle theta:")
bins = [(0, 30), (30, 60), (60, 90), (90, 120), (120, 180)]
hdr = "  rep   " + "".join(f"  {lo:>3}-{hi:<3}" for lo, hi in bins)
print(hdr)
for name, fn in REPS.items():
    errs = {b: [] for b in bins}
    for Rs, Rks in samples:
        Rk = Rks[-1]
        theta = np.degrees(geo(Rs, Rk))
        v = fn(Rs, Rk)
        for _ in range(4):
            noisy = v + rng.standard_normal(v.shape) * EPS * stds[name]
            errs_deg = np.degrees(geo(DECODE[name](noisy, Rs), Rk))
            for b in bins:
                if b[0] <= theta < b[1]:
                    errs[b].append(errs_deg)
    row = f"  {name} "
    for b in bins:
        e = np.array(errs[b])
        row += f"  {np.median(e):6.2f}" if len(e) else "      --"
    counts = "  (n=" + ",".join(str(len(errs[b])) for b in bins) + ")"
    print(row + counts)
print("\n(flat row across bins = well-conditioned at all angles our data visits;")
print(" rising row = the chart amplifies a fixed-size prediction error at large angles)")
print("DONE")
