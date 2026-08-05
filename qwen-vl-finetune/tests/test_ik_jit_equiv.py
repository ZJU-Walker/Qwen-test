"""Equivalence + speed gates for the numba IK chunk solver vs the numpy reference.

Gates:
  1. On real chunks from BOTH datasets: identical convergence flags/failure counts and
     joint outputs within float64 round-off of the numpy path.
  2. Degenerate 6D rows hold (no raise) in both paths, same outputs.
  3. Speed: jit chunk median must beat numpy by >=10x.
"""

import glob
import statistics
import time

import numpy as np
import pandas as pd

import qwenvl.data.ee_repr as ee_repr
from qwenvl.data.ee_repr import joints_to_ee
from qwenvl.data.wxai_kinematics_jit import ee_chunk_to_joints_jit

assert ee_repr._JIT_CHUNK is not None, "numba path not active"


def numpy_ref(chunk, q0, max_iters=60):
    saved = ee_repr._JIT_CHUNK
    ee_repr._JIT_CHUNK = None
    try:
        return ee_repr.ee_chunk_to_joints(chunk, q0, max_iters=max_iters)
    finally:
        ee_repr._JIT_CHUNK = saved


DIRS = ["/iris/projects/humanoid/trossen_data/0717_green_yellow_block_mem_merged",
        "/iris/projects/humanoid/trossen_data/0731_green_yellow_merged"]
parquets = sorted(p for d in DIRS for p in glob.glob(f"{d}/data/chunk-*/episode_*.parquet"))[::7][:40]

max_diff, n_chunks, t_np, t_jit = 0.0, 0, [], []
for pq in parquets:
    df = pd.read_parquet(pq, columns=["observation.state", "action"])
    act = np.stack(df["action"].to_numpy())[:, 7:14]
    st = np.stack(df["observation.state"].to_numpy())[:, 7:14]
    if len(act) < 60:
        continue
    for frac in (0.15, 0.45, 0.75):
        t0 = int(len(act) * frac)
        chunk = joints_to_ee(act[t0:t0 + 50]).astype(np.float64)
        q0 = st[t0].astype(np.float64)

        s = time.perf_counter(); out_np, bad_np = numpy_ref(chunk, q0); t_np.append(time.perf_counter() - s)
        s = time.perf_counter(); out_jit, bad_jit = ee_chunk_to_joints_jit(
            np.ascontiguousarray(chunk), np.ascontiguousarray(q0[:6]), 60, ee_repr._LIMIT_MARGIN)
        t_jit.append(time.perf_counter() - s)

        assert bad_np == bad_jit, f"{pq} frac={frac}: failures {bad_np} vs {bad_jit}"
        d = float(np.abs(out_np.astype(np.float64) - out_jit.astype(np.float64)).max())
        max_diff = max(max_diff, d)
        n_chunks += 1

assert n_chunks >= 60, f"only {n_chunks} chunks tested"
# Outputs are float32; compiled float64 arithmetic associates ops differently than
# numpy/BLAS, so allow a few float32 ULPs (1 ULP ~ 1.2e-7 near 1.0). Robot joint
# resolution is ~1e-4 rad -- three orders of magnitude coarser than this gate.
assert max_diff < 1e-6, f"outputs diverge: max diff {max_diff:.3e}"
print(f"1. equivalence: {n_chunks} real chunks, identical failure counts, max |diff| {max_diff:.2e}  OK")

# degenerate rows: zero first column + parallel columns must hold, not raise, in both paths
chunk = joints_to_ee(np.tile(np.array([0, 1.0, 0.5, 0.3, 0, 0, 0.02]), (10, 1))).astype(np.float64)
chunk[3, 3:9] = 0.0
chunk[7, 3:6] = chunk[7, 6:9] = np.array([0.5, 0.5, 0.0])
q0 = np.array([0, 1.0, 0.5, 0.3, 0, 0, 0.02])
out_np, bad_np = numpy_ref(chunk, q0)
out_jit, bad_jit = ee_chunk_to_joints_jit(np.ascontiguousarray(chunk),
                                          np.ascontiguousarray(q0[:6]), 60, ee_repr._LIMIT_MARGIN)
assert bad_np == bad_jit == 2, f"degenerate rows: {bad_np} vs {bad_jit} (want 2)"
assert float(np.abs(out_np - out_jit).max()) < 1e-9
assert np.allclose(out_np[3], out_np[2]) and np.allclose(out_np[7], out_np[6]), "hold semantics broken"
print(f"2. degenerate 6D rows: both paths hold previous command (2 held), no raise  OK")

mn, mj = statistics.median(t_np) * 1e3, statistics.median(t_jit) * 1e3
print(f"3. speed: numpy {mn:.1f} ms -> jit {mj:.2f} ms per 50-step chunk ({mn / mj:.0f}x)")
assert mn / mj >= 10, "jit speedup below 10x"

# worst case: fully unreachable chunk at the serving cap
bad_chunk = chunk.copy(); bad_chunk[:, 0] += 1.0
s = time.perf_counter()
ee_chunk_to_joints_jit(np.ascontiguousarray(bad_chunk), np.ascontiguousarray(q0[:6]), 60, ee_repr._LIMIT_MARGIN)
print(f"   worst-case unreachable 10-row chunk @ cap 60: {(time.perf_counter() - s) * 1e3:.1f} ms")

print("\nALL IK JIT GATES PASS")
