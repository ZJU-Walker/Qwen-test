"""End-effector action/state representation for the Trossen wxai right arm.

Representation (action_space="ee6d"), chosen 2026-08-02 after measuring the real data
(tests/probe_rotation_representation.py, tests/probe_rotation_smoothness.py):

    [x, y, z,  r11 r21 r31 r12 r22 r32,  jaw]   -- 10 dims

* xyz: tool-frame position in the base frame, meters (SDK get_cartesian_positions frame).
* 6 rotation dims: the first two COLUMNS of the rotation matrix, column-major
  (Zhou et al., arXiv:1812.07035 -- the continuous "6D" representation Diffusion Policy
  uses). Decoded back to a matrix by Gram-Schmidt, which also projects away prediction
  noise (measured: the most noise-robust delta representation on our chunks).
* jaw: passthrough of the raw gripper dim.

Delta convention: per-dim NAIVE subtraction vs the current state on dims 0..8 (position
AND the 6D dims), jaw absolute -- i.e. delta_mask "9,-1". Measured valid: naive 6D
subtraction decoded via state-add-back + Gram-Schmidt beats proper rotvec composition on
noise robustness at every delta angle our data visits (flat Zhou-style curves), while
naive ROTVEC subtraction is off by ~7 deg median and is NOT used anywhere.

Serving: the model emits EE chunks; ee_chunk_to_joints() converts to joint commands via
Gram-Schmidt + damped-least-squares IK, each step seeded from the previous solution so
the solver stays on the executed branch. The robot-facing interface stays joint-space.
"""

import os
from typing import Tuple

import numpy as np

from qwenvl.data.wxai_kinematics import JOINT_LIMITS, fk_matrix, ik

# Compiled chunk solver (numba): same math ~30-50x faster; QWEN_IK_NUMBA=0 forces the
# numpy path (and numba simply not being installed degrades gracefully to it too).
_JIT_CHUNK = None
if os.environ.get("QWEN_IK_NUMBA", "1") != "0":
    try:
        from qwenvl.data.wxai_kinematics_jit import ee_chunk_to_joints_jit as _JIT_CHUNK
    except ImportError:
        _JIT_CHUNK = None

EE_DIM = 10
_ARM = slice(0, 6)   # arm joints within the 7-dim right-arm selection
_JAW = 6
# Reject IK solutions beyond URDF limits + this margin (rad). The margin absorbs the
# measured firmware-vs-URDF limit slack (<=0.015 rad) with room to spare while still
# catching branch-flip/runaway solutions, which sit far outside it.
_LIMIT_MARGIN = 0.1


def sixd_from_matrix(R: np.ndarray) -> np.ndarray:
    """First two columns of R, column-major: [r11 r21 r31 r12 r22 r32]."""
    return R[:, :2].T.reshape(6).copy()


def gs_decode(v: np.ndarray) -> np.ndarray:
    """6D vector -> rotation matrix via Gram-Schmidt (Zhou et al. eq. 15).

    Well-defined for any v whose first column is nonzero and not parallel to the
    second; prediction noise is projected onto SO(3) rather than propagated."""
    a1, a2 = v[:3].astype(np.float64), v[3:6].astype(np.float64)
    n1 = np.linalg.norm(a1)
    if n1 < 1e-8:
        raise ValueError("degenerate 6D rotation: first column ~ 0")
    b1 = a1 / n1
    a2p = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(a2p)
    if n2 < 1e-8:
        raise ValueError("degenerate 6D rotation: columns are parallel")
    b2 = a2p / n2
    return np.stack([b1, b2, np.cross(b1, b2)], axis=1)


def joints_to_ee(joints: np.ndarray) -> np.ndarray:
    """(..., 7) right-arm joints [q0..q5, jaw] -> (..., 10) EE representation."""
    joints = np.asarray(joints, dtype=np.float64)
    flat = joints.reshape(-1, joints.shape[-1])
    if flat.shape[-1] != 7:
        raise ValueError(f"expected 7 joint dims (6 arm + jaw), got {flat.shape[-1]}")
    out = np.empty((flat.shape[0], EE_DIM), dtype=np.float32)
    for i, q in enumerate(flat):
        T = fk_matrix(q[_ARM])
        out[i, :3] = T[:3, 3]
        out[i, 3:9] = sixd_from_matrix(T[:3, :3])
        out[i, 9] = q[_JAW]
    return out.reshape(*joints.shape[:-1], EE_DIM)


def ee_to_joints(ee: np.ndarray, q_seed: np.ndarray, max_iters: int = 200) -> Tuple[np.ndarray, bool]:
    """One EE vector (10,) -> right-arm joints (7,), seeded from q_seed (7 or 6).

    Returns (joints, converged). Non-convergence leaves the caller to decide
    (serving holds the previous verified command rather than executing a bad solve).
    max_iters: DLS budget per solve. Warm-seeded reachable solves converge in <10;
    serving caps this (default 60 there) so an unreachable-pose chunk cannot stall a
    request for seconds -- exhausting the budget just means (ok=False -> hold)."""
    ee = np.asarray(ee, dtype=np.float64).reshape(EE_DIM)
    try:
        R = gs_decode(ee[3:9])
    except ValueError:
        # Degenerate predicted 6D rotation: report as a failed solve (caller holds the
        # previous command) instead of raising -- one garbage row must not kill a request.
        return np.concatenate([np.asarray(q_seed, dtype=np.float64).reshape(-1)[:6],
                               [ee[9]]]).astype(np.float32), False
    # ik() takes [xyz, rotvec]; build the rotvec from the decoded matrix.
    from qwenvl.data.wxai_kinematics import matrix_to_rotvec

    target = np.concatenate([ee[:3], matrix_to_rotvec(R)])
    # clamp_limits=False: the REAL robot exceeds the URDF limits by up to ~0.015 rad on
    # joints 2/4 (measured over the datasets, ~1% of frames), so limit-clamped IK can
    # never reach those recorded poses. Warm seeding keeps the unclamped solver on the
    # executed branch; the widened-limit check below still rejects runaway solutions.
    q, ok = ik(target, np.asarray(q_seed, dtype=np.float64).reshape(-1)[:6],
               max_iters=max_iters, clamp_limits=False)
    if ok and ((q < JOINT_LIMITS[:, 0] - _LIMIT_MARGIN) | (q > JOINT_LIMITS[:, 1] + _LIMIT_MARGIN)).any():
        ok = False
    return np.concatenate([q, [ee[9]]]).astype(np.float32), ok


def ee_chunk_to_joints(chunk: np.ndarray, q_current: np.ndarray,
                       max_iters: int = 200) -> Tuple[np.ndarray, int]:
    """(H, 10) EE chunk -> (H, 7) joint commands, IK seeded sequentially from
    q_current (the measured joints at capture). Returns (joints, num_unconverged);
    an unconverged step reuses the previous step's joints (hold, never jump)."""
    chunk = np.asarray(chunk, dtype=np.float64)
    seed = np.asarray(q_current, dtype=np.float64).reshape(-1)[:6]
    if _JIT_CHUNK is not None:
        return _JIT_CHUNK(np.ascontiguousarray(chunk), np.ascontiguousarray(seed),
                          max_iters, _LIMIT_MARGIN)
    out = np.empty((chunk.shape[0], 7), dtype=np.float32)
    prev = np.concatenate([seed, [chunk[0, 9]]]).astype(np.float32)
    bad = 0
    for i, ee in enumerate(chunk):
        q, ok = ee_to_joints(ee, seed, max_iters=max_iters)
        if ok:
            out[i] = q
            seed = q[:6].astype(np.float64)
            prev = q
        else:
            bad += 1
            out[i] = prev
    return out, bad
