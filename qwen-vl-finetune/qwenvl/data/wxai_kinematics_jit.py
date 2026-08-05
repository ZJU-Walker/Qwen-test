"""Numba-compiled port of the wxai DLS-IK chunk solver (serving hot path).

Bit-faithful transcription of wxai_kinematics.py fk/jacobian/ik + ee_repr's chunk loop
(sequential warm seeding, unclamped solve, widened-limit rejection, jaw passthrough) into
one nopython call, ~30-50x faster than the numpy/interpreter path. No fastmath: results
match the numpy reference to float64 rounding (verified by tests/test_ik_jit_equiv.py).

Differences from the numpy path: NONE in math. One in error handling -- a degenerate
predicted 6D rotation (zero/parallel columns) cannot raise from compiled code, so it is
reported as that step failing (hold previous command), which is also the intended serving
semantics. The numpy fallback in ee_repr mirrors this.

First call triggers JIT compilation (~2-4 s); the server warms it at startup.
"""

import numpy as np
from numba import njit

from qwenvl.data.wxai_kinematics import _AXIS, _XYZ, GRIPPER, JOINT_LIMITS

_TOOL_X = float(GRIPPER[0])
_LIMIT_LO = JOINT_LIMITS[:, 0].copy()
_LIMIT_HI = JOINT_LIMITS[:, 1].copy()
_XYZ_C = _XYZ.copy()
_AXIS_C = _AXIS.copy()


@njit(cache=True)
def _rotvec_to_matrix(rv, out):
    angle = np.sqrt(rv[0] * rv[0] + rv[1] * rv[1] + rv[2] * rv[2])
    if angle < 1e-12:
        for i in range(3):
            for j in range(3):
                out[i, j] = 1.0 if i == j else 0.0
        return
    kx, ky, kz = rv[0] / angle, rv[1] / angle, rv[2] / angle
    s, c = np.sin(angle), np.cos(angle)
    one_c = 1.0 - c
    out[0, 0] = c + kx * kx * one_c
    out[0, 1] = kx * ky * one_c - kz * s
    out[0, 2] = kx * kz * one_c + ky * s
    out[1, 0] = ky * kx * one_c + kz * s
    out[1, 1] = c + ky * ky * one_c
    out[1, 2] = ky * kz * one_c - kx * s
    out[2, 0] = kz * kx * one_c - ky * s
    out[2, 1] = kz * ky * one_c + kx * s
    out[2, 2] = c + kz * kz * one_c


@njit(cache=True)
def _matrix_to_rotvec(R, out):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    cos_a = (tr - 1.0) / 2.0
    if cos_a > 1.0:
        cos_a = 1.0
    elif cos_a < -1.0:
        cos_a = -1.0
    angle = np.arccos(cos_a)
    if angle < 1e-12:
        out[0] = 0.0; out[1] = 0.0; out[2] = 0.0
        return
    if np.pi - angle < 1e-6:
        # near 180 deg: axis from the largest diagonal element of A = (R + I) / 2,
        # off-diagonal terms A[i, j] = R[i, j] / 2 (bit-faithful to the numpy path).
        ax = np.empty(3)
        ax[0] = np.sqrt(max((R[0, 0] + 1.0) / 2.0, 0.0))
        ax[1] = np.sqrt(max((R[1, 1] + 1.0) / 2.0, 0.0))
        ax[2] = np.sqrt(max((R[2, 2] + 1.0) / 2.0, 0.0))
        i = 0
        if ax[1] > ax[0]:
            i = 1
        if ax[2] > ax[i]:
            i = 2
        if ax[i] > 1e-12:
            j, k = (i + 1) % 3, (i + 2) % 3
            ax[j] = (R[i, j] / 2.0) / ax[i]
            ax[k] = (R[i, k] / 2.0) / ax[i]
        n = np.sqrt(ax[0] * ax[0] + ax[1] * ax[1] + ax[2] * ax[2])
        out[0] = ax[0] / n * angle
        out[1] = ax[1] / n * angle
        out[2] = ax[2] / n * angle
        return
    d = 2.0 * np.sin(angle)
    out[0] = (R[2, 1] - R[1, 2]) / d * angle
    out[1] = (R[0, 2] - R[2, 0]) / d * angle
    out[2] = (R[1, 0] - R[0, 1]) / d * angle


@njit(cache=True)
def _fk_and_jacobian(q, xyz, axis, T_out, J_out, origins, axes, want_jacobian):
    """Forward chain -> tool pose T_out (4x4). If want_jacobian, also fill J_out (6x6)
    using per-joint origins/axes exactly like wxai_kinematics.jacobian."""
    T = np.eye(4)
    R_step = np.empty((3, 3))
    rv = np.empty(3)
    for i in range(6):
        # translate by joint origin
        px = T[0, 0] * xyz[i, 0] + T[0, 1] * xyz[i, 1] + T[0, 2] * xyz[i, 2] + T[0, 3]
        py = T[1, 0] * xyz[i, 0] + T[1, 1] * xyz[i, 1] + T[1, 2] * xyz[i, 2] + T[1, 3]
        pz = T[2, 0] * xyz[i, 0] + T[2, 1] * xyz[i, 1] + T[2, 2] * xyz[i, 2] + T[2, 3]
        T[0, 3], T[1, 3], T[2, 3] = px, py, pz
        if want_jacobian:
            origins[i, 0], origins[i, 1], origins[i, 2] = px, py, pz
            for r in range(3):
                axes[i, r] = T[r, 0] * axis[i, 0] + T[r, 1] * axis[i, 1] + T[r, 2] * axis[i, 2]
        # rotate about the joint axis
        rv[0] = axis[i, 0] * q[i]
        rv[1] = axis[i, 1] * q[i]
        rv[2] = axis[i, 2] * q[i]
        _rotvec_to_matrix(rv, R_step)
        for r in range(3):
            c0 = T[r, 0] * R_step[0, 0] + T[r, 1] * R_step[1, 0] + T[r, 2] * R_step[2, 0]
            c1 = T[r, 0] * R_step[0, 1] + T[r, 1] * R_step[1, 1] + T[r, 2] * R_step[2, 1]
            c2 = T[r, 0] * R_step[0, 2] + T[r, 1] * R_step[1, 2] + T[r, 2] * R_step[2, 2]
            T[r, 0], T[r, 1], T[r, 2] = c0, c1, c2
    # tool offset: straight out along the flange x axis (rotation part is identity)
    T[0, 3] += T[0, 0] * _TOOL_X
    T[1, 3] += T[1, 0] * _TOOL_X
    T[2, 3] += T[2, 0] * _TOOL_X
    for r in range(4):
        for c in range(4):
            T_out[r, c] = T[r, c]
    if want_jacobian:
        for i in range(6):
            dx = T[0, 3] - origins[i, 0]
            dy = T[1, 3] - origins[i, 1]
            dz = T[2, 3] - origins[i, 2]
            J_out[0, i] = axes[i, 1] * dz - axes[i, 2] * dy
            J_out[1, i] = axes[i, 2] * dx - axes[i, 0] * dz
            J_out[2, i] = axes[i, 0] * dy - axes[i, 1] * dx
            J_out[3, i] = axes[i, 0]
            J_out[4, i] = axes[i, 1]
            J_out[5, i] = axes[i, 2]


@njit(cache=True)
def _dls_solve(target_pos, R_target, q0, max_iters, tol, damping):
    """Unclamped damped-least-squares IK; returns (q, converged)."""
    q = q0.copy()
    T = np.empty((4, 4))
    J = np.empty((6, 6))
    origins = np.empty((6, 3))
    axes = np.empty((6, 3))
    err = np.empty(6)
    R_res = np.empty((3, 3))
    rv = np.empty(3)
    for _ in range(max_iters):
        _fk_and_jacobian(q, _XYZ_C, _AXIS_C, T, J, origins, axes, True)
        for r in range(3):
            err[r] = target_pos[r] - T[r, 3]
        # residual rotation R_target @ T[:3,:3].T -> rotvec
        for r in range(3):
            for c in range(3):
                R_res[r, c] = (R_target[r, 0] * T[c, 0] + R_target[r, 1] * T[c, 1]
                               + R_target[r, 2] * T[c, 2])
        _matrix_to_rotvec(R_res, rv)
        err[3], err[4], err[5] = rv[0], rv[1], rv[2]
        n = 0.0
        for r in range(6):
            n += err[r] * err[r]
        if np.sqrt(n) < tol:
            return q, True
        A = J @ J.T
        for r in range(6):
            A[r, r] += damping
        x = np.linalg.solve(A, err)
        dq = J.T @ x
        sn = 0.0
        for r in range(6):
            sn += dq[r] * dq[r]
        sn = np.sqrt(sn)
        if sn > 0.5:
            for r in range(6):
                dq[r] *= 0.5 / sn
        for r in range(6):
            q[r] += dq[r]
    _fk_and_jacobian(q, _XYZ_C, _AXIS_C, T, J, origins, axes, False)
    for r in range(3):
        err[r] = target_pos[r] - T[r, 3]
    for r in range(3):
        for c in range(3):
            R_res[r, c] = (R_target[r, 0] * T[c, 0] + R_target[r, 1] * T[c, 1]
                           + R_target[r, 2] * T[c, 2])
    _matrix_to_rotvec(R_res, rv)
    err[3], err[4], err[5] = rv[0], rv[1], rv[2]
    n = 0.0
    for r in range(6):
        n += err[r] * err[r]
    return q, bool(np.sqrt(n) < tol)


@njit(cache=True)
def ee_chunk_to_joints_jit(chunk, q_current, max_iters, limit_margin):
    """(H, 10) EE chunk -> ((H, 7) float32 joints, num_unconverged). Mirrors
    ee_repr.ee_chunk_to_joints: Gram-Schmidt decode, sequential warm seeding,
    unclamped DLS, widened-limit rejection, hold-previous on any failure."""
    H = chunk.shape[0]
    out = np.empty((H, 7), dtype=np.float32)
    seed = q_current[:6].copy()
    prev = np.empty(7, dtype=np.float32)
    for r in range(6):
        prev[r] = np.float32(seed[r])
    prev[6] = np.float32(chunk[0, 9])
    bad = 0
    R = np.empty((3, 3))
    for i in range(H):
        ok = True
        # Gram-Schmidt decode of the 6D block (columns of R); degenerate -> step fails
        a1x, a1y, a1z = chunk[i, 3], chunk[i, 4], chunk[i, 5]
        a2x, a2y, a2z = chunk[i, 6], chunk[i, 7], chunk[i, 8]
        n1 = np.sqrt(a1x * a1x + a1y * a1y + a1z * a1z)
        if n1 < 1e-8:
            ok = False
        else:
            b1x, b1y, b1z = a1x / n1, a1y / n1, a1z / n1
            d = b1x * a2x + b1y * a2y + b1z * a2z
            p2x, p2y, p2z = a2x - d * b1x, a2y - d * b1y, a2z - d * b1z
            n2 = np.sqrt(p2x * p2x + p2y * p2y + p2z * p2z)
            if n2 < 1e-8:
                ok = False
            else:
                b2x, b2y, b2z = p2x / n2, p2y / n2, p2z / n2
                R[0, 0], R[1, 0], R[2, 0] = b1x, b1y, b1z
                R[0, 1], R[1, 1], R[2, 1] = b2x, b2y, b2z
                R[0, 2] = b1y * b2z - b1z * b2y
                R[1, 2] = b1z * b2x - b1x * b2z
                R[2, 2] = b1x * b2y - b1y * b2x
        if ok:
            q, ok = _dls_solve(chunk[i, :3].copy(), R, seed, max_iters, 1e-6, 1e-4)
            if ok:
                for r in range(6):
                    if q[r] < _LIMIT_LO[r] - limit_margin or q[r] > _LIMIT_HI[r] + limit_margin:
                        ok = False
                        break
        if ok:
            for r in range(6):
                out[i, r] = np.float32(q[r])
            out[i, 6] = np.float32(chunk[i, 9])
            for r in range(6):
                seed[r] = q[r]
            for r in range(7):
                prev[r] = out[i, r]
        else:
            bad += 1
            for r in range(7):
                out[i, r] = prev[r]
    return out, bad
