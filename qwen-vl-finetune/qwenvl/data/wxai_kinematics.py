#!/usr/bin/env python3
"""Offline forward/inverse kinematics for the Trossen WidowX AI (wxai_v0) arm.

Converts recorded joint angles <-> end-effector poses WITHOUT a connected robot, using
numpy only (runs in the `lerobot` conda env, which has no scipy).

The kinematic chain below is transcribed from the URDF that ships EMBEDDED inside the
trossen_arm 1.9.1 SDK (site-packages/trossen_arm/trossen_arm.cpython-*.so contains
`<robot name="wxai">...`; the on-robot controller uses this same model for its own
cartesian API). It was diffed joint-by-joint against Trossen's official
github.com/TrossenRobotics/trossen_arm_description urdf/generated/wxai/wxai_follower.urdf
and matches exactly (the official file only adds fixed camera-mount frames).

Pose convention -- identical to the SDK's driver.get_cartesian_positions():
    pose = [x, y, z, rx, ry, rz]
    * first 3: translation of the end-effector frame in the base frame, meters
    * last 3:  rotation of that frame as a rotation vector (angle-axis: axis * angle), radians
The "end-effector frame" is the TOOL frame: for every standard wxai_v0 gripper the SDK
uses t_flange_tool = [0.156062, 0, 0, 0, 0, 0], i.e. the URDF's `ee_gripper_link`,
0.156062 m straight out along the flange (link_6) x-axis -- the gripper finger midpoint.
Pass tool=FLANGE to get the flange (link_6) pose instead.

Joint conventions match what the LeRobot TrossenArmDriver records: q[0:6] are the arm
joints in radians (q[6], the gripper linear position in meters, is accepted and ignored).

Example -- convert a recorded LeRobot episode to EE poses:
    state = episode["observation.state"]        # (T, 7) or (T, 14) with right arm in [7:14]
    q = state[:, 7:13] if state.shape[1] == 14 else state[:, :6]
    poses = np.stack([fk_pose(qt) for qt in q])  # (T, 6) [x y z rx ry rz]
"""

import numpy as np

# joint origins (xyz, meters) and rotation axes, base_link -> link_6, from the wxai URDF.
# All joint-origin rpy values are zero, so the chain is fully described by these two tables.
_XYZ = np.array([
    [0.0,     0.0, 0.05725],   # joint_0 (waist)
    [0.02,    0.0, 0.04625],   # joint_1 (shoulder)
    [-0.264,  0.0, 0.0],       # joint_2 (elbow)
    [0.245,   0.0, 0.06],      # joint_3
    [0.06775, 0.0, 0.0455],    # joint_4
    [0.02895, 0.0, -0.0455],   # joint_5
])
_AXIS = np.array([
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, -1.0, 0.0],
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
])
JOINT_LIMITS = np.array([
    [-3.0543261909900767, 3.0543261909900767],
    [0.0, np.pi],
    [0.0, 2.356194490192345],
    [-np.pi / 2, np.pi / 2],
    [-np.pi / 2, np.pi / 2],
    [-np.pi, np.pi],
])

# Tool offsets relative to the flange (link_6), as [xyz, rotvec].
GRIPPER = np.array([0.156062, 0.0, 0.0, 0.0, 0.0, 0.0])  # SDK default t_flange_tool = ee_gripper_link
FLANGE = np.zeros(6)


def rotvec_to_matrix(rv: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, rad) -> 3x3 rotation matrix (Rodrigues)."""
    rv = np.asarray(rv, dtype=np.float64)
    angle = np.linalg.norm(rv)
    if angle < 1e-12:
        return np.eye(3)
    k = rv / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> rotation vector (axis * angle, rad)."""
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_a)
    if angle < 1e-12:
        return np.zeros(3)
    if np.pi - angle < 1e-6:  # near 180 deg: off-diagonal formula is degenerate
        # axis from the largest diagonal element of (R + I) / 2
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        i = int(np.argmax(axis))
        if axis[i] > 0:
            axis[(i + 1) % 3] = A[i, (i + 1) % 3] / axis[i] if axis[i] > 1e-12 else axis[(i + 1) % 3]
            axis[(i + 2) % 3] = A[i, (i + 2) % 3] / axis[i] if axis[i] > 1e-12 else axis[(i + 2) % 3]
        axis /= np.linalg.norm(axis)
        return axis * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(angle))
    return axis * angle


def _arm_q(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.shape[0] not in (6, 7):
        raise ValueError(f"expected 6 arm joints (or 7 with gripper), got {q.shape[0]}")
    return q[:6]


def fk_matrix(q, tool: np.ndarray = GRIPPER) -> np.ndarray:
    """Joint angles -> 4x4 homogeneous transform of the tool frame in the base frame."""
    q = _arm_q(q)
    T = np.eye(4)
    for i in range(6):
        step = np.eye(4)
        step[:3, 3] = _XYZ[i]
        step[:3, :3] = rotvec_to_matrix(_AXIS[i] * q[i])
        T = T @ step
    tool_T = np.eye(4)
    tool_T[:3, 3] = tool[:3]
    tool_T[:3, :3] = rotvec_to_matrix(tool[3:])
    return T @ tool_T


def fk_pose(q, tool: np.ndarray = GRIPPER) -> np.ndarray:
    """Joint angles -> [x, y, z, rx, ry, rz] pose (same convention as get_cartesian_positions)."""
    T = fk_matrix(q, tool)
    return np.concatenate([T[:3, 3], matrix_to_rotvec(T[:3, :3])])


def jacobian(q, tool: np.ndarray = GRIPPER) -> np.ndarray:
    """6x6 geometric Jacobian (linear on top, angular below) of the tool frame, base frame."""
    q = _arm_q(q)
    # forward pass: origin + axis of every joint in the base frame
    T = np.eye(4)
    origins, axes = [], []
    for i in range(6):
        step = np.eye(4)
        step[:3, 3] = _XYZ[i]
        T = T @ step
        origins.append(T[:3, 3].copy())
        axes.append(T[:3, :3] @ _AXIS[i])
        R = np.eye(4)
        R[:3, :3] = rotvec_to_matrix(_AXIS[i] * q[i])
        T = T @ R
    p_ee = fk_matrix(q, tool)[:3, 3]
    J = np.zeros((6, 6))
    for i in range(6):
        J[:3, i] = np.cross(axes[i], p_ee - origins[i])
        J[3:, i] = axes[i]
    return J


def ik(target_pose, q0, tool: np.ndarray = GRIPPER, max_iters: int = 200,
       tol: float = 1e-6, damping: float = 1e-4, clamp_limits: bool = True):
    """Damped-least-squares IK: [x y z rx ry rz] -> joint angles, seeded from q0.

    q0 should be a nearby configuration (e.g. the previous timestep) so the solver picks
    the intended branch. Returns (q, converged). Errors are position (m) + orientation
    (rad, rotvec of the residual rotation), so tol mixes both -- 1e-6 is tight for either.
    """
    target = np.asarray(target_pose, dtype=np.float64)
    R_target = rotvec_to_matrix(target[3:])
    q = _arm_q(q0).copy()
    for _ in range(max_iters):
        T = fk_matrix(q, tool)
        err = np.concatenate([target[:3] - T[:3, 3], matrix_to_rotvec(R_target @ T[:3, :3].T)])
        if np.linalg.norm(err) < tol:
            return q, True
        J = jacobian(q, tool)
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
        step = np.linalg.norm(dq)
        if step > 0.5:  # keep the linearization honest far from the target
            dq *= 0.5 / step
        q += dq
        if clamp_limits:
            q = np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
    T = fk_matrix(q, tool)
    err = np.concatenate([target[:3] - T[:3, 3], matrix_to_rotvec(R_target @ T[:3, :3].T)])
    return q, bool(np.linalg.norm(err) < tol)


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    zero = np.zeros(6)
    home = np.array([0, np.pi / 3, np.pi / 6, np.pi / 5, 0, 0])  # driver home_pose
    for name, q in [("sleep (all zero)", zero), ("home", home)]:
        print(f"{name:18s} q={q}  ->  pose={fk_pose(q)}")
