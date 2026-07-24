#!/usr/bin/env python3
"""SYNCHRONOUS RTC validation client for the Qwen3-VL action-expert server (no real-time).

Purpose: prove the *model side* of RTC works before adding any real-time logic. The robot
pauses between segments -- motion is NOT smooth here, that's fine. What we're checking:

  1. The server HONORS the prefix: the new chunk's first `d` actions equal the prefix we
     sent (clamp). This passes even on a non-RTC checkpoint -- it's pure plumbing.
  2. Seam continuity: the new chunk's first action continues from where the previous
     segment left off (compare against a prefix-free request to see RTC's effect).
  3. [bound] prefix->postfix coherence INSIDE the new chunk -- the real "did RTC training
     work" signal. A checkpoint trained WITHOUT --rtc_prefix_max_length will jump here
     even though checks 1-2 pass.

IMPORTANT (delta actions): training uses delta joint actions, so the model's action space is
relative to the observation state and NOT portable across observations. The prefix is sent in
ABSOLUTE joint space -- a slice of the previously returned `actions` -- and the SERVER
re-expresses it against the current state (qwen_action_expert_server.py `action_prefix`).

Flow:
  - cold-start inference -> chunk A (50 absolute actions)
  - execute A[0:EXEC_HORIZON], then STOP
  - send A[EXEC:EXEC+d] (ABSOLUTE) as the prefix, request chunk B
  - verify B[0:d] == A[EXEC:EXEC+d] (clamp) and report seam/bound continuity
  - execute B[0:EXEC_HORIZON], repeat

Run against the normal server (qwen_action_expert_server.py) with an RTC-trained checkpoint
(--rtc_prefix_max_length > 0 at training). Set IMAGE_HISTORY / SUBTASK below to match the
server's mode (check GET /health).
"""

import io
import json
import sys
import time
from collections import deque

import cv2
import numpy as np
import requests
import torch

# Point Python at your LeRobot hardware fork (same as the reference client).
LEROBOT_FORK_PATH = "/home/iris/lerobot"
if LEROBOT_FORK_PATH and LEROBOT_FORK_PATH not in sys.path:
    sys.path.insert(0, LEROBOT_FORK_PATH)

from lerobot.common.robot_devices.robots.configs import TrossenAIRightArmOnlyRobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config

# ----------------------------- CONFIG -----------------------------
SERVER_URL = "http://10.79.12.149:8003/infer"   # <-- GPU node running qwen_action_expert_server.py
HL_CAM = "cam_high"
WRIST_CAM = "cam_right_wrist"
# --- these MUST match the server's checkpoint (see /health) ---
IMAGE_HISTORY = True            # True: send NUM_FRAMES strided cam_high frames; False: 1 current frame
SUBTASK = "pick up yellow block"  # subtask-input mode input; ignored by predict-subtask servers
NUM_FRAMES = 10                 # history length (history mode only)
STRIDE_TICKS = 10               # training stride: one cam_high history frame every 10 ticks
CONTROL_HZ = 30                 # rate at which we execute waypoints within a segment

EXEC_HORIZON = 10               # waypoints executed per segment (the "10 of 50")
DELAY_STEPS = 4                 # length of the prefix d (upcoming actions A[EXEC:EXEC+d])
MAX_SEGMENTS = 6                # stop after this many segments (it's a test, not a deployment)
STOP_PAUSE_S = 0.5              # visible pause between segments
COMPARE_NO_PREFIX = True        # also request a prefix-free chunk to show RTC changes the seam
NUM_FLOW_STEPS = 10
REQUEST_TIMEOUT_S = 60.0
JPEG_QUALITY = 90

# Clamp tolerance in ABSOLUTE joint space. The clamp round-trips through the delta+quantile
# normalization (float32), so expect ~1e-5, not 0.
CLAMP_TOL = 1e-2
# ------------------------------------------------------------------


def cam_rgb(obs, cam: str) -> np.ndarray:
    return obs[f"observation.images.{cam}"].detach().cpu().numpy().astype(np.uint8)


def encode_jpeg(rgb: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return enc.tobytes()


def request_chunk(session, cam_high_frames, wrist_rgb, state7, action_prefix=None, debug=False):
    """Blocking inference. `action_prefix` is (d, 7) ABSOLUTE joints. Returns (actions, response dict).
    `debug=True` also returns model_input_text -- the decoded token sequence the model saw."""
    files = []
    for i, frame in enumerate(cam_high_frames):  # oldest-first
        files.append(("cam_high", (f"f{i}.jpg", io.BytesIO(encode_jpeg(frame)), "image/jpeg")))
    files.append(("wrist", ("wrist.jpg", io.BytesIO(encode_jpeg(wrist_rgb)), "image/jpeg")))
    data = {
        "state": json.dumps([float(x) for x in np.asarray(state7).reshape(-1)]),
        "subtask": SUBTASK,
        "run_fast": "false",
        "num_flow_steps": str(NUM_FLOW_STEPS),
        "debug": "true" if debug else "false",
    }
    if action_prefix is not None and len(action_prefix) > 0:
        data["action_prefix"] = json.dumps(np.asarray(action_prefix, dtype=np.float32).tolist())
    resp = session.post(SERVER_URL, files=files, data=data, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    j = resp.json()
    return np.asarray(j["actions"], dtype=np.float32), j


def report_continuity(prev_raw, new_raw, cold_raw):
    """RTC correctness checks for one segment transition (all in absolute joint space)."""
    d, e = DELAY_STEPS, EXEC_HORIZON
    sent_prefix = prev_raw[e : e + d]

    # --- Check 1: did the server clamp the prefix? ---
    clamp_diff = float(np.max(np.abs(new_raw[:d] - sent_prefix)))
    clamp_ok = clamp_diff < CLAMP_TOL
    print(f"  [clamp] max|new[:{d}] - prefix|  = {clamp_diff:.2e}   "
          f"({'PASS' if clamp_ok else 'FAIL'}, tol {CLAMP_TOL:.0e})")

    # --- Check 2: seam continuity (what we execute) ---
    last_exec = prev_raw[e - 1]
    seam_jump = float(np.linalg.norm(new_raw[0] - last_exec))
    steps = np.linalg.norm(np.diff(prev_raw[:e], axis=0), axis=1)
    typical = float(np.mean(steps)) if len(steps) else float("nan")
    print(f"  [seam ] ||new[0] - last_executed|| = {seam_jump:.4f}   (typical in-chunk step = {typical:.4f})")
    if cold_raw is not None:
        cold_jump = float(np.linalg.norm(cold_raw[0] - last_exec))
        print(f"  [seam ] WITHOUT prefix (cold)      = {cold_jump:.4f}   <- expect LARGER than RTC seam")

    # --- Check 3: prefix -> postfix coherence INSIDE the new chunk (the real RTC signal) ---
    boundary = float(np.linalg.norm(new_raw[d] - new_raw[d - 1]))
    new_steps = np.linalg.norm(np.diff(new_raw[:e], axis=0), axis=1)
    new_typical = float(np.mean(new_steps)) if len(new_steps) else float("nan")
    print(f"  [bound] ||postfix[0] - prefix[-1]|| = {boundary:.4f}   (typical step {new_typical:.4f})")
    return clamp_ok


def main():
    print("Initializing Trossen robot and cameras...")
    robot_config = TrossenAIRightArmOnlyRobotConfig(
        max_relative_target=None,
        min_time_to_move_multiplier=4.0,
        camera_interface="opencv",
    )
    robot = make_robot_from_config(robot_config)
    robot.connect()
    print("Warming cameras...")
    robot.capture_observation()

    session = requests.Session()
    period = 1.0 / CONTROL_HZ
    all_clamps_ok = True

    # cam_high history primed with the first frame (mirrors training's clamped indices at t=0).
    obs = robot.capture_observation()
    history = deque([cam_rgb(obs, HL_CAM)] * NUM_FRAMES, maxlen=NUM_FRAMES)
    tick = 0

    def capture(with_append=True):
        """Capture obs; returns (cam_high frames to send, wrist, state)."""
        obs = robot.capture_observation()
        cur = cam_rgb(obs, HL_CAM)
        if with_append:
            history.append(cur)
        frames = list(history) if IMAGE_HISTORY else [cur]
        return frames, cam_rgb(obs, WRIST_CAM), obs["observation.state"].detach().cpu().numpy().astype(np.float32)

    try:
        print(f"\n[segment 0] cold-start inference (subtask={SUBTASK!r})...")
        frames, wrist, state7 = capture()
        plan_raw, resp = request_chunk(session, frames, wrist, state7, debug=True)
        print(f"  got chunk {plan_raw.shape} | subtask={resp.get('subtask')!r} | "
              f"server={resp.get('timing_ms', {}).get('total_ms')} ms")
        # Input sanity check: the exact prompt + full decoded model input the server built.
        print(f"\n=== prompt (server-side) ===\n{resp.get('prompt')}\n"
              f"=== full model input (decoded input_ids) ===\n{resp.get('model_input_text')}\n"
              f"============================\n")

        for segment in range(MAX_SEGMENTS):
            # 1. Execute EXEC_HORIZON waypoints of the current chunk.
            print(f"\n[segment {segment}] executing waypoints 0..{EXEC_HORIZON - 1}")
            for i in range(EXEC_HORIZON):
                t0 = time.perf_counter()
                robot.send_action(torch.from_numpy(np.asarray(plan_raw[i], dtype=np.float32)))
                if IMAGE_HISTORY and (tick + 1) % STRIDE_TICKS == 0:
                    history.append(cam_rgb(robot.capture_observation(), HL_CAM))
                tick += 1
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))

            # 2. STOP between segments (this is the "synchronous, not smooth" part).
            print(f"  STOP. pausing {STOP_PAUSE_S}s before replanning.")
            time.sleep(STOP_PAUSE_S)

            # 3. Send the ABSOLUTE upcoming d actions as the prefix; request the next chunk.
            prefix_abs = plan_raw[EXEC_HORIZON : EXEC_HORIZON + DELAY_STEPS]
            frames, wrist, state7 = capture()
            new_raw, resp = request_chunk(session, frames, wrist, state7, prefix_abs)
            cold_raw = request_chunk(session, frames, wrist, state7)[0] if COMPARE_NO_PREFIX else None

            # 4. Verify clamp + continuity.
            print(f"  [continuity check, segment {segment} -> {segment + 1}]  "
                  f"server={resp.get('timing_ms', {}).get('total_ms')} ms | prompt={resp.get('prompt')!r}")
            all_clamps_ok = report_continuity(plan_raw, new_raw, cold_raw) and all_clamps_ok

            # 5. Advance.
            plan_raw = new_raw

        print(f"\n==== DONE: clamp checks {'ALL PASSED' if all_clamps_ok else 'HAD FAILURES'} ====")

    except KeyboardInterrupt:
        print("\nStopping via Ctrl+C...")
    finally:
        print("Disconnecting robot devices...")
        robot.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()
