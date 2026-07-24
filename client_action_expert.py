#!/usr/bin/env python3
"""Robot-side client for the single Qwen3-VL action-expert server (no RTC).

Each cycle:
  1. capture the observation: the cam_high frame(s) + the current cam_right_wrist still + the
     7-dim right-arm joint state. IMAGE_HISTORY=True sends 10 strided cam_high frames (history
     video); IMAGE_HISTORY=False sends only the current cam_high frame,
  2. POST it to the server, which returns the action expert's absolute-joint action chunk. In
     predict-subtask mode the server GENERATES the subtask itself; in subtask-input mode it uses
     the SUBTASK string we send (hard-coded here). These MUST match the server's checkpoint --
     see /health,
  3. execute the first EXEC_HORIZON actions on the arm at CONTROL_HZ, one per tick.

This is the simple open-loop-chunking path (RTC intentionally omitted): grab observation ->
get actions -> execute -> repeat. During inference the arm holds its last commanded pose.
FAST is a server-side DEBUG signal the action expert ignores; leave RUN_FAST=False in
deployment.

Robot access matches client_rtc_qwen_reference_only.py: TrossenAIRightArmOnlyRobotConfig +
capture_observation()/send_action(), 7-dim right-arm state/action. Copy this file to the
ROBOT computer and run it there; the server runs on the GPU node.

  python client_action_expert.py            # set SERVER_URL below to the GPU node first
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
HL_CAM = "cam_high"            # history video camera (must match training / server)
WRIST_CAM = "cam_right_wrist"  # current-timestep wrist still
# --- these three MUST match how the server's checkpoint was trained (see /health) ---
IMAGE_HISTORY = True           # True: send NUM_FRAMES strided cam_high frames (history video).
                               # False: send only the 1 current cam_high frame (no-history model).
SUBTASK = "pick up yellow block"  # subtask-input mode: the language instruction we hard-code and
                               # send every cycle. MUST be one of the exact training strings:
                               # "pick up yellow block" / "pick up green block" / "waiting".
                               # IGNORED by the server in predict-subtask mode (it generates its own).
NUM_FRAMES = 10                # MUST equal the server's ds.num_frames (history mode only)
CONTROL_HZ = 30                # control cadence; one action executed per tick
STRIDE_TICKS = 10              # training stride: a cam_high frame every 10 ticks (~0.333 s @30Hz)
EXEC_HORIZON = 25              # actions executed per chunk before replanning (<= chunk length 50)
RUN_FAST = False               # server-side FAST debug signal; the action expert ignores it
NUM_FLOW_STEPS = 10            # flow-ODE Euler steps requested from the server
REQUEST_TIMEOUT_S = 30.0
JPEG_QUALITY = 90
# ------------------------------------------------------------------


def cam_rgb(obs, cam: str) -> np.ndarray:
    """HWC uint8 RGB frame from a LeRobot observation (LeRobot returns RGB)."""
    return obs[f"observation.images.{cam}"].detach().cpu().numpy().astype(np.uint8)


def encode_jpeg(rgb: np.ndarray) -> bytes:
    """RGB -> BGR -> JPEG bytes. cv2.imencode expects BGR and produces a standard JPEG that
    the server decodes back to correct RGB (matches the reference client's color handling)."""
    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return enc.tobytes()


def request_chunk(session: requests.Session, cam_high_frames: list, wrist_rgb: np.ndarray,
                  state7: np.ndarray, debug: bool = False):
    """POST the observation, return (actions[H,7] absolute joints, response dict).

    `cam_high_frames` is the oldest-first strided history (history mode) or a single-element
    list holding just the current frame (no-history mode). `debug=True` asks the server to also
    return `model_input_text` -- the decoded token sequence the model conditioned on."""
    files = []
    for i, frame in enumerate(cam_high_frames):  # oldest-first
        files.append(("cam_high", (f"f{i}.jpg", io.BytesIO(encode_jpeg(frame)), "image/jpeg")))
    files.append(("wrist", ("wrist.jpg", io.BytesIO(encode_jpeg(wrist_rgb)), "image/jpeg")))
    data = {
        "state": json.dumps([float(x) for x in np.asarray(state7).reshape(-1)]),
        "subtask": SUBTASK,   # ignored by the server in predict-subtask mode
        "run_fast": "true" if RUN_FAST else "false",
        "num_flow_steps": str(NUM_FLOW_STEPS),
        "debug": "true" if debug else "false",
    }
    resp = session.post(SERVER_URL, files=files, data=data, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    j = resp.json()
    actions = np.asarray(j["actions"], dtype=np.float32)  # (H, 7)
    return actions, j


def main():
    print("Initializing Trossen robot and cameras...")
    robot_config = TrossenAIRightArmOnlyRobotConfig(
        max_relative_target=None,
        min_time_to_move_multiplier=4.0,
        camera_interface="opencv",
    )
    robot = make_robot_from_config(robot_config)
    robot.connect()
    cameras = list(robot_config.cameras.keys())
    print(f"Connected. cameras: {cameras}")
    for need in (HL_CAM, WRIST_CAM):
        if need not in cameras:
            print(f"WARNING: required camera '{need}' not in {cameras}")

    # Warm the cameras once on the main thread (first async_read starts capture; ~2 s).
    print("Warming cameras (first frame can take ~2 s)...")
    robot.capture_observation()

    session = requests.Session()
    period = 1.0 / CONTROL_HZ
    exec_horizon = min(EXEC_HORIZON, 50)

    # Prime the strided cam_high history by duplicating the first frame (mirrors training's
    # clamping of negative indices at the episode start).
    obs = robot.capture_observation()
    history = deque([cam_rgb(obs, HL_CAM)] * NUM_FRAMES, maxlen=NUM_FRAMES)
    tick = 0

    print(f"Running (control {CONTROL_HZ} Hz, replan every {exec_horizon} steps, "
          f"image_history={IMAGE_HISTORY}, FAST={'on' if RUN_FAST else 'off'}). Ctrl+C to stop.\n")
    first_request = True
    try:
        while True:
            # --- observation at t: newest cam_high aligned with the wrist/state we send ---
            obs = robot.capture_observation()
            cur_high = cam_rgb(obs, HL_CAM)
            history.append(cur_high)
            wrist_rgb = cam_rgb(obs, WRIST_CAM)
            state7 = obs["observation.state"].detach().cpu().numpy().astype(np.float32)

            # history mode: send the strided window; no-history mode: just the current frame.
            cam_high_frames = list(history) if IMAGE_HISTORY else [cur_high]

            # --- inference (blocking; arm holds its last pose meanwhile) ---
            t0 = time.perf_counter()
            actions, resp = request_chunk(session, cam_high_frames, wrist_rgb, state7,
                                          debug=first_request)
            rtt_ms = (time.perf_counter() - t0) * 1e3
            timing = resp.get("timing_ms", {})
            if first_request:
                # Input sanity check: the exact prompt + the full decoded model input the
                # server templatized (vision pad tokens collapsed to <|image_pad|>xN).
                print(f"\n=== prompt (server-side) ===\n{resp.get('prompt')}\n"
                      f"=== full model input (decoded input_ids) ===\n{resp.get('model_input_text')}\n"
                      f"============================\n")
                first_request = False
            print(f"[{time.strftime('%X')}] subtask={resp.get('subtask')!r} | "
                  f"prompt={resp.get('prompt')!r} | chunk={actions.shape} | "
                  f"rtt={rtt_ms:.0f} ms | server_total={timing.get('total_ms')} ms "
                  f"(subtask={timing.get('subtask_ms')} expert={timing.get('expert_ms')})")

            # --- execute the chunk open-loop, one action per control tick ---
            n_exec = min(exec_horizon, len(actions))
            for i in range(n_exec):
                loop0 = time.perf_counter()
                robot.send_action(torch.from_numpy(np.asarray(actions[i], dtype=np.float32)))
                # keep the cam_high history strided during execution (history mode only)
                if IMAGE_HISTORY and (tick + 1) % STRIDE_TICKS == 0:
                    history.append(cam_rgb(robot.capture_observation(), HL_CAM))
                tick += 1
                time.sleep(max(0.0, period - (time.perf_counter() - loop0)))
    except KeyboardInterrupt:
        print("\nStopping via Ctrl+C...")
    finally:
        print("Disconnecting robot devices...")
        robot.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()
