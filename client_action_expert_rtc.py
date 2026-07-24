#!/usr/bin/env python3
"""Asynchronous real-time-chunking (RTC) client for the Qwen3-VL action-expert server.

How this differs from the plain client (client_action_expert.py):
  - The plain client requests a chunk, executes ~25 actions, then PAUSES (arm holds) while it
    requests the next chunk. Each chunk is generated from scratch, so the seam can jerk.
  - This client runs inference in a BACKGROUND thread while it keeps executing the current
    chunk, and tells the server which upcoming actions it is committing to during that
    inference (the "prefix"). The server (with an RTC-trained checkpoint, i.e. trained with
    --rtc_prefix_max_length > 0) hard-clamps those leading actions, so the new chunk continues
    smoothly from where the robot already is -- reactive AND smooth, with no pause.

The control loop runs at CONTROL_HZ; every tick executes one action. When a replan is due, it
captures the observation (cheap -- LeRobot async_read returns the latest buffered frame) and
hands JPEG-encoding + the HTTP request to a worker thread, sending the DELAY_STEPS actions it
will execute during the wait as the prefix. When the result arrives (~d ticks later) its first
d actions already match what was executed, so it splices in seamlessly.

Key timing requirement: DELAY_STEPS must comfortably exceed the real server latency in control
ticks (latency_s * CONTROL_HZ) -- with the Qwen server that's ~0.6-0.8 s expert-only, so ~20-25
ticks at 30 Hz -- and be <= the training --rtc_prefix_max_length. If inference exceeds
DELAY_STEPS ticks the client STOPS-AND-WAITS (holds the last committed action) rather than
running past the prefix; watch for the "holding" message and raise DELAY_STEPS if it recurs.
NOTE: predict-subtask servers add ~0.3-0.9 s of subtask generation per request -- usually too
slow for RTC at 30 Hz. Deploy RTC with a subtask-input checkpoint (--subtask_input server).

Delta-action gotcha: training uses delta joint actions, so the model's action space is relative
to the observation state and NOT portable across observations. Prefixes are therefore sent in
ABSOLUTE joint space (a slice of the previously returned actions); the server re-expresses them
against the current state before clamping.

Validate the model side FIRST with client_action_expert_rtc_sync.py (clamp / seam / [bound]).
Copy this file to the ROBOT computer; the server runs on the GPU node.
"""

import io
import json
import select
import sys
import termios
import time
import tty
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

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
CONTROL_HZ = 30                 # control cadence; one action executed per tick

# --- RTC SCHEDULING ---
# DELAY_STEPS (d): control ticks of inference latency we plan around. Must be >= the real
# latency in ticks WITH MARGIN, and <= the training --rtc_prefix_max_length.
DELAY_STEPS = 25
# EXEC_HORIZON (s): ticks between successive replans. Smaller = more reactive.
# Requires EXEC_HORIZON >= DELAY_STEPS and EXEC_HORIZON + DELAY_STEPS <= chunk length (50).
EXEC_HORIZON = 25
NUM_FLOW_STEPS = 10             # drop to 5 (as in the paper) if you need to shave latency
REQUEST_TIMEOUT_S = 60.0
JPEG_QUALITY = 90
LOG_TIMING = True
# ------------------------------------------------------------------


@dataclass
class Plan:
    """A chunk currently being executed, anchored to an absolute control-loop step."""

    start_step: int      # global step that this chunk's slot 0 corresponds to
    raw: np.ndarray      # (H, 7) ABSOLUTE joint actions (what we send to the arm)


@dataclass
class ChunkResult:
    raw: np.ndarray
    subtask: str | None
    prompt: str | None            # the exact prompt string the server templatized
    model_input_text: str | None  # debug only: decoded input_ids the model conditioned on
    round_trip_ms: float
    server_total_ms: float | None


@dataclass
class PendingRequest:
    """An in-flight RTC request anchored to an absolute control-loop step."""

    future: Future
    anchor: int
    launch_step: int
    launch_time: float
    prefix: np.ndarray = field(repr=False)


class NonBlockingKeyReader:
    """Reads a single keystroke from stdin without blocking execution."""

    def __init__(self):
        self._fd = sys.stdin.fileno()
        self._old_settings = None

    def __enter__(self):
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def get_key(self) -> str | None:
        r, _, _ = select.select([sys.stdin], [], [], 0)
        if r:
            return sys.stdin.read(1).lower()
        return None


def cam_rgb(obs, cam: str) -> np.ndarray:
    return obs[f"observation.images.{cam}"].detach().cpu().numpy().astype(np.uint8)


def encode_jpeg(rgb: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return enc.tobytes()


def post_request(session, cam_high_frames, wrist_rgb, state7, action_prefix, debug=False) -> ChunkResult:
    """Encode + POST one observation. Runs on the WORKER thread (JPEG encoding of up to 10
    frames + the HTTP round trip stay out of the control loop). `action_prefix` is (d, 7)
    ABSOLUTE joints or None (cold start). `debug=True` asks the server to also return the
    decoded token sequence the model conditioned on (input sanity check)."""
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
    t0 = time.perf_counter()
    resp = session.post(SERVER_URL, files=files, data=data, timeout=REQUEST_TIMEOUT_S)
    resp.raise_for_status()
    j = resp.json()
    rtt_ms = (time.perf_counter() - t0) * 1e3
    timing = j.get("timing_ms", {})
    return ChunkResult(
        raw=np.asarray(j["actions"], dtype=np.float32),
        subtask=j.get("subtask"),
        prompt=j.get("prompt"),
        model_input_text=j.get("model_input_text"),
        round_trip_ms=rtt_ms,
        server_total_ms=timing.get("total_ms"),
    )


def extract_prefix(plan: Plan, anchor: int, delay_steps: int) -> np.ndarray:
    """ABSOLUTE prefix of exactly (delay_steps, 7) starting at absolute step `anchor` -- the
    actions the control loop will execute while inference runs. Clamped to the plan and padded
    by repeating the last action (the scheduler's invariants mean padding should never trigger;
    repeat-last is the safe hold behavior if it ever does)."""
    p0 = max(0, min(anchor - plan.start_step, len(plan.raw)))
    chunk = plan.raw[p0 : p0 + delay_steps]
    if len(chunk) < delay_steps:
        print(f"  WARNING: prefix window ran off the plan ({len(chunk)}/{delay_steps} rows); padding.")
        last = chunk[-1:] if len(chunk) else plan.raw[-1:]
        chunk = np.concatenate([chunk, np.repeat(last, delay_steps - len(chunk), axis=0)], axis=0)
    return chunk


def main():
    print("Initializing Trossen robot and cameras...")
    robot_config = TrossenAIRightArmOnlyRobotConfig(
        max_relative_target=None,
        min_time_to_move_multiplier=4.0,
        camera_interface="opencv",
    )
    robot = make_robot_from_config(robot_config)
    robot.connect()
    print("Warming cameras (first frame can take ~2 s)...")
    robot.capture_observation()

    if EXEC_HORIZON < DELAY_STEPS:
        raise ValueError("EXEC_HORIZON must be >= DELAY_STEPS for the RTC pipeline to stay ahead.")

    session = requests.Session()
    executor = ThreadPoolExecutor(max_workers=1)
    period = 1.0 / CONTROL_HZ

    # cam_high history primed with the first frame (mirrors training's clamped indices at t=0).
    obs = robot.capture_observation()
    history = deque([cam_rgb(obs, HL_CAM)] * NUM_FRAMES, maxlen=NUM_FRAMES)
    tick = 0

    def capture():
        """Capture the observation on the CONTROL thread (async_read is a few ms; keeping all
        camera access on one thread avoids cross-thread capture). Appends the current cam_high
        frame to the history, as the plain client does at each request."""
        obs = robot.capture_observation()
        cur = cam_rgb(obs, HL_CAM)
        history.append(cur)
        frames = list(history) if IMAGE_HISTORY else [cur]
        return frames, cam_rgb(obs, WRIST_CAM), obs["observation.state"].detach().cpu().numpy().astype(np.float32)

    # --- Cold start: one synchronous, prefix-free inference (also warms the server). ---
    print(f"Cold-start inference (no prefix, subtask={SUBTASK!r})...")
    frames, wrist, state7 = capture()
    result0 = post_request(session, frames, wrist, state7, None, debug=True)
    print(f"  cold-start: chunk={result0.raw.shape} | subtask={result0.subtask!r} | "
          f"round_trip={result0.round_trip_ms:.0f} ms | server={result0.server_total_ms} ms")
    # Input sanity check: the exact prompt + the full decoded model input the server
    # templatized (vision pad tokens collapsed to <|image_pad|>xN). Verify the task/question
    # text, the discretized state, and the subtask turn look right BEFORE the loop starts.
    print(f"\n=== prompt (server-side) ===\n{result0.prompt}\n"
          f"=== full model input (decoded input_ids) ===\n{result0.model_input_text}\n"
          f"============================\n")
    horizon = len(result0.raw)
    if EXEC_HORIZON + DELAY_STEPS > horizon:
        raise ValueError(f"EXEC_HORIZON + DELAY_STEPS ({EXEC_HORIZON + DELAY_STEPS}) exceeds chunk length ({horizon}).")
    plan = Plan(start_step=0, raw=result0.raw)

    # --- Warm the prefix path once end-to-end (validates the server accepts action_prefix and
    #     surfaces a checkpoint/latency problem BEFORE the real-time loop). ---
    print("Warming the prefix path...")
    warm_prefix = extract_prefix(plan, EXEC_HORIZON, DELAY_STEPS)
    frames, wrist, state7 = capture()
    warm = post_request(session, frames, wrist, state7, warm_prefix)
    clamp = float(np.max(np.abs(warm.raw[:DELAY_STEPS] - warm_prefix)))
    est_ticks = warm.round_trip_ms / 1e3 * CONTROL_HZ
    print(f"  prefix warmup: round_trip={warm.round_trip_ms:.0f} ms (~{est_ticks:.0f} ticks) | clamp={clamp:.1e}")
    if est_ticks > DELAY_STEPS:
        print(f"  WARNING: measured latency (~{est_ticks:.0f} ticks) exceeds DELAY_STEPS={DELAY_STEPS}; "
              "the loop will stop-and-wait every chunk. Raise DELAY_STEPS (and retrain if it "
              "exceeds --rtc_prefix_max_length).")

    global_step = 0
    next_request_step = EXEC_HORIZON  # when to fire the first async replan
    pending: PendingRequest | None = None
    wait_ticks = 0
    wait_notice_every = max(1, CONTROL_HZ)
    overrun_notice_every = max(1, CONTROL_HZ)

    print(f"\nRTC running at {CONTROL_HZ} Hz | d={DELAY_STEPS} | replan every {EXEC_HORIZON} steps | "
          f"image_history={IMAGE_HISTORY}")
    print("Press [q] to quit.\n")

    try:
        with NonBlockingKeyReader() as key_reader:
            while True:
                loop_start = time.perf_counter()

                if key_reader.get_key() == "q":
                    print("\nQuit key 'q' pressed.")
                    break

                # 1. Install a finished async inference. Its first DELAY_STEPS actions are
                #    clamped to the prefix we committed, so it splices in seamlessly.
                if pending is not None and pending.future.done():
                    try:
                        result = pending.future.result()
                        actual_ticks = global_step - pending.launch_step
                        clamp_diff = float(np.max(np.abs(result.raw[:DELAY_STEPS] - pending.prefix)))
                        plan = Plan(start_step=pending.anchor, raw=result.raw)
                        if LOG_TIMING:
                            print(f"  chunk ready: ticks={actual_ticks} | clamp={clamp_diff:.1e} | "
                                  f"round_trip={result.round_trip_ms:.0f} ms | server={result.server_total_ms} ms | "
                                  f"subtask={result.subtask!r} | prompt={result.prompt!r}")
                        if wait_ticks:
                            print(f"  chunk arrived after holding {wait_ticks} tick(s); resuming.")
                    except Exception as e:  # noqa: BLE001
                        print(f"  Inference request failed, keeping current plan: {e}")
                    pending = None
                    wait_ticks = 0

                # 2. STOP-AND-WAIT: we committed to exactly DELAY_STEPS actions as the prefix.
                #    If the new chunk hasn't arrived once they're executed, HOLD the last
                #    committed action (freeze global_step) until it lands -- a brief pause in
                #    exchange for guaranteed continuity (no jump past the prefix).
                if pending is not None and global_step >= pending.anchor + DELAY_STEPS:
                    hold_idx = (pending.anchor + DELAY_STEPS - 1) - plan.start_step
                    hold_idx = min(max(hold_idx, 0), len(plan.raw) - 1)
                    robot.send_action(torch.from_numpy(np.asarray(plan.raw[hold_idx], dtype=np.float32)))
                    if wait_ticks % wait_notice_every == 0:
                        print(f"  holding for late chunk (inference > DELAY_STEPS={DELAY_STEPS} ticks)...")
                    wait_ticks += 1
                    time.sleep(max(0.0, period - (time.perf_counter() - loop_start)))
                    continue  # global_step frozen; re-check next tick

                # 3. Fire the next async inference: capture here (control thread, a few ms),
                #    encode+POST on the worker. The prefix = the DELAY_STEPS actions we are
                #    about to execute while inference runs.
                if pending is None and global_step >= next_request_step:
                    anchor = global_step  # new chunk's slot 0 == this step
                    prefix = extract_prefix(plan, anchor, DELAY_STEPS)
                    frames, wrist, state7 = capture()
                    pending = PendingRequest(
                        future=executor.submit(post_request, session, frames, wrist, state7, prefix),
                        anchor=anchor,
                        launch_step=global_step,
                        launch_time=time.perf_counter(),
                        prefix=prefix,
                    )
                    next_request_step = anchor + EXEC_HORIZON

                # 4. Execute the action for this step from the current plan.
                idx = global_step - plan.start_step
                idx = min(max(idx, 0), len(plan.raw) - 1)  # bounded by stop-and-wait; clamp for safety
                robot.send_action(torch.from_numpy(np.asarray(plan.raw[idx], dtype=np.float32)))

                # keep the strided cam_high history fresh during execution (history mode)
                if IMAGE_HISTORY and (tick + 1) % STRIDE_TICKS == 0:
                    history.append(cam_rgb(robot.capture_observation(), HL_CAM))
                tick += 1
                global_step += 1

                # 5. Maintain the control rate.
                elapsed = time.perf_counter() - loop_start
                if LOG_TIMING and elapsed > period and global_step % overrun_notice_every == 0:
                    print(f"  control-loop overrun: {elapsed * 1000:.1f} ms > {period * 1000:.1f} ms")
                time.sleep(max(0.0, period - elapsed))

    except KeyboardInterrupt:
        print("\nStopping control loop safely via Ctrl+C...")
    finally:
        executor.shutdown(wait=False)
        print("Disconnecting robot devices...")
        robot.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()
