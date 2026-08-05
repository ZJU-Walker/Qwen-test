#!/usr/bin/env python3
"""Hierarchical real-time client: Qwen3-VL high-level planner -> OpenPI (pi) low-level RTC policy.

This fuses two clients that already work standalone:
  - client_rtc.py        : the async real-time-chunking (RTC) controller for the Trossen arm. It owns
                           the robot + cameras (via the LeRobot fork) and runs the pi policy at a fixed
                           control rate, executing one action per tick while inference runs in a
                           background thread. (See client_rtc.py for the full RTC explanation.)
  - client_qwen_4b.py    : a Qwen3-VL-4B "task planner" that watches a rolling window of cam_high
                           frames and answers a question (e.g. "which colored block did the human hand
                           point to?"), returning a short string.

Hierarchy: Qwen runs SLOW and asynchronously. Every time it produces an answer we turn that answer into
a pi prompt (see build_instruction) and stash it. The fast RTC loop reads the MOST RECENT instruction
each time it replans, so the low-level policy is always driven by the freshest high-level intent without
ever blocking on Qwen.

Shared camera: only ONE process may own the cameras, so the LeRobot robot owns them. cam_high is read
from the LeRobot camera's cached frame (async_read) by the GUI thread and shared, in memory, with the
Qwen worker -- no second VideoCapture, no camera contention. LeRobot returns RGB; OpenCV (encode +
display) wants BGR, so we convert once.

Threads:
  - main           : GUI + cam_high reader + 'q' to quit. Reads cam_high, publishes it, draws the
                     dashboard (live frame, Qwen's strided window, Qwen text, derived pi prompt).
  - control thread : the RTC control loop (lifted from client_rtc.py, timing unchanged).
  - inference worker (ThreadPoolExecutor): captures the full observation + runs pi inference.
  - HL/Qwen thread : builds the strided frame window, POSTs to the Qwen server, updates the instruction.

Robot access stays exactly as in client_rtc.py: the inference worker reads (capture_observation) and
the control thread writes (send_action). The GUI/HL only ever read the camera's cached frame.
"""
import io
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np
import requests
import torch

# 1. Point Python DIRECTLY to the 'src' folder using your absolute path
OPENPI_CLIENT_SRC = "/home/iris/openpi-trossen-brian/packages/openpi-client/src"
if OPENPI_CLIENT_SRC not in sys.path:
    sys.path.insert(0, OPENPI_CLIENT_SRC)

# 2. Point Python to your LeRobot hardware fork
LEROBOT_FORK_PATH = "/home/iris/lerobot"
if LEROBOT_FORK_PATH and LEROBOT_FORK_PATH not in sys.path:
    sys.path.insert(0, LEROBOT_FORK_PATH)

from lerobot.common.robot_devices.robots.configs import TrossenAIRightArmOnlyRobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config
from openpi_client import websocket_client_policy
import openpi_client.image_tools as image_tools

# --- CONFIGURATION ---
SERVER_IP = "10.79.12.149"
SERVER_PORT = 8005                                   # OpenPI (pi low-level) websocket server
QWEN_URL = "http://10.79.12.149:8002/predict"        # Qwen3-VL-4B high-level server (qwen_server_4b.py)
# Match the dataset/control cadence used by the prior Trossen RTC path unless you have a measured
# reason to slow this down. One action is executed per tick.
CONTROL_HZ = 30

# --- RTC SCHEDULING (see client_rtc.py for the full rationale) ---
# DELAY_STEPS (d): control ticks of inference latency we plan around. Must be >= real server latency in
# ticks WITH MARGIN, and <= the training rtc_prefix_max_length.
DELAY_STEPS = 5
# EXEC_HORIZON (s): ticks executed between replans. Requires EXEC_HORIZON >= DELAY_STEPS and
# EXEC_HORIZON + DELAY_STEPS <= action chunk length.
EXEC_HORIZON = 5
LOG_TIMING = True

# --- HIGH-LEVEL (Qwen) PLANNER ---
HL_PROMPT = "which colored block did the human hand point to?"  # question asked of Qwen each request
HL_CAM = "cam_high"                                  # which LeRobot camera feeds Qwen + the GUI
HL_NUM_FRAMES = 10                                   # MUST equal the server's NUM_FRAMES
HL_CAMERA_FPS = 30                                   # cam_high capture rate (from the robot config)
HL_STRIDE_FRAMES = 10                                # training stride (PrototypeRobotDataset)
HL_STRIDE_SEC = HL_STRIDE_FRAMES / HL_CAMERA_FPS     # ~0.333 s between buffered frames, ~3 s window
HL_TIMEOUT_S = 5.0                                   # 4B is heavier than 3B; allow some headroom

# Qwen already emits the exact prompt the pi policy expects -- "pick up green block", "pick up yellow
# block", or "waiting" -- so we feed its output through verbatim (see build_instruction).
PI_FALLBACK_PROMPT = "waiting"                        # safe default (pi holds) until the first Qwen result

# --- GUI ---
SHOW_GUI = True
GUI_HZ = 30
WINDOW_NAME = "Hierarchical RTC: Qwen HL -> pi LL"
# ---------------------

# --- SHARED STATE (guarded by state_lock) ---
state_lock = threading.Lock()
latest_frame_bgr: np.ndarray | None = None  # newest cam_high frame (BGR uint8); written by the GUI thread
latest_instruction: str | None = None       # newest derived pi prompt; None until first Qwen result
latest_qwen_raw: str | None = None           # newest raw Qwen text (for display/debug)
hl_display_frames: list = []                 # snapshot of Qwen's strided window (BGR), for the dashboard
qwen_rate_hz: float = 0.0
qwen_latency_ms: float = 0.0
pi_status: str = "starting"


def build_instruction(pred: str) -> str | None:
    """Normalize a raw Qwen answer into the pi prompt.

    Qwen emits the full prompt the pi policy expects ("pick up green block", "pick up yellow block",
    "waiting"), so we strip whitespace and feed it through verbatim. Empty -> None so the caller keeps
    the previous instruction / fallback.
    """
    text = pred.strip()
    return text if text else None


def set_status(status: str) -> None:
    global pi_status
    with state_lock:
        pi_status = status


def current_pi_prompt() -> str:
    """The most recent instruction Qwen has produced, or the fallback before the first result."""
    with state_lock:
        return latest_instruction if latest_instruction else PI_FALLBACK_PROMPT


@dataclass
class Plan:
    """A chunk currently being executed, anchored to an absolute control-loop step."""

    start_step: int  # global step that this chunk's slot 0 corresponds to
    raw: np.ndarray  # (H, 14) actions in robot space (what we send to the arm)
    model: np.ndarray | None = None  # optional diagnostics: pre-output-transform model actions


@dataclass
class ChunkResult:
    """Result of one policy request plus the timing needed to debug real-time execution."""

    raw: np.ndarray
    model: np.ndarray | None
    capture_ms: float
    round_trip_ms: float
    server_infer_ms: float | None


@dataclass
class PendingRequest:
    """An in-flight RTC request anchored to an absolute control-loop step."""

    future: Future
    anchor: int
    launch_step: int
    launch_time: float
    prefix: np.ndarray


def capture_payload(robot, cameras: list) -> dict:
    """Capture the current observation and pack it into the server's expected payload format.

    The prompt is the freshest Qwen-derived instruction (or the fallback). Because this runs on every
    replan, the low-level policy always tracks the latest high-level intent.
    """
    obs = robot.capture_observation()

    # Spoof a 14-dim state: this robot is right-arm-only, so the left arm [0:7] stays zero.
    right_arm_state = obs["observation.state"].detach().cpu().numpy().astype(np.float32)
    full_state = np.zeros(14, dtype=np.float32)
    full_state[7:14] = right_arm_state

    images = {}
    for cam in cameras:
        img = obs[f"observation.images.{cam}"].detach().cpu().numpy()  # HWC, RGB
        img = image_tools.resize_with_pad(img, height=224, width=224)
        images[cam] = np.transpose(img, (2, 0, 1))  # HWC -> CHW

    return {"state": full_state, "images": images, "prompt": current_pi_prompt()}


def request_chunk(client, payload: dict, action_prefix: np.ndarray | None = None, prefix_length: int = 0):
    """Blocking inference call. `action_prefix`, when present, is absolute robot-space actions."""
    payload = dict(payload)
    if action_prefix is not None and prefix_length > 0:
        payload["action_prefix"] = np.asarray(action_prefix, dtype=np.float32)
        payload["prefix_length"] = int(prefix_length)
    t0 = time.perf_counter()
    response = client.infer(payload)
    round_trip_ms = (time.perf_counter() - t0) * 1000.0
    raw = np.asarray(response["actions"], dtype=np.float32)
    model = np.asarray(response["actions_model"], dtype=np.float32) if "actions_model" in response else None
    policy_timing = response.get("policy_timing", {})
    server_infer_ms = policy_timing.get("infer_ms")
    if server_infer_ms is not None:
        server_infer_ms = float(server_infer_ms)
    return ChunkResult(raw=raw, model=model, capture_ms=0.0, round_trip_ms=round_trip_ms, server_infer_ms=server_infer_ms)


def request_chunk_from_robot(
    client,
    robot,
    cameras: list,
    action_prefix: np.ndarray | None = None,
    prefix_length: int = 0,
) -> ChunkResult:
    """Capture observation and run inference.

    This is what async replans submit to the worker thread. Keeping camera capture out of the
    control-loop thread avoids a periodic command-timing hitch at every replan.
    """
    t0 = time.perf_counter()
    payload = capture_payload(robot, cameras)
    capture_ms = (time.perf_counter() - t0) * 1000.0
    result = request_chunk(client, payload, action_prefix, prefix_length)
    result.capture_ms = capture_ms
    return result


def extract_prefix(plan: "Plan", anchor: int, delay_steps: int) -> np.ndarray:
    """Absolute robot-space prefix of exactly (delay_steps, 14), starting at absolute step `anchor`.

    Clamped to the plan and zero-padded so the shape is ALWAYS the same. A short/empty slice would
    change the array shape and force the JAX server to recompile mid-run (a multi-second stall).
    """
    action_dim = plan.raw.shape[1]
    p0 = max(0, min(anchor - plan.start_step, len(plan.raw)))
    chunk = plan.raw[p0 : p0 + delay_steps]
    if len(chunk) < delay_steps:
        pad = np.zeros((delay_steps - len(chunk), action_dim), dtype=plan.raw.dtype)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk


def send_arm(robot, action14: np.ndarray) -> None:
    """Send the right-arm slice [7:14] of a 14-dim model action to this hardware."""
    robot.send_action(torch.from_numpy(np.asarray(action14[7:14], dtype=np.float32)))


# =====================================================================================
# High-level (Qwen) worker
# =====================================================================================
def high_level_worker(stop_event: threading.Event) -> None:
    """Background thread: run Qwen as fast as the server allows on the shared cam_high stream.

    Never touches a camera directly. Reads frames from latest_frame_bgr (published by the GUI thread),
    maintains a strided HL_NUM_FRAMES window (training parity), and writes latest_instruction.
    """
    global latest_instruction, latest_qwen_raw, hl_display_frames, qwen_rate_hz, qwen_latency_ms

    # Wait for the first shared frame to exist.
    first = None
    while not stop_event.is_set():
        with state_lock:
            first = None if latest_frame_bgr is None else latest_frame_bgr.copy()
        if first is not None:
            break
        time.sleep(0.05)
    if stop_event.is_set():
        return

    # Prime the window by duplicating the first frame (mirrors training's clamping of negative indices).
    window = [first.copy() for _ in range(HL_NUM_FRAMES)]
    last_grab = time.time()
    req_times: deque = deque(maxlen=10)
    print("[HL] Qwen worker active.")

    while not stop_event.is_set():
        # Pull the newest frame into the window at a fixed stride so the frames span a short window of
        # time rather than being identical.
        now = time.time()
        if now - last_grab >= HL_STRIDE_SEC:
            with state_lock:
                frame = None if latest_frame_bgr is None else latest_frame_bgr.copy()
            if frame is not None:
                window.append(frame)
                window = window[-HL_NUM_FRAMES:]
                last_grab = now
                with state_lock:
                    hl_display_frames = [f.copy() for f in window]

        # Encode the window as JPEGs, oldest-first. cv2.imencode expects BGR (which is what we store),
        # so the resulting bytes match the standalone client and decode to correct RGB on the server.
        files_payload = []
        for i, frame in enumerate(window):
            ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if ok:
                files_payload.append(("files", (f"frame_{i}.jpg", io.BytesIO(enc.tobytes()), "image/jpeg")))
        if len(files_payload) != HL_NUM_FRAMES:
            time.sleep(0.01)
            continue

        t0 = time.time()
        try:
            resp = requests.post(QWEN_URL, files=files_payload, data={"prompt": HL_PROMPT}, timeout=HL_TIMEOUT_S)
            if resp.status_code == 200:
                raw = resp.json()["prediction"].strip()
                instr = build_instruction(raw)
                latency_ms = (time.time() - t0) * 1000.0
                req_times.append(time.time())
                rate = (len(req_times) - 1) / (req_times[-1] - req_times[0]) if len(req_times) >= 2 else 0.0
                with state_lock:
                    latest_qwen_raw = raw
                    if instr is not None:
                        latest_instruction = instr
                    qwen_latency_ms = latency_ms
                    qwen_rate_hz = rate
                print(
                    f"[HL] {time.strftime('%X')} qwen={raw!r} -> pi={instr!r} "
                    f"| latency={latency_ms:.0f} ms | rate={rate:.2f} Hz"
                )
            else:
                print(f"[HL] server {resp.status_code}: {resp.text[:120]}")
                time.sleep(0.1)
        except requests.exceptions.RequestException as e:
            print(f"[HL] network fault: {e}")
            time.sleep(0.1)

    print("[HL] Qwen worker stopped.")


# =====================================================================================
# Visualization (main thread)
# =====================================================================================
def build_dashboard(live_bgr, frames, qwen_raw, pi_prompt, rate_hz, latency_ms, status):
    """Live cam_high large, Qwen's strided window as a history strip, HL/LL text at the bottom."""
    main_w, main_h = 640, 480
    history_count = max(1, HL_NUM_FRAMES - 1)
    hist_w = main_w // history_count
    hist_h = int(hist_w * (main_h / main_w))

    canvas = np.zeros((40 + main_h + hist_h + 130, main_w, 3), dtype=np.uint8)

    if live_bgr is None and not frames:
        cv2.putText(canvas, "Waiting for cam_high...", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return canvas

    cv2.putText(canvas, "Qwen HL  ->  pi LL  (hierarchical RTC)", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    main_src = live_bgr if live_bgr is not None else frames[-1]
    canvas[40:40 + main_h, 0:main_w] = cv2.resize(main_src, (main_w, main_h))

    # History strip: the exact frames Qwen is reasoning over, oldest -> newest left to right.
    for i in range(min(history_count, len(frames))):
        x = i * hist_w
        y = 40 + main_h
        canvas[y:y + hist_h, x:x + hist_w] = cv2.resize(frames[i], (hist_w, hist_h))
        cv2.putText(canvas, f"t-{len(frames) - 1 - i}", (x + 4, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    base = 40 + main_h + hist_h
    qwen_txt = qwen_raw if qwen_raw is not None else "(waiting for Qwen)"
    prompt_txt = pi_prompt if pi_prompt is not None else PI_FALLBACK_PROMPT
    cv2.putText(canvas, f"Qwen: {qwen_txt}", (15, base + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)   # cyan
    cv2.putText(canvas, f"pi prompt: {prompt_txt}", (15, base + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)     # green
    cv2.putText(canvas, f"Qwen rate: {rate_hz:.2f} Hz | {latency_ms:.0f} ms    pi: {status}",
                (15, base + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(canvas, "[q] quit", (15, base + 118),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    return canvas


def gui_loop(robot, stop_event: threading.Event) -> None:
    """Main thread: read cam_high, publish it for Qwen, and draw the dashboard. 'q' quits everything."""
    global latest_frame_bgr

    cam = robot.cameras.get(HL_CAM)
    if cam is None:
        print(f"[GUI] camera '{HL_CAM}' not found in {list(robot.cameras)}; Qwen will get no frames.")

    gui_ok = SHOW_GUI
    period = 1.0 / GUI_HZ
    while not stop_event.is_set():
        t0 = time.perf_counter()

        # Read the camera's latest cached frame (RGB, resized) and publish it as BGR.
        live = None
        if cam is not None:
            try:
                rgb = np.asarray(cam.async_read(), dtype=np.uint8)
                live = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                with state_lock:
                    latest_frame_bgr = live
            except Exception as e:  # noqa: BLE001
                print(f"[GUI] camera read error: {e}")

        if gui_ok:
            with state_lock:
                disp_frames = list(hl_display_frames)
                raw = latest_qwen_raw
                instr = latest_instruction
                rate = qwen_rate_hz
                latency = qwen_latency_ms
                status = pi_status
            dash = build_dashboard(live, disp_frames, raw, instr, rate, latency, status)
            try:
                cv2.imshow(WINDOW_NAME, dash)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    print("\n[GUI] quit key pressed.")
                    break
            except cv2.error as e:
                print(f"[GUI] no display available, running headless: {e}")
                gui_ok = False

        elapsed = time.perf_counter() - t0
        time.sleep(max(0.0, period - elapsed))

    stop_event.set()
    if SHOW_GUI:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


# =====================================================================================
# Low-level (pi) RTC control loop  -- timing identical to client_rtc.py
# =====================================================================================
def control_loop(client, robot, cameras: list, stop_event: threading.Event) -> None:
    if EXEC_HORIZON < DELAY_STEPS:
        raise ValueError("EXEC_HORIZON must be >= DELAY_STEPS for the RTC pipeline to stay ahead.")

    executor = ThreadPoolExecutor(max_workers=1)
    period = 1.0 / CONTROL_HZ

    try:
        # --- Cold start: one synchronous, prefix-free inference. Also warms the cameras (first-frame
        #     grab is ~2 s) and the server's plain code path. ---
        print("Cold-start inference (no prefix)...")
        set_status("cold start")
        result0 = request_chunk_from_robot(client, robot, cameras)
        raw0, model0 = result0.raw, result0.model
        if LOG_TIMING:
            print(
                "  cold-start timing: "
                f"capture={result0.capture_ms:.0f} ms | round_trip={result0.round_trip_ms:.0f} ms"
                + (f" | server={result0.server_infer_ms:.0f} ms" if result0.server_infer_ms is not None else "")
            )
        horizon = len(raw0)
        if EXEC_HORIZON + DELAY_STEPS > horizon:
            raise ValueError(
                f"EXEC_HORIZON + DELAY_STEPS ({EXEC_HORIZON + DELAY_STEPS}) exceeds chunk length ({horizon})."
            )
        plan = Plan(start_step=0, raw=raw0, model=model0)

        # --- Pre-warm the server's PREFIX code path (first action-prefix inference triggers a
        #     multi-second XLA compile). Do it here with the exact runtime shape. ---
        print("Pre-warming the server's prefix path (compiling the action-prefix signature)...")
        set_status("warming prefix")
        warm_prefix = extract_prefix(plan, EXEC_HORIZON, DELAY_STEPS)  # (DELAY_STEPS, 14)
        for i in range(2):
            t0 = time.perf_counter()
            warm_result = request_chunk_from_robot(client, robot, cameras, warm_prefix, DELAY_STEPS)
            total_ms = (time.perf_counter() - t0) * 1000
            print(
                f"  warmup prefix call {i}: total={total_ms:.0f} ms | "
                f"capture={warm_result.capture_ms:.0f} ms | round_trip={warm_result.round_trip_ms:.0f} ms"
                + (f" | server={warm_result.server_infer_ms:.0f} ms" if warm_result.server_infer_ms is not None else "")
            )
        print("Prefix path warm.")

        global_step = 0
        next_request_step = EXEC_HORIZON  # when to fire the first async replan
        pending: PendingRequest | None = None
        wait_ticks = 0  # how long we've been holding for a late chunk
        wait_notice_every = max(1, CONTROL_HZ)  # throttle the "holding" message to ~1/sec
        overrun_notice_every = max(1, CONTROL_HZ)

        print(f"\nRTC running at {CONTROL_HZ} Hz | d={DELAY_STEPS} | replan every {EXEC_HORIZON} steps")
        print("Press [q] in the GUI window (or Ctrl+C) to quit.\n")
        set_status("running")

        while not stop_event.is_set():
            loop_start = time.perf_counter()

            # 1. Install a finished async inference. Its first DELAY_STEPS actions are clamped to the
            #    prefix we committed, so it splices in seamlessly at idx == DELAY_STEPS.
            if pending is not None and pending.future.done():
                try:
                    result = pending.future.result()
                    actual_ticks = global_step - pending.launch_step
                    elapsed_ms = (time.perf_counter() - pending.launch_time) * 1000.0
                    clamp_diff = float(np.max(np.abs(result.raw[:DELAY_STEPS] - pending.prefix)))
                    plan = Plan(start_step=pending.anchor, raw=result.raw, model=result.model)
                    if LOG_TIMING:
                        print(
                            "  chunk ready: "
                            f"ticks={actual_ticks} | elapsed={elapsed_ms:.0f} ms | "
                            f"clamp={clamp_diff:.1e} | capture={result.capture_ms:.0f} ms | "
                            f"round_trip={result.round_trip_ms:.0f} ms"
                            + (
                                f" | server={result.server_infer_ms:.0f} ms"
                                if result.server_infer_ms is not None
                                else ""
                            )
                        )
                    if wait_ticks:
                        print(f"  chunk arrived after holding {wait_ticks} tick(s); resuming.")
                except Exception as e:  # noqa: BLE001
                    print(f"  Inference request failed, keeping current plan: {e}")
                pending = None
                wait_ticks = 0
                set_status("running")

            # 2. STOP-AND-WAIT: we committed to exactly DELAY_STEPS actions as the prefix. If the new
            #    chunk has not arrived once we've executed them, HOLD the last committed action (do NOT
            #    advance global_step, do NOT run past the prefix) until it lands.
            if pending is not None and global_step >= pending.anchor + DELAY_STEPS:
                hold_idx = (pending.anchor + DELAY_STEPS - 1) - plan.start_step
                hold_idx = min(max(hold_idx, 0), len(plan.raw) - 1)
                send_arm(robot, plan.raw[hold_idx])
                if wait_ticks % wait_notice_every == 0:
                    print(f"  holding for late chunk (inference > DELAY_STEPS={DELAY_STEPS} ticks)...")
                    set_status("holding for late chunk")
                wait_ticks += 1
                elapsed = time.perf_counter() - loop_start
                time.sleep(max(0.0, period - elapsed))
                continue  # global_step frozen; re-check for the chunk next tick

            # 3. Fire the next async inference at the start of this control tick, sending the actions we
            #    are about to commit to during the expected delay.
            if pending is None and global_step >= next_request_step:
                anchor = global_step  # new chunk's slot 0 == this step
                prefix = extract_prefix(plan, anchor, DELAY_STEPS)  # always (DELAY_STEPS, 14)
                pending = PendingRequest(
                    future=executor.submit(request_chunk_from_robot, client, robot, cameras, prefix, DELAY_STEPS),
                    anchor=anchor,
                    launch_step=global_step,
                    launch_time=time.perf_counter(),
                    prefix=prefix,
                )
                next_request_step = anchor + EXEC_HORIZON

            # 4. Execute the action for this step from the current plan.
            idx = global_step - plan.start_step
            idx = min(max(idx, 0), len(plan.raw) - 1)  # bounded by stop-and-wait; clamp for safety
            send_arm(robot, plan.raw[idx])

            global_step += 1

            # 5. Maintain the control rate.
            elapsed = time.perf_counter() - loop_start
            if LOG_TIMING and elapsed > period and global_step % overrun_notice_every == 0:
                print(f"  control-loop overrun: {elapsed * 1000:.1f} ms > {period * 1000:.1f} ms")
            time.sleep(max(0.0, period - elapsed))

    except Exception as e:  # noqa: BLE001
        print(f"\n[control] fatal error, stopping: {e}")
    finally:
        executor.shutdown(wait=False)
        set_status("stopped")
        stop_event.set()  # tell the GUI/HL threads to exit too
        print("[control] loop stopped.")


def main():
    print(f"Connecting to OpenPI (pi) server at {SERVER_IP}:{SERVER_PORT}...")
    client = websocket_client_policy.WebsocketClientPolicy(host=SERVER_IP, port=SERVER_PORT)

    print("Initializing Trossen robot and cameras...")
    robot_config = TrossenAIRightArmOnlyRobotConfig(
        max_relative_target=None,
        min_time_to_move_multiplier=4.0,
        camera_interface="opencv",
    )
    robot = make_robot_from_config(robot_config)
    robot.connect()
    cameras = list(robot_config.cameras.keys())
    print(f"Successfully connected to robot cameras: {cameras}")
    if HL_CAM not in cameras:
        print(f"WARNING: HL_CAM '{HL_CAM}' not in {cameras}; Qwen will get no frames.")

    # Warm the cameras on the main thread BEFORE spawning readers: the first async_read starts each
    # camera's internal capture thread, and LeRobot doesn't guard that start against concurrent callers.
    # Doing it once here means the GUI thread and the inference worker never race to start it (and it
    # absorbs the ~2 s first-frame latency up front).
    print("Warming cameras (first frame can take ~2 s)...")
    try:
        robot.capture_observation()
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: camera warm-up failed: {e}")

    stop_event = threading.Event()
    control_thread = threading.Thread(target=control_loop, args=(client, robot, cameras, stop_event), daemon=True)
    hl_thread = threading.Thread(target=high_level_worker, args=(stop_event,), daemon=True)
    control_thread.start()
    hl_thread.start()

    try:
        gui_loop(robot, stop_event)  # blocks on the main thread until quit
    except KeyboardInterrupt:
        print("\nStopping via Ctrl+C...")
    finally:
        stop_event.set()
        control_thread.join(timeout=3.0)
        hl_thread.join(timeout=3.0)
        print("Disconnecting robot devices...")
        robot.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()
