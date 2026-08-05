#!/usr/bin/env python3
"""Asynchronous real-time-chunking (RTC) client for the Qwen3-VL action-expert server --
2-WRIST VARIANT for ee6d-era checkpoints (trained with cam_right_wrist + cam_left_wrist).

Identical to client_action_expert_rtc_use_me.py except it also captures and uploads the
CURRENT left-wrist frame each request (`wrist_left` field; the server 400s without it for
a 2-wrist checkpoint). The action interface is unchanged even though the checkpoint is
end-effector-space: the client still sends 7-dim joint state and receives (50, 7) joint
actions -- all representation conversion (FK in, Gram-Schmidt + IK out) is server-side.

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
NOTE: a predict-subtask server that DECODES a subtask every request adds ~0.3-0.9 s -- usually
too slow for RTC. Two ways to avoid it: (a) a subtask-input checkpoint (--subtask_input); or
(b) an implicit-HL / insulated checkpoint (--insulated_subtask --subtask_every 0), where the
expert ignores the subtask so the server never decodes one and (with SmolVLA layer skipping)
early-exits the VLM -- this is the fast mode-c deployment and works fine with RTC.

Delta-action gotcha: training uses delta joint actions, so the model's action space is relative
to the observation state and NOT portable across observations. Prefixes are therefore sent in
ABSOLUTE joint space (a slice of the previously returned actions); the server re-expresses them
against the current state before clamping.

Validate the model side FIRST with client_action_expert_rtc_sync.py (clamp / seam / [bound]).
Copy this file to the ROBOT computer; the server runs on the GPU node.
"""

import io
import itertools
import json
import select
import sys
import termios
import time
import tty
import uuid
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

# Shared pop-up visualizer (viz_utils.py lives next to this file in scripts/).
from viz_utils import InputVisualizer

# ----------------------------- CONFIG -----------------------------
SERVER_URL = "http://10.79.12.149:8003/infer"   # <-- GPU node running qwen_action_expert_server.py
HL_CAM = "cam_high"
WRIST_CAM = "cam_right_wrist"
WRIST_LEFT_CAM = "cam_left_wrist"
SEND_LEFT_WRIST = True   # 2-wrist checkpoints (ee6d run onward); False for older ones
SEND_STATE_HISTORY = True  # --state_history checkpoints (0804dag run onward): send the 9
                           # joint states captured at the history-frame ticks; False for older
# --- these MUST match the server's checkpoint (see /health) ---
IMAGE_HISTORY = True            # True: send NUM_FRAMES strided cam_high frames; False: 1 current frame
SUBTASK = "pick up yellow block"  # subtask-input mode input; ignored by predict-subtask servers
NUM_FRAMES = 10                 # history length (history mode only)
STRIDE_TICKS = 10               # training stride: one cam_high history frame every 10 ticks
CONTROL_HZ = 12                 # control cadence; one action executed per tick

# --- RTC SCHEDULING ---
# DELAY_STEPS (d): control ticks of inference latency we plan around. Must be >= the real
# latency in ticks WITH MARGIN, and <= the training --rtc_prefix_max_length.
DELAY_STEPS = 8
# EXEC_HORIZON (s): ticks between successive replans. Smaller = more reactive.
# Requires EXEC_HORIZON >= DELAY_STEPS and EXEC_HORIZON + DELAY_STEPS <= chunk length (50).
EXEC_HORIZON = 8
NUM_FLOW_STEPS = 10             # 10 >> 5 in on-robot rollouts (despite offline MSE parity);
                                # keep 10 -- open-loop MSE is a weak proxy for closed-loop quality
REQUEST_TIMEOUT_S = 60.0
JPEG_QUALITY = 90
LOG_TIMING = True
# --- visualization (pop-up of the frames + prompt/model input we send; see viz_utils.py) ---
SHOW_VIZ = True                # False to run headless (no display, zero overhead)
VIZ_WINDOW = "rtc client - model input"
VIZ_PUMP_EVERY = 3             # re-show the window every N control ticks (keep the 30 Hz loop light)
SHOW_MODEL_INPUT = True        # also render the full decoded model_input_text panel (verbose)
VIZ_DEBUG = False               # ask the server for prompt/model_input on EVERY chunk so the panel
                               # updates live (adds a little server/decoding + network overhead).
                               # Set False to only show the cold-start prompt (no per-chunk cost).
# --- frame dedup (latency optimization) ---
# The strided history window advances every STRIDE_TICKS while we replan every EXEC_HORIZON,
# so consecutive requests share most of their 10 cam_high frames. Each history frame gets a
# monotonically increasing id at append time; requests send the full ID window plus JPEGs
# ONLY for ids the server hasn't cached yet (usually 0-1 instead of 10 -- less encoding, less
# network, less server-side decoding). If the server responds 409 (cache miss, e.g. it was
# restarted mid-run), the client automatically re-sends everything once. Requires a server
# with supports_frame_dedup (see /health); set False to always upload every frame.
FRAME_DEDUP = True
SESSION_ID = uuid.uuid4().hex[:16]  # fresh per client run -> the server starts a clean cache
# ------------------------------------------------------------------

# ids the server is known to have cached (exactly the previous request's window after each
# success). Touched only by the request path, which is serialized (single worker, one
# in-flight request at a time).
_sent_ids: set[int] = set()


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
    frames_uploaded: int = -1     # cam_high JPEGs this request actually carried (dedup: ~0-1)


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


def raise_with_detail(resp) -> None:
    """Surface the server's error `detail` (remediation text) before raising -- the bare
    HTTPError message drops the response body, leaving only '400 Bad Request'."""
    if not resp.ok:
        try:
            detail = resp.json().get("detail", resp.text[:300])
        except Exception:  # noqa: BLE001
            detail = resp.text[:300]
        print(f"  SERVER ERROR {resp.status_code}: {detail}")
    raise_with_detail(resp)


def encode_jpeg(rgb: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return enc.tobytes()


def post_request(session, cam_high_frames, frame_ids, wrist_rgb, state7, action_prefix,
                 wrist_left_rgb=None, state_history=None, debug=False) -> ChunkResult:
    """Encode + POST one observation. Runs on the WORKER thread (JPEG encoding + the HTTP
    round trip stay out of the control loop). `action_prefix` is (d, 7) ABSOLUTE joints or
    None (cold start). `frame_ids` is the id window matching cam_high_frames (dedup mode) or
    None (upload every frame). `debug=True` asks the server to also return the decoded token
    sequence the model conditioned on (input sanity check)."""
    data = {
        "state": json.dumps([float(x) for x in np.asarray(state7).reshape(-1)]),
        "subtask": SUBTASK,
        "run_fast": "false",
        "num_flow_steps": str(NUM_FLOW_STEPS),
        "debug": "true" if debug else "false",
    }
    if state_history is not None:
        # (NUM_FRAMES-1, 7) joint states at the history-frame ticks, oldest first --
        # required by --state_history checkpoints (see /health).
        data["state_history"] = json.dumps(np.asarray(state_history, dtype=np.float32).tolist())
    if action_prefix is not None and len(action_prefix) > 0:
        data["action_prefix"] = json.dumps(np.asarray(action_prefix, dtype=np.float32).tolist())

    def _post(upload_pairs, extra):
        """upload_pairs: [(frame_id_or_None, frame)] cam_high frames to actually upload."""
        files = [("cam_high", (f"f{i}.jpg", io.BytesIO(encode_jpeg(frame)), "image/jpeg"))
                 for i, (_, frame) in enumerate(upload_pairs)]
        files.append(("wrist", ("wrist.jpg", io.BytesIO(encode_jpeg(wrist_rgb)), "image/jpeg")))
        if wrist_left_rgb is not None:
            files.append(("wrist_left", ("wrist_left.jpg",
                          io.BytesIO(encode_jpeg(wrist_left_rgb)), "image/jpeg")))
        return session.post(SERVER_URL, files=files, data={**data, **extra},
                            timeout=REQUEST_TIMEOUT_S), len(upload_pairs)

    t0 = time.perf_counter()
    if frame_ids is None:
        # legacy: upload the full window every request
        resp, n_up = _post([(None, f) for f in cam_high_frames], {})
        raise_with_detail(resp)
    else:
        # dedup: upload only ids the server hasn't cached (each unique id at most once --
        # the primed window repeats one id NUM_FRAMES times but uploads it a single time).
        def new_pairs(skip_known: bool):
            seen: set[int] = set()
            return [(fid, frame) for fid, frame in zip(frame_ids, cam_high_frames)
                    if (not skip_known or fid not in _sent_ids)
                    and not (fid in seen or seen.add(fid))]

        def dedup_extra(pairs):
            return {
                "session_id": SESSION_ID,
                "cam_high_ids": json.dumps(list(map(int, frame_ids))),
                "cam_high_new_ids": json.dumps([int(fid) for fid, _ in pairs]),
            }

        pairs = new_pairs(skip_known=True)
        resp, n_up = _post(pairs, dedup_extra(pairs))
        if resp.status_code == 409:
            # server lost its cache (restart?) -> re-send the whole window once
            print("  frame cache miss on server; re-sending the full window.")
            _sent_ids.clear()
            pairs = new_pairs(skip_known=False)
            resp, n_up = _post(pairs, dedup_extra(pairs))
        raise_with_detail(resp)
        _sent_ids.clear()
        _sent_ids.update(map(int, frame_ids))  # server pruned its cache to exactly this window

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
        frames_uploaded=j.get("frames_uploaded", n_up),
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
    obs0 = robot.capture_observation()
    # Pre-flight: the RightArmOnly lerobot config historically does NOT register the left
    # wrist camera (pi05-era clients zero-spoofed it). This checkpoint trained on REAL
    # left-wrist images, so spoofing is a train/serve mismatch and SEND_LEFT_WRIST=False
    # is rejected by a 2-wrist server -- the only correct fix is exposing the camera.
    if SEND_LEFT_WRIST and f"observation.images.{WRIST_LEFT_CAM}" not in obs0:
        cams = sorted(k.removeprefix("observation.images.") for k in obs0
                      if k.startswith("observation.images."))
        robot.disconnect()  # release the hardware first: home -> sleep pose, not torqued-at-home
        raise RuntimeError(
            f"robot config does not expose '{WRIST_LEFT_CAM}' (available cameras: {cams}). "
            "Add the left wrist camera to TrossenAIRightArmOnlyRobotConfig in the robot's "
            "lerobot fork (mirror the cam_right_wrist entry / the recording rig's "
            "trossen_ai_solo config, which has it). Do NOT zero-spoof it and do NOT set "
            "SEND_LEFT_WRIST=False for this 2-wrist checkpoint.")

    if EXEC_HORIZON < DELAY_STEPS:
        raise ValueError("EXEC_HORIZON must be >= DELAY_STEPS for the RTC pipeline to stay ahead.")

    session = requests.Session()
    executor = ThreadPoolExecutor(max_workers=1)
    period = 1.0 / CONTROL_HZ

    # cam_high history primed with the first frame (mirrors training's clamped indices at t=0).
    obs = robot.capture_observation()
    history = deque([cam_rgb(obs, HL_CAM)] * NUM_FRAMES, maxlen=NUM_FRAMES)
    # State history in lockstep with the frame history: primed with the current state
    # (mirrors training's edge clamp: repeat the first state), appended at the same
    # STRIDE_TICKS cadence below.
    state_hist = deque([obs["observation.state"].detach().cpu().numpy().astype(np.float32)]
                       * NUM_FRAMES, maxlen=NUM_FRAMES)
    # Frame-dedup bookkeeping: one id per history ENTRY, advanced in lockstep with `history`
    # (same deque semantics; the append below is the only place ids are minted). The primed
    # duplicates share id 0 -- one identical frame, uploaded once.
    history_ids = deque([0] * NUM_FRAMES, maxlen=NUM_FRAMES)
    next_frame_id = itertools.count(1)

    viz = InputVisualizer(history_mode=IMAGE_HISTORY, stride_ticks=STRIDE_TICKS,
                          control_hz=CONTROL_HZ, window=VIZ_WINDOW, enabled=SHOW_VIZ,
                          show_model_input=SHOW_MODEL_INPUT)
    current_subtask = current_prompt = current_model_input = None  # latest server text fields
    last_rtt = None  # round-trip of the most recently installed chunk (for the viz banner)

    def capture():
        """Build the request payload on the CONTROL thread (async_read is a few ms; keeping all
        camera access on one thread avoids cross-thread capture). The cam_high history is fed
        elsewhere at a single constant STRIDE_TICKS cadence -- this ONLY reads that window and
        grabs the CURRENT wrist still + state (both aligned to the request instant).

        Also snapshots the frame-id window (dedup): ids and frames are read together on this
        thread, so they always correspond even while the stride feed keeps appending. In
        no-history mode the single current frame is always new -- no dedup (ids=None)."""
        obs = robot.capture_observation()
        cur = cam_rgb(obs, HL_CAM)
        frames = list(history) if IMAGE_HISTORY else [cur]
        ids = list(history_ids) if (IMAGE_HISTORY and FRAME_DEDUP) else None
        left = cam_rgb(obs, WRIST_LEFT_CAM) if SEND_LEFT_WRIST else None
        # 9 oldest buffered states = the states at the ticks of history frames 0..8 (the
        # newest frame's state is the CURRENT state, sent separately below).
        shist = [s.tolist() for s in list(state_hist)[:-1]] if SEND_STATE_HISTORY else None
        return frames, ids, cam_rgb(obs, WRIST_CAM), left, shist, \
            obs["observation.state"].detach().cpu().numpy().astype(np.float32)

    # --- Cold start: one synchronous, prefix-free inference (also warms the server). ---
    print(f"Cold-start inference (no prefix, subtask={SUBTASK!r})...")
    frames, ids, wrist, wrist_left, shist, state7 = capture()
    try:
        result0 = post_request(session, frames, ids, wrist, state7, None,
                               wrist_left_rgb=wrist_left, state_history=shist, debug=True)
    except Exception:
        robot.disconnect()  # release the hardware (home -> sleep) before dying at startup
        raise
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
    current_subtask = result0.subtask
    # Cold start always requests debug (for the console dump above), but only feed the prompt/
    # model-input panels when VIZ_DEBUG is on -- otherwise they'd show a frozen cold-start
    # snapshot the loop never refreshes. VIZ_DEBUG=False => panels hidden entirely.
    current_prompt = result0.prompt if VIZ_DEBUG else None
    current_model_input = result0.model_input_text if VIZ_DEBUG else None
    last_rtt = result0.round_trip_ms
    viz.render(frames, wrist, current_subtask, tick=0, rtt_ms=last_rtt,
               prompt=current_prompt, model_input=current_model_input)

    # --- Warm the prefix path once end-to-end (validates the server accepts action_prefix and
    #     surfaces a checkpoint/latency problem BEFORE the real-time loop). ---
    print("Warming the prefix path...")
    warm_prefix = extract_prefix(plan, EXEC_HORIZON, DELAY_STEPS)
    frames, ids, wrist, wrist_left, shist, state7 = capture()
    try:
        warm = post_request(session, frames, ids, wrist, state7, warm_prefix,
                            wrist_left_rgb=wrist_left, state_history=shist)
    except Exception:
        robot.disconnect()
        raise
    if ids is not None:
        print(f"  frame dedup active (session {SESSION_ID}): warm call uploaded "
              f"{warm.frames_uploaded}/{len(frames)} cam_high frames")
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
                        # New chunk is live: update the text fields; the next constant-rate
                        # stride render (below) picks them up for the banner/panels.
                        current_subtask = result.subtask
                        last_rtt = result.round_trip_ms
                        if result.prompt is not None:            # only present when VIZ_DEBUG
                            current_prompt = result.prompt
                        if result.model_input_text is not None:
                            current_model_input = result.model_input_text
                        if LOG_TIMING:
                            print(f"  chunk ready: ticks={actual_ticks} | clamp={clamp_diff:.1e} | "
                                  f"round_trip={result.round_trip_ms:.0f} ms | server={result.server_total_ms} ms | "
                                  f"frames_sent={result.frames_uploaded}/{NUM_FRAMES if IMAGE_HISTORY else 1} | "
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

                # ~. Feed the cam_high history at ONE constant STRIDE_TICKS cadence -- the single
                #    source of truth for the window (matches training frame_stride) -- and refresh
                #    the viz on the same tick. Runs BEFORE the fire so a request captured this tick
                #    sees the just-appended frame (newest = current when EXEC_HORIZON aligns with
                #    STRIDE_TICKS). Frozen during a hold (we never reach here after the continue).
                if global_step % STRIDE_TICKS == 0 and (IMAGE_HISTORY or viz.enabled):
                    obs = robot.capture_observation()
                    if IMAGE_HISTORY:
                        history.append(cam_rgb(obs, HL_CAM))
                        history_ids.append(next(next_frame_id))  # dedup id, minted ONLY here
                        state_hist.append(
                            obs["observation.state"].detach().cpu().numpy().astype(np.float32))
                    viz.render(list(history) if IMAGE_HISTORY else [cam_rgb(obs, HL_CAM)],
                               cam_rgb(obs, WRIST_CAM), current_subtask, tick=global_step,
                               rtt_ms=last_rtt, prompt=current_prompt, model_input=current_model_input)

                # 3. Fire the next async inference: capture here (control thread, a few ms),
                #    encode+POST on the worker. The prefix = the DELAY_STEPS actions we are
                #    about to execute while inference runs.
                if pending is None and global_step >= next_request_step:
                    anchor = global_step  # new chunk's slot 0 == this step
                    prefix = extract_prefix(plan, anchor, DELAY_STEPS)
                    frames, ids, wrist, wrist_left, shist, state7 = capture()
                    pending = PendingRequest(
                        future=executor.submit(post_request, session, frames, ids, wrist, state7, prefix,
                                               wrist_left, shist, VIZ_DEBUG and viz.enabled),
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

                global_step += 1

                # keep the window responsive without burdening the 30 Hz loop every tick
                if global_step % VIZ_PUMP_EVERY == 0:
                    viz.pump()
                if viz.quit_requested:
                    print("\nQuit requested from the viz window.")
                    break

                # 5. Maintain the control rate.
                elapsed = time.perf_counter() - loop_start
                if LOG_TIMING and elapsed > period and global_step % overrun_notice_every == 0:
                    print(f"  control-loop overrun: {elapsed * 1000:.1f} ms > {period * 1000:.1f} ms")
                time.sleep(max(0.0, period - elapsed))

    except KeyboardInterrupt:
        print("\nStopping control loop safely via Ctrl+C...")
    finally:
        viz.close()
        executor.shutdown(wait=False)
        print("Disconnecting robot devices...")
        robot.disconnect()
        print("Done!")


if __name__ == "__main__":
    main()
