#!/usr/bin/env python3
"""DAgger data collection: RTC policy rollout + pedal-triggered kinesthetic intervention.

Runs the SAME asynchronous real-time-chunking rollout as client_action_expert_rtc_use_me.py
(so the on-policy distribution matches deployment), but adds a foot pedal that lets a human
take over MID-EPISODE and records the entire episode -- rollout, intervention, and recovery --
as a LeRobot dataset trainable by the openpi pipeline.

    HOLD PEDAL  -> follower arm drops into gravity compensation (Torque_Enable=0 ->
                   external_effort mode + zero efforts: the arm floats and you move it by
                   hand); the policy's plan and any in-flight request are DISCARDED (the
                   model outputs delta actions, so a chunk planned before your correction
                   would snap the arm back if resumed).
    RELEASE     -> position mode again (Torque_Enable=1; firmware holds the current pose).
                   With AUTOSAVE_ON_RELEASE (default, single-intervention protocol) the
                   episode then ends and SAVES automatically. With it off, the policy is
                   resumed instead via a prefix-free COLD-START inference from the new state.

Recording never stops during an episode: one frame per control tick in every phase, with a
per-frame `is_intervention` feature (1.0 while the human holds the arm, else 0.0) and a
sidecar meta/dagger_interventions.json holding the exact press/release/resume frame indices
per episode.

A finished episode is first PARKED for review: while idle, LEFT ARROW deletes it (e.g. a
clean rollout you saved with SPACE but don't want, since only corrected episodes matter);
it is committed to the dataset when the next episode starts or the script exits. The full
key reference is reprinted at every state change; summary (global via pynput, no window
focus needed; the pedal is a keyboard device):
    pedal press ('a')    episode running: start intervening | idle: free-move the arm
    pedal release ('b')  end intervention -> auto-save | idle: hold pose again
    SPACE                idle: start an episode (START_DELAY_S setup wait, then a BEEP as
                         the robot starts -- collaboration cue) | running: stop + SAVE now
    LEFT ARROW           running: stop + DISCARD | idle: DELETE the parked episode
    ESC                  quit (parked + in-progress episodes are saved first)
    v                    toggle the live camera window
Headless/no-pynput fallback (terminal keys): s=start/save  d=discard/delete  i=fake pedal
v  q.

DATASET (LeRobot v2.1 via the /home/iris/lerobot fork -- record with THAT fork, its
info.json trossen_subversion is required to reopen the dataset locally):
    observation.images.{cam_high,cam_left_wrist,cam_right_wrist}   video
    observation.state / action    float32 (14,) ABSOLUTE joints -- right arm in dims 7:14,
                                  left-arm dims 0:7 zero (matches training's AlohaInputs +
                                  MaskLeftArmActions; do NOT record deltas)
    is_intervention               float32 (1,)
Training notes: openpi's pi05_trossen_memory reads $HF_LEROBOT_HOME/<repo_id>; merge these
episodes there (merge_lerobot_datasets.py), recompute norm stats, and mind the prompt source
(subtask_labels.json / default_prompt -- the LeRobot task string is NOT the training prompt).
DATASET_FPS below == CONTROL_HZ, the true wall-clock frame rate of the recording.

Run on the ROBOT computer in the `lerobot` conda env:
    conda activate lerobot && python scripts/dagger_record_rtc.py
"""

import io
import itertools
import json
import queue
import select
import shutil
import subprocess
import sys
import tempfile
import termios
import time
import tty
import uuid
import wave
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock, Thread

import cv2
import numpy as np
import requests
import torch

# Point Python at the LeRobot hardware fork (same as the deployment client).
LEROBOT_FORK_PATH = "/home/iris/lerobot"
if LEROBOT_FORK_PATH and LEROBOT_FORK_PATH not in sys.path:
    sys.path.insert(0, LEROBOT_FORK_PATH)

from lerobot.common.datasets import lerobot_dataset as _lerobot_dataset
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.robot_devices.robots.configs import TrossenAIRightArmOnlyRobotConfig
from lerobot.common.robot_devices.robots.utils import make_robot_from_config

# Shared pop-up visualizers (viz_utils.py lives next to this file in scripts/).
from viz_utils import InputVisualizer, LiveCameraVisualizer, display_available

# ----------------------------- CONFIG -----------------------------
SERVER_URL = "http://10.79.12.149:8003/infer"   # GPU node running qwen_action_expert_server.py
HL_CAM = "cam_high"
WRIST_CAM = "cam_right_wrist"
FOLLOWER_ARM = "right"          # the arm the pedal toggles between position/gravity-comp
# --- these MUST match the server's checkpoint (see /health) ---
IMAGE_HISTORY = True            # True: send NUM_FRAMES strided cam_high frames; False: 1 frame
SUBTASK = "pick up yellow block"  # subtask-input mode input; ignored by predict-subtask servers
NUM_FRAMES = 10                 # history length (history mode only)
STRIDE_TICKS = 10               # one cam_high history frame every 10 ticks (training stride)
CONTROL_HZ = 25                 # control cadence; one action / one recorded frame per tick
# --- RTC scheduling (same semantics as client_action_expert_rtc_use_me.py) ---
DELAY_STEPS = 7                 # ticks of inference latency planned around (<= trained prefix len)
EXEC_HORIZON = 7                # ticks between replans (>= DELAY_STEPS)
NUM_FLOW_STEPS = 10
REQUEST_TIMEOUT_S = 60.0
JPEG_QUALITY = 90
FRAME_DEDUP = True
SESSION_ID = uuid.uuid4().hex[:16]
LOG_TIMING = True
# --- pedal (PCsensor FootSwitch emulating a keyboard; same chars as control_robot_pedal.py) ---
PEDAL_PRESS_CHAR = "a"          # typed by the pedal on physical PRESS
PEDAL_RELEASE_CHAR = "b"        # typed by the pedal on physical RELEASE
RESUME_SETTLE_TICKS = 6         # recorded hold ticks after torque-on before the resume request
# One intervention per episode: releasing the pedal ends AND saves the episode immediately
# (the policy is not resumed). Set False to return to multi-intervention episodes, where
# release hands control back to the policy and SPACE ends the episode.
AUTOSAVE_ON_RELEASE = True
# --- human-robot collaboration cues ---
# SPACE -> wait START_DELAY_S (human gets into position; the first observation is captured
# AFTER the wait so the VLA sees the human's final pose) -> cold-start inference -> BEEP
# (audible episode-start cue) -> the robot starts moving.
START_DELAY_S = 1.0
START_SOUND = True
# --- dataset ---
REPO_ID = "trossen_data/dagger_block"
DATA_ROOT = Path(__file__).resolve().parent / "recordings" / "dagger"  # + timestamp subdir
TASK_DESCRIPTION = SUBTASK      # LeRobot per-frame task string (required; not the train prompt)
DATASET_FPS = CONTROL_HZ        # honest wall-clock rate of the recorded frames (see docstring)
IMAGE_WRITER_THREADS_PER_CAM = 4
VIDEO_CODEC = "libx264"         # this machine's ffmpeg lacks libsvtav1 (the fork's default);
VIDEO_CRF = 23                  # h264 needs a lower crf than av1 for comparable quality
STATE_DIM = 14                  # training expects 14-dim (left arm zeros, right arm in 7:14)
RIGHT_SLICE = slice(7, 14)
# --- visualization ---
SHOW_LIVE_VIZ = True            # smooth real-time camera monitor ([v] toggles)
SHOW_INPUT_VIZ = False          # the strided "what the model sees" window (jittery by design)
# ------------------------------------------------------------------

# encode_episode_videos hardcodes the fork's default encoder (libsvtav1), which this
# machine's ffmpeg does not have -- route every episode encode through VIDEO_CODEC instead.
_orig_encode_video_frames = _lerobot_dataset.encode_video_frames
_lerobot_dataset.encode_video_frames = lambda *a, **kw: _orig_encode_video_frames(
    *a, **{"vcodec": VIDEO_CODEC, "crf": VIDEO_CRF, **kw})

# ids the server is known to have cached (see the RTC client; single-worker => serialized).
_sent_ids: set[int] = set()


@dataclass
class Plan:
    start_step: int      # global control step that this chunk's slot 0 corresponds to
    raw: np.ndarray      # (H, 7) ABSOLUTE joint actions


@dataclass
class ChunkResult:
    raw: np.ndarray
    subtask: str | None
    round_trip_ms: float
    server_total_ms: float | None
    frames_uploaded: int = -1


@dataclass
class PendingRequest:
    future: Future
    anchor: int | None   # None => resume cold start (anchored at install time)
    launch_step: int
    prefix: np.ndarray | None = field(repr=False, default=None)


def cam_rgb(obs, cam: str) -> np.ndarray:
    return obs[f"observation.images.{cam}"].detach().cpu().numpy().astype(np.uint8)


def encode_jpeg(rgb: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return enc.tobytes()


def post_request(session, cam_high_frames, frame_ids, wrist_rgb, state7, action_prefix) -> ChunkResult:
    """Encode + POST one observation (worker thread). Same protocol as the RTC client:
    `action_prefix` is (d, 7) ABSOLUTE joints or None (cold start); `frame_ids` enables the
    server-side frame cache (409 => re-send everything once)."""
    data = {
        "state": json.dumps([float(x) for x in np.asarray(state7).reshape(-1)]),
        "subtask": SUBTASK,
        "run_fast": "false",
        "num_flow_steps": str(NUM_FLOW_STEPS),
        "debug": "false",
    }
    if action_prefix is not None and len(action_prefix) > 0:
        data["action_prefix"] = json.dumps(np.asarray(action_prefix, dtype=np.float32).tolist())

    def _post(upload_pairs, extra):
        files = [("cam_high", (f"f{i}.jpg", io.BytesIO(encode_jpeg(frame)), "image/jpeg"))
                 for i, (_, frame) in enumerate(upload_pairs)]
        files.append(("wrist", ("wrist.jpg", io.BytesIO(encode_jpeg(wrist_rgb)), "image/jpeg")))
        return session.post(SERVER_URL, files=files, data={**data, **extra},
                            timeout=REQUEST_TIMEOUT_S), len(upload_pairs)

    t0 = time.perf_counter()
    if frame_ids is None:
        resp, n_up = _post([(None, f) for f in cam_high_frames], {})
        resp.raise_for_status()
    else:
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
            print("  frame cache miss on server; re-sending the full window.")
            _sent_ids.clear()
            pairs = new_pairs(skip_known=False)
            resp, n_up = _post(pairs, dedup_extra(pairs))
        resp.raise_for_status()
        _sent_ids.clear()
        _sent_ids.update(map(int, frame_ids))

    j = resp.json()
    timing = j.get("timing_ms", {})
    return ChunkResult(
        raw=np.asarray(j["actions"], dtype=np.float32),
        subtask=j.get("subtask"),
        round_trip_ms=(time.perf_counter() - t0) * 1e3,
        server_total_ms=timing.get("total_ms"),
        frames_uploaded=j.get("frames_uploaded", n_up),
    )


def extract_prefix(plan: Plan, anchor: int, delay_steps: int) -> np.ndarray:
    """ABSOLUTE (delay_steps, 7) prefix starting at absolute step `anchor` (see RTC client)."""
    p0 = max(0, min(anchor - plan.start_step, len(plan.raw)))
    chunk = plan.raw[p0 : p0 + delay_steps]
    if len(chunk) < delay_steps:
        print(f"  WARNING: prefix window ran off the plan ({len(chunk)}/{delay_steps}); padding.")
        last = chunk[-1:] if len(chunk) else plan.raw[-1:]
        chunk = np.concatenate([chunk, np.repeat(last, delay_steps - len(chunk), axis=0)], axis=0)
    return chunk


# ----------------------------- input (pedal + command keys) -----------------------------

class TokenInput:
    """Funnels the pedal and the command keys into one thread-safe token queue.

    Primary source: a GLOBAL pynput listener (the pedal is a keyboard device typing
    PEDAL_PRESS_CHAR / PEDAL_RELEASE_CHAR; command keys work regardless of window focus).
    Fallback (headless / pynput missing): the terminal in cbreak mode.
    Tokens: pedal_press, pedal_release, toggle_episode, discard, quit, viz.
    """

    def __init__(self):
        self._q: queue.Queue[str] = queue.Queue()
        self._listener = None
        self._term_fd = None
        self._term_old = None
        self._fake_down = False
        # cbreak the terminal whenever we have one: in fallback mode it's the key source; in
        # pynput mode it swallows the chars the pedal "types" into the shell (no echo, no
        # leftover 'a'/'b' spam after exit).
        if sys.stdin.isatty():
            self._term_fd = sys.stdin.fileno()
            self._term_old = termios.tcgetattr(self._term_fd)
            tty.setcbreak(self._term_fd)
        if display_available():
            try:
                from pynput import keyboard

                def on_press(key):
                    char = getattr(key, "char", None)
                    char = char.lower() if char else None   # pedal must work with CapsLock on
                    if char == PEDAL_PRESS_CHAR:
                        self._q.put("pedal_press")
                    elif char == PEDAL_RELEASE_CHAR:
                        self._q.put("pedal_release")
                    elif char == "v":
                        self._q.put("viz")
                    elif key == keyboard.Key.space:
                        self._q.put("toggle_episode")
                    elif key == keyboard.Key.left:
                        self._q.put("discard")
                    elif key == keyboard.Key.esc:
                        self._q.put("quit")

                self._listener = keyboard.Listener(on_press=on_press)
                self._listener.start()
                print("[input] pynput listener active: pedal a/b | SPACE start/save | "
                      "LEFT discard | ESC quit | v live-viz")
                return
            except Exception as e:  # noqa: BLE001
                print(f"[input] pynput unavailable ({e}); falling back to terminal keys.")
        if self._term_fd is None:
            print("[input] WARNING: no display and no tty -- only the viz windows / Ctrl+C "
                  "can stop the script, and the pedal will NOT work.")
        else:
            print("[input] terminal keys: s=start/save  d=discard  i=fake-pedal toggle  "
                  "v=live-viz  q=quit  (pedal chars a/b also work with terminal focus)")

    _TERM_MAP = {"s": "toggle_episode", "d": "discard", "q": "quit", "v": "viz",
                 PEDAL_PRESS_CHAR: "pedal_press", PEDAL_RELEASE_CHAR: "pedal_release"}

    def _poll_terminal(self):
        if self._term_fd is None:
            return
        while select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1).lower()
            if self._listener is not None:
                continue        # pynput handles input; just swallow the pedal's typed chars
            if ch == "i":       # fake pedal: toggle press/release for testing without hardware
                self._fake_down = not self._fake_down
                self._q.put("pedal_press" if self._fake_down else "pedal_release")
            elif ch in self._TERM_MAP:
                self._q.put(self._TERM_MAP[ch])

    def drain(self) -> list[str]:
        """All tokens received since the last call (control-loop thread)."""
        self._poll_terminal()
        tokens = []
        while True:
            try:
                tokens.append(self._q.get_nowait())
            except queue.Empty:
                return tokens

    def close(self):
        if self._listener is not None:
            self._listener.stop()
        if self._term_fd is not None and self._term_old is not None:
            termios.tcsetattr(self._term_fd, termios.TCSADRAIN, self._term_old)


# ----------------------------- start-cue sound -----------------------------

_beep_wav: Path | None = None


def _ensure_beep_wav() -> Path | None:
    """Synthesize the start-cue tone once (0.35 s, 880 Hz, 20 ms fades -- no click)."""
    global _beep_wav
    if _beep_wav is not None and _beep_wav.exists():
        return _beep_wav
    try:
        sr, dur, freq = 44100, 0.35, 880.0
        t = np.arange(int(sr * dur)) / sr
        fade = np.minimum(1.0, np.minimum(t, dur - t) / 0.02)
        pcm = (np.sin(2 * np.pi * freq * t) * fade * 0.85 * 32767).astype(np.int16)
        path = Path(tempfile.gettempdir()) / "dagger_start_beep.wav"
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm.tobytes())
        _beep_wav = path
        return path
    except Exception as e:  # noqa: BLE001
        print(f"[sound] could not synthesize the beep: {e}")
        return None


def play_start_sound() -> None:
    """Audible cue for the human collaborator that the policy is starting. Non-blocking;
    degrades to the terminal bell when no audio player/output is available."""
    if not START_SOUND:
        return
    wav = _ensure_beep_wav()
    if wav is not None:
        for player in ("paplay", "aplay"):
            exe = shutil.which(player)
            if exe:
                try:
                    subprocess.Popen([exe, str(wav)], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                    return
                except Exception:  # noqa: BLE001
                    continue
    print("\a", end="", flush=True)


# ----------------------------- gravity compensation -----------------------------

def set_gravity_comp(robot, on: bool) -> None:
    """Pedal-controlled follower mode switch (trossen_arm_driver.write 'Torque_Enable'):
    on  -> external_effort mode + zero efforts: gravity-compensated, hand-movable, no drop;
    off -> position mode: firmware re-seeds the goal to the CURRENT pose, so the arm holds
    where the human left it (no jump; same sequence disable_teleoperation relies on)."""
    robot.follower_arms[FOLLOWER_ARM].write("Torque_Enable", 0 if on else 1)


# ----------------------------- dataset -----------------------------

JOINT_NAMES = [f"{side}_joint_{i}" for side in ("left", "right") for i in range(7)]


def build_features(robot) -> dict:
    """Explicit feature dict: the robot would yield 7-dim motor features (right arm only),
    but training expects 14-dim with the right arm in dims 7:14 -- so state/action are
    declared 14-dim here and every frame is padded by the recorder."""
    features = {
        "observation.state": {"dtype": "float32", "shape": (STATE_DIM,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (STATE_DIM,), "names": JOINT_NAMES},
        "is_intervention": {"dtype": "float32", "shape": (1,), "names": ["is_intervention"]},
    }
    for key, ft in robot.camera_features.items():   # observation.images.* with (h, w, c) shapes
        features[key] = {"dtype": "video", **ft}
    return features


class AsyncEpisodeSaver:
    """Background save_episode_batch worker (parquet + ffmpeg encoding off the control loop);
    mirrors control_robot_pedal.py's AsyncEpisodeBatchSaver."""

    def __init__(self, dataset: LeRobotDataset):
        self.dataset = dataset
        self._q: queue.Queue = queue.Queue()
        self._exc = None
        self._lock = Lock()
        self._thread = Thread(target=self._run, name="dagger-episode-saver", daemon=True)
        self._thread.start()

    def enqueue(self, episode_buffers: list[dict]) -> None:
        self.raise_if_failed()
        self._q.put(list(episode_buffers))

    def close(self) -> None:
        self._q.put(None)
        self._q.join()
        self._thread.join()
        self.raise_if_failed()

    def raise_if_failed(self) -> None:
        # The exception is deliberately NOT cleared: after a failed save, meta.total_episodes
        # stops matching the buffer indices, so every later save would fail too -- the whole
        # session must stop (the caller treats this as fatal).
        with self._lock:
            if self._exc is not None:
                raise RuntimeError("Background episode save failed.") from self._exc

    def _run(self) -> None:
        while True:
            batch = self._q.get()
            try:
                if batch is None:
                    return
                indices = [int(b["episode_index"]) for b in batch]
                print(f"  [saver] encoding episode(s) {indices} in the background...")
                self.dataset.episode_batch = batch
                self.dataset.save_episode_batch()
                print(f"  [saver] episode(s) {indices} saved.")
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    if self._exc is None:
                        self._exc = exc
                print(f"  [saver] FAILED: {exc}")
            finally:
                self._q.task_done()


class EpisodeRecorder:
    """Owns the LeRobot dataset + the per-episode intervention sidecar."""

    def __init__(self, robot, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.root = root / datetime.now().strftime("%Y%m%d_%H%M%S")  # create() needs a fresh dir
        self.dataset = LeRobotDataset.create(
            REPO_ID,
            DATASET_FPS,
            root=self.root,
            robot_type=robot.robot_type,
            features=build_features(robot),
            use_videos=True,
            image_writer_threads=IMAGE_WRITER_THREADS_PER_CAM * len(robot.cameras),
        )
        self.saver = AsyncEpisodeSaver(self.dataset)
        self.episode_index = self.dataset.meta.total_episodes
        self.segments: list[dict] = []       # current episode's intervention segments
        self.start_wall: float | None = None
        self._parked: tuple | None = None    # (episode_buffer, index, sidecar entry)
        self.sidecar_path = self.root / "meta" / "dagger_interventions.json"
        self.sidecar: dict = {}

    @property
    def frame_count(self) -> int:
        return self.dataset.episode_buffer["size"]

    def begin_episode(self) -> None:
        self.segments = []
        self.start_wall = time.time()

    def add(self, obs: dict, action7: np.ndarray, intervening: bool) -> None:
        state14 = np.zeros(STATE_DIM, dtype=np.float32)
        state14[RIGHT_SLICE] = obs["observation.state"].detach().cpu().numpy().astype(np.float32)
        action14 = np.zeros(STATE_DIM, dtype=np.float32)
        action14[RIGHT_SLICE] = np.asarray(action7, dtype=np.float32)
        frame = {
            "observation.state": state14,
            "action": action14,
            "is_intervention": np.array([1.0 if intervening else 0.0], dtype=np.float32),
            "task": TASK_DESCRIPTION,
        }
        for key in self.dataset.meta.camera_keys:
            frame[key] = obs[key]           # torch uint8 HWC; add_frame converts
        self.dataset.add_frame(frame)

    def mark_press(self) -> None:
        self.segments.append({"press_frame": self.frame_count})

    def mark_release(self) -> None:
        if self.segments and "release_frame" not in self.segments[-1]:
            self.segments[-1]["release_frame"] = self.frame_count

    def mark_resume(self) -> None:
        if self.segments and "resume_frame" not in self.segments[-1]:
            self.segments[-1]["resume_frame"] = self.frame_count

    def park_episode(self) -> bool:
        """Finish the current episode but hold it for review: it can still be DELETED (left
        arrow while idle) until the next episode starts or the script exits, at which point
        commit_parked() queues it for the real background save. Returns False if empty.
        A finished episode can't be un-saved once committed (parquet/mp4/meta surgery), so
        the review window is the parked period."""
        n = self.frame_count
        if n == 0:
            return False
        self.commit_parked()                # at most one parked episode at a time
        duration = time.time() - self.start_wall if self.start_wall else 0.0
        actual_hz = n / duration if duration > 0 else 0.0
        entry = {
            "task": TASK_DESCRIPTION,
            "subtask": SUBTASK,
            "num_frames": n,
            "control_hz": CONTROL_HZ,
            "dataset_fps": DATASET_FPS,
            "actual_hz": round(actual_hz, 3),
            "interventions": self.segments,
            "server_url": SERVER_URL,
            "rtc": {"delay_steps": DELAY_STEPS, "exec_horizon": EXEC_HORIZON,
                    "num_frames": NUM_FRAMES, "stride_ticks": STRIDE_TICKS},
            "started_at": datetime.fromtimestamp(self.start_wall).isoformat(timespec="seconds"),
        }
        self._parked = (self.dataset.episode_buffer, self.episode_index, entry)
        self.episode_index += 1
        self.dataset.episode_buffer = self.dataset.create_episode_buffer(
            episode_index=self.episode_index)
        print(f"  episode_{self.episode_index - 1:06d}: {n} frames, {duration:.1f}s "
              f"(actual {actual_hz:.1f} Hz vs fps {DATASET_FPS}), "
              f"{len(self.segments)} intervention(s)")
        if actual_hz and abs(actual_hz - DATASET_FPS) / DATASET_FPS > 0.05:
            print(f"  WARNING: actual rate {actual_hz:.1f} Hz deviates >5% from the declared "
                  f"fps={DATASET_FPS}; the video will play back off-speed.")
        return True

    @property
    def parked_index(self) -> int | None:
        return self._parked[1] if self._parked else None

    def commit_parked(self) -> None:
        """Queue the parked episode for the background save; after this it is permanent."""
        if not self._parked:
            return
        buffer, idx, entry = self._parked
        self._parked = None
        self.sidecar[f"episode_{idx:06d}"] = entry
        self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        self.sidecar_path.write_text(json.dumps(self.sidecar, indent=2) + "\n")
        self.saver.enqueue([buffer])
        print(f"  episode_{idx:06d} committed -> saving in background.")

    def delete_parked(self) -> bool:
        """Drop the parked (most recent, uncommitted) episode. Only valid while IDLE, i.e.
        the live buffer is empty -- its index is rewound so numbering stays consecutive."""
        if not self._parked:
            return False
        _, idx, entry = self._parked
        self._parked = None
        self.dataset._wait_image_writer()
        self.dataset._delete_episode_images([idx])
        self.episode_index = idx
        self.dataset.episode_buffer = self.dataset.create_episode_buffer(episode_index=idx)
        print(f"  episode_{idx:06d} DELETED ({entry['num_frames']} frames, "
              f"{len(entry['interventions'])} intervention(s)).")
        return True

    def discard_episode(self) -> None:
        n = self.frame_count
        self.dataset._wait_image_writer()
        self.dataset._delete_episode_images([self.episode_index])
        self.dataset.episode_buffer = self.dataset.create_episode_buffer(
            episode_index=self.episode_index)
        print(f"  in-progress episode discarded ({n} frames dropped).")

    def close(self) -> None:
        self.commit_parked()
        print("Waiting for background episode saves (video encoding)...")
        self.saver.close()


# ----------------------------- main -----------------------------

# Control states. IDLE: between episodes (pedal = free-move for scene reset). POLICY: RTC
# rollout executing + recording. INTERVENE: gravity comp, human corrects, recording.
# SETTLE: torque back on, recording hold frames for RESUME_SETTLE_TICKS. RESUME: cold-start
# request in flight, recording hold frames until the fresh chunk installs.
IDLE, POLICY, INTERVENE, SETTLE, RESUME = "IDLE", "POLICY", "INTERVENE", "SETTLE", "RESUME"


def main():
    if EXEC_HORIZON < DELAY_STEPS:
        raise ValueError("EXEC_HORIZON must be >= DELAY_STEPS for the RTC pipeline to stay ahead.")

    print("Initializing Trossen robot and cameras...")
    robot_config = TrossenAIRightArmOnlyRobotConfig(
        max_relative_target=None,
        min_time_to_move_multiplier=4.0,
        camera_interface="opencv",
    )
    robot = make_robot_from_config(robot_config)
    robot.connect()                       # NOTE: the arm moves to its home pose here
    print("Warming cameras (first frame can take ~2 s)...")
    robot.capture_observation()

    recorder = EpisodeRecorder(robot, DATA_ROOT)
    print(f"Dataset: {recorder.root}  (repo_id {REPO_ID}, fps {DATASET_FPS})")

    session = requests.Session()
    executor = ThreadPoolExecutor(max_workers=1)
    period = 1.0 / CONTROL_HZ

    history: deque = deque(maxlen=NUM_FRAMES)
    history_ids: deque = deque(maxlen=NUM_FRAMES)
    next_frame_id = itertools.count(0)

    def prime_history(obs) -> None:
        """(Re)prime the strided window like a cold start: NUM_FRAMES copies of the current
        frame sharing ONE freshly minted id (ids must stay monotone for the dedup cache)."""
        frame, fid = cam_rgb(obs, HL_CAM), next(next_frame_id)
        history.clear(), history_ids.clear()
        history.extend([frame] * NUM_FRAMES)
        history_ids.extend([fid] * NUM_FRAMES)

    def request_args(obs):
        frames = list(history) if IMAGE_HISTORY else [cam_rgb(obs, HL_CAM)]
        ids = list(history_ids) if (IMAGE_HISTORY and FRAME_DEDUP) else None
        state7 = obs["observation.state"].detach().cpu().numpy().astype(np.float32)
        return frames, ids, cam_rgb(obs, WRIST_CAM), state7

    input_viz = InputVisualizer(history_mode=IMAGE_HISTORY, stride_ticks=STRIDE_TICKS,
                                control_hz=CONTROL_HZ, window="dagger - model input",
                                enabled=SHOW_INPUT_VIZ, show_model_input=False)
    live_viz = LiveCameraVisualizer(window="dagger - live cameras", enabled=SHOW_LIVE_VIZ)
    tokens: TokenInput | None = None    # created inside the try: its cbreak needs the finally

    state = IDLE
    plan: Plan | None = None
    pending: PendingRequest | None = None
    global_step = 0
    next_request_step = 0
    settle_left = 0
    wait_ticks = 0
    idle_grav = False           # gravity comp active while IDLE (scene repositioning)
    current_subtask = None
    last_rtt = None
    quit_requested = False

    def drop_pending() -> None:
        nonlocal pending
        pending = None          # worker may still finish the HTTP call; its result is ignored

    def start_episode() -> None:
        """Synchronous cold start, then flip to POLICY. First recorded frame = first rollout tick."""
        nonlocal state, plan, global_step, next_request_step, idle_grav, current_subtask, last_rtt
        recorder.commit_parked()    # starting the next episode ends the review window
        if idle_grav:
            set_gravity_comp(robot, False)
            idle_grav = False
            time.sleep(0.5)     # let the pose settle before the policy observes it
        if START_DELAY_S > 0:
            # Collaboration setup window: capture happens AFTER this wait, so the VLA's
            # first observation includes the human in their final position.
            print(f"\n=== episode_{recorder.episode_index:06d}: get in position -- "
                  f"starting in {START_DELAY_S:.0f} s... ===")
            time.sleep(START_DELAY_S)
        print(f"=== episode_{recorder.episode_index:06d}: cold-start inference... ===")
        obs0 = robot.capture_observation()
        prime_history(obs0)
        try:
            # Through the single-worker executor (not directly): a request dropped by a quick
            # stop-episode may still be in flight, and _sent_ids/the server frame cache must
            # only ever be touched by one request at a time.
            frames, ids, wrist, st7 = request_args(obs0)
            result = executor.submit(post_request, session, frames, ids, wrist, st7, None).result()
        except Exception as e:  # noqa: BLE001
            print(f"  cold start FAILED ({e}); staying idle.")
            return
        plan = Plan(start_step=0, raw=result.raw)
        global_step = 0
        next_request_step = EXEC_HORIZON
        current_subtask, last_rtt = result.subtask, result.round_trip_ms
        recorder.begin_episode()
        play_start_sound()          # audible cue: the robot starts moving NOW
        state = POLICY
        print(f"  rolling out (subtask={result.subtask!r}).")
        print_controls()

    def stop_episode(save: bool) -> None:
        """Leave any active state safely: torque on, drop the plan/pending, save or discard."""
        nonlocal state, plan, idle_grav
        if state == INTERVENE:
            set_gravity_comp(robot, False)
            recorder.mark_release()
        drop_pending()
        plan = None
        idle_grav = False
        if save:
            if not recorder.park_episode():
                print("  empty episode; nothing to save.")
        else:
            recorder.discard_episode()
        state = IDLE
        print_controls()

    def print_controls() -> None:
        """Full key reference, reprinted at every state change so operation is always clear."""
        parked = recorder.parked_index
        print(f"\n---- [{state}] controls " + "-" * 46)
        if state == IDLE:
            print(f"  PEDAL hold    free-move the arm (gravity comp) for scene reset\n"
                  f"  SPACE         start episode_{recorder.episode_index:06d} "
                  f"({START_DELAY_S:.0f} s setup wait, beep = robot starts)"
                  + (f"  (commits parked episode_{parked:06d})" if parked is not None else ""))
            if parked is not None:
                print(f"  LEFT ARROW    DELETE parked episode_{parked:06d} "
                      "(e.g. no intervention was needed)")
            else:
                print("  LEFT ARROW    (nothing parked to delete)")
        else:
            print("  PEDAL press   intervene: gravity comp, correct the robot by hand"
                  if state == POLICY else
                  ("  PEDAL release end intervention -> episode auto-saves"
                   if (state == INTERVENE and AUTOSAVE_ON_RELEASE) else
                   "  PEDAL release end intervention -> policy resumes"))
            print("  SPACE         stop + save this episode now\n"
                  "  LEFT ARROW    stop + DISCARD this episode (not saved)")
        print("  ESC           quit (parked + in-progress episodes are saved first)\n"
              "  v             toggle live camera view\n" + "-" * 70)

    try:
        tokens = TokenInput()

        # One synchronous warmup request so server/checkpoint problems surface before
        # recording -- inside the try so a dead server still restores terminal + robot.
        print(f"Warming up the server (subtask={SUBTASK!r})...")
        obs = robot.capture_observation()
        prime_history(obs)
        warm = post_request(session, *request_args(obs), None)
        horizon = len(warm.raw)
        if EXEC_HORIZON + DELAY_STEPS > horizon:
            raise ValueError(f"EXEC_HORIZON + DELAY_STEPS ({EXEC_HORIZON + DELAY_STEPS}) "
                             f"exceeds chunk length ({horizon}).")
        est_ticks = warm.round_trip_ms / 1e3 * CONTROL_HZ
        print(f"  chunk={warm.raw.shape} | subtask={warm.subtask!r} | "
              f"round_trip={warm.round_trip_ms:.0f} ms (~{est_ticks:.0f} ticks vs d={DELAY_STEPS})")
        if est_ticks > DELAY_STEPS:
            print(f"  WARNING: latency exceeds DELAY_STEPS={DELAY_STEPS}; the rollout will "
                  "stop-and-wait every chunk. Raise DELAY_STEPS (<= trained prefix length).")
        # Warm the prefix path too, so a checkpoint that rejects action_prefix fails HERE, not
        # at the first mid-episode replan (nothing is executed; the arm is still at home).
        warm_prefix = warm.raw[EXEC_HORIZON:EXEC_HORIZON + DELAY_STEPS]
        obs = robot.capture_observation()
        warm2 = post_request(session, *request_args(obs), warm_prefix)
        clamp = float(np.max(np.abs(warm2.raw[:DELAY_STEPS] - warm_prefix)))
        print(f"  prefix warmup: round_trip={warm2.round_trip_ms:.0f} ms | clamp={clamp:.1e}"
              + ("" if clamp < 1e-2 else "  <-- WARNING: prefix not clamped; RTC checkpoint?"))
        current_subtask, last_rtt = warm.subtask, warm.round_trip_ms
        if START_SOUND:
            play_start_sound()
            print("  (audio check: you should have just heard the episode-start beep)")

        print_controls()

        while not quit_requested:
            loop_start = time.perf_counter()

            # ---- input tokens (from the pynput thread / terminal), applied on THIS thread ----
            for token in tokens.drain():
                if token == "quit":
                    quit_requested = True
                elif token == "viz":
                    print(f"  live camera view {'ON' if live_viz.toggle() else 'OFF'}.")
                elif token == "toggle_episode":
                    if state == IDLE:
                        start_episode()
                    else:
                        stop_episode(save=True)
                elif token == "discard":
                    if state != IDLE:
                        stop_episode(save=False)
                    elif not recorder.delete_parked():
                        print("  nothing to delete (no parked episode).")
                    else:
                        print_controls()
                elif token == "pedal_press":
                    if state == IDLE and not idle_grav:
                        set_gravity_comp(robot, True)
                        idle_grav = True
                        print("  [idle] gravity comp ON -- move the arm freely.")
                    elif state in (POLICY, SETTLE, RESUME):
                        drop_pending()      # stale plan/prefix would snap the arm back
                        plan = None
                        wait_ticks = 0      # an aborted stop-and-wait hold must not linger
                        set_gravity_comp(robot, True)
                        recorder.mark_press()
                        state = INTERVENE
                        print(f"  INTERVENTION started at frame {recorder.frame_count}.")
                        print_controls()
                elif token == "pedal_release":
                    if state == IDLE and idle_grav:
                        set_gravity_comp(robot, False)
                        idle_grav = False
                        print("  [idle] gravity comp OFF -- arm holds.")
                    elif state == INTERVENE:
                        if AUTOSAVE_ON_RELEASE:
                            # Single-intervention protocol: the correction is the end of the
                            # episode -- torque on, save, back to idle for the scene reset.
                            print(f"  intervention ended at frame {recorder.frame_count}; "
                                  "auto-saving the episode.")
                            stop_episode(save=True)
                        else:
                            set_gravity_comp(robot, False)
                            recorder.mark_release()
                            settle_left = RESUME_SETTLE_TICKS
                            state = SETTLE
                            print(f"  intervention ended at frame {recorder.frame_count}; "
                                  "resuming the policy...")
            if live_viz.quit_requested or input_viz.quit_requested:
                quit_requested = True
            if quit_requested:
                break

            # ---- IDLE: no recording; keep the live view fresh via async_read ----
            if state == IDLE:
                if live_viz.enabled:
                    try:
                        live_viz.render(robot.cameras[HL_CAM].async_read(),
                                        robot.cameras[WRIST_CAM].async_read(),
                                        top_label=f"cam_high | IDLE"
                                                  f"{' | GRAV-COMP' if idle_grav else ''}",
                                        wrist_label=f"right wrist | next: "
                                                    f"episode_{recorder.episode_index:06d}")
                    except Exception as e:  # noqa: BLE001 -- a monitor glitch must not stop us
                        print(f"  live view error, disabling it: {e}")
                        live_viz.enabled = False
                input_viz.pump()
                time.sleep(max(0.0, period - (time.perf_counter() - loop_start)))
                continue

            # ---- episode active: ONE observation per tick feeds control, recording, viz ----
            obs = robot.capture_observation()
            state7 = obs["observation.state"].detach().cpu().numpy().astype(np.float32)

            if state == SETTLE:
                settle_left -= 1
                if settle_left <= 0:
                    prime_history(obs)          # fresh window; old frames predate the correction
                    frames, ids, wrist, st7 = request_args(obs)
                    pending = PendingRequest(
                        future=executor.submit(post_request, session, frames, ids, wrist, st7, None),
                        anchor=None, launch_step=global_step)
                    state = RESUME
                action7 = state7                # torque on, arm holds; label = hold in place

            elif state == RESUME:
                action7 = state7
                if pending is not None and pending.future.done():
                    try:
                        result = pending.future.result()
                        plan = Plan(start_step=global_step, raw=result.raw)
                        next_request_step = global_step + EXEC_HORIZON
                        current_subtask, last_rtt = result.subtask, result.round_trip_ms
                        recorder.mark_resume()
                        state = POLICY
                        if LOG_TIMING:
                            print(f"  resumed with a fresh chunk (round_trip="
                                  f"{result.round_trip_ms:.0f} ms) at frame {recorder.frame_count}.")
                    except Exception as e:  # noqa: BLE001
                        print(f"  resume inference failed ({e}); retrying...")
                        settle_left = 1
                        state = SETTLE
                    finally:
                        pending = None

            elif state == INTERVENE:
                action7 = state7                # kinesthetic label = follower position readback

            if state == POLICY:
                # -- the RTC scheduler, same tick order as the deployment client --
                if pending is not None and pending.future.done():
                    try:
                        result = pending.future.result()
                        clamp = float(np.max(np.abs(result.raw[:DELAY_STEPS] - pending.prefix)))
                        plan = Plan(start_step=pending.anchor, raw=result.raw)
                        current_subtask, last_rtt = result.subtask, result.round_trip_ms
                        if LOG_TIMING:
                            print(f"  chunk ready: clamp={clamp:.1e} | round_trip="
                                  f"{result.round_trip_ms:.0f} ms | server={result.server_total_ms} ms | "
                                  f"frames_sent={result.frames_uploaded}")
                        if wait_ticks:
                            print(f"  chunk arrived after holding {wait_ticks} tick(s); resuming.")
                    except Exception as e:  # noqa: BLE001
                        print(f"  inference request failed, keeping current plan: {e}")
                    pending = None
                    wait_ticks = 0

                if pending is not None and global_step >= pending.anchor + DELAY_STEPS:
                    # STOP-AND-WAIT: hold the last committed action; global_step frozen.
                    hold_idx = (pending.anchor + DELAY_STEPS - 1) - plan.start_step
                    hold_idx = min(max(hold_idx, 0), len(plan.raw) - 1)
                    action7 = plan.raw[hold_idx]
                    robot.send_action(torch.from_numpy(np.asarray(action7, dtype=np.float32)))
                    if wait_ticks % CONTROL_HZ == 0:
                        print(f"  holding for late chunk (inference > {DELAY_STEPS} ticks)...")
                    wait_ticks += 1
                else:
                    # Strided history feed + input viz, single cadence (frozen during holds).
                    if IMAGE_HISTORY and global_step % STRIDE_TICKS == 0:
                        history.append(cam_rgb(obs, HL_CAM))
                        history_ids.append(next(next_frame_id))
                        input_viz.render(list(history), cam_rgb(obs, WRIST_CAM), current_subtask,
                                         tick=global_step, rtt_ms=last_rtt)
                    # Fire the next async replan; prefix = the actions we're committing to.
                    if pending is None and global_step >= next_request_step:
                        prefix = extract_prefix(plan, global_step, DELAY_STEPS)
                        frames, ids, wrist, st7 = request_args(obs)
                        pending = PendingRequest(
                            future=executor.submit(post_request, session, frames, ids, wrist,
                                                   st7, prefix),
                            anchor=global_step, launch_step=global_step, prefix=prefix)
                        next_request_step = global_step + EXEC_HORIZON
                    # Execute this step's action.
                    idx = min(max(global_step - plan.start_step, 0), len(plan.raw) - 1)
                    action7 = plan.raw[idx]
                    robot.send_action(torch.from_numpy(np.asarray(action7, dtype=np.float32)))
                    global_step += 1

            # ---- record this tick (every active phase) ----
            recorder.add(obs, action7, intervening=(state == INTERVENE))
            try:
                recorder.saver.raise_if_failed()
            except RuntimeError as e:
                # Fatal: after one failed save, episode indices no longer match the dataset
                # meta, so every later save would fail too -- don't let the operator keep
                # collecting unrecoverable episodes.
                print(f"  FATAL: {e}\n  Stopping the session. Raw frames remain under "
                      f"{recorder.root}/images for manual recovery.")
                quit_requested = True

            if live_viz.enabled:
                try:
                    live_viz.render(cam_rgb(obs, HL_CAM), cam_rgb(obs, WRIST_CAM),
                                    tick=recorder.frame_count,
                                    top_label=f"cam_high | {state} | REC "
                                              f"episode_{recorder.episode_index:06d}",
                                    wrist_label=f"right wrist | frame {recorder.frame_count}"
                                                f"{' | INTERVENING' if state == INTERVENE else ''}")
                except Exception as e:  # noqa: BLE001 -- a monitor glitch must not stop us
                    print(f"  live view error, disabling it: {e}")
                    live_viz.enabled = False
            input_viz.pump()

            elapsed = time.perf_counter() - loop_start
            if LOG_TIMING and elapsed > period and recorder.frame_count % CONTROL_HZ == 0:
                print(f"  control-loop overrun: {elapsed * 1000:.1f} ms > {period * 1000:.1f} ms")
            time.sleep(max(0.0, period - elapsed))

    except KeyboardInterrupt:
        print("\nStopping via Ctrl+C...")
    finally:
        # Save any in-progress episode, restore position mode, then release the hardware.
        try:
            if state != IDLE:
                stop_episode(save=True)
            elif idle_grav:
                set_gravity_comp(robot, False)
                time.sleep(0.5)
        except Exception as e:  # noqa: BLE001
            print(f"  cleanup warning: {e}")
        if tokens is not None:
            tokens.close()
        input_viz.close()
        live_viz.close()
        executor.shutdown(wait=False)
        print("Disconnecting robot (the arm will move home, then to its sleep pose)...")
        robot.disconnect()
        try:
            recorder.close()                # waits for background video encoding
        except Exception as e:  # noqa: BLE001
            print(f"  FINAL SAVE FAILED: {e}\n  Raw frames remain under {recorder.root}/images.")
        print(f"\nDone. {recorder.episode_index} episode(s) in {recorder.root}")


if __name__ == "__main__":
    main()
