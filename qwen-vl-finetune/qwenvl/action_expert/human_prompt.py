"""Human-demo prompt sampling and bounded serve-time storage.

The training dataset decodes a 30 fps clip, keeps every ``stride``-th frame, always
keeps the final grasp-outcome frame, and uniformly re-spaces the result when it is
longer than ``max_frames``. Serving must use the same rule. This module is deliberately
independent of FastAPI and the GPU model so the contract can be tested on CPU.
"""

from __future__ import annotations

import hashlib
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")


def validate_prompt_key(value: str, field: str) -> str:
    """Validate a cache-key component supplied by a remote client."""
    value = value.strip()
    if not _ID_RE.fullmatch(value):
        raise ValueError(
            f"{field} must be 1-128 characters from A-Z, a-z, 0-9, '_', '-', or '.'"
        )
    return value


def sample_prompt_indices(total: int, stride: int, max_frames: int) -> list[int]:
    """Return the exact frame indices used by ``_extract_human_prompt`` in training."""
    if total <= 0:
        raise ValueError("human prompt must contain at least one frame")
    if stride <= 0:
        raise ValueError(f"human prompt stride must be positive, got {stride}")
    if max_frames <= 0:
        raise ValueError(f"human prompt max_frames must be positive, got {max_frames}")

    idxs = list(range(0, total, stride))
    if idxs[-1] != total - 1:
        idxs.append(total - 1)
    if len(idxs) > max_frames:
        if max_frames == 1:
            # The outcome is the semantically critical frame. With only one slot it
            # wins over the initial frame, preserving the documented final-frame rule.
            idxs = [idxs[-1]]
        else:
            selected = np.linspace(0, len(idxs) - 1, max_frames)
            idxs = [idxs[int(round(pos))] for pos in selected]
    return idxs


def sample_prompt_frames(
    frames: Sequence[Image.Image], stride: int, max_frames: int
) -> tuple[list[Image.Image], list[int]]:
    """Sample decoded raw clip frames and return ``(frames, source_indices)``."""
    idxs = sample_prompt_indices(len(frames), stride, max_frames)
    return [frames[i] for i in idxs], idxs


def sampled_video_metadata(
    source_frame_count: int,
    source_fps: float,
    source_indices: Sequence[int],
) -> dict:
    """Transformers metadata for an already-sampled video frame list.

    The Qwen3-VL processor inserts textual ``<x.y seconds>`` markers before each temporal
    patch from this metadata. (It also derives ``second_per_grid_ts``; the local Qwen3
    timestamp RoPE helper intentionally ignores that scalar and relies on the textual
    markers.) Without metadata, a list of PIL frames silently defaults to consecutive
    frames at 24 fps, which is badly wrong for the V2 1-Hz prompt and especially its
    forced final frame.
    """
    source_frame_count = int(source_frame_count)
    source_fps = float(source_fps)
    indices = [int(i) for i in source_indices]
    if source_frame_count <= 0:
        raise ValueError("source_frame_count must be positive")
    if not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError(f"source_fps must be positive and finite, got {source_fps}")
    if not indices:
        raise ValueError("source_indices must not be empty")
    if any(i < 0 or i >= source_frame_count for i in indices):
        raise ValueError(
            f"source indices {indices} outside [0, {source_frame_count - 1}]"
        )
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError(f"source_indices must be strictly increasing, got {indices}")
    return {
        "total_num_frames": source_frame_count,
        "fps": source_fps,
        "frames_indices": indices,
    }


def history_video_metadata(num_frames: int, frame_stride: int, fps: float) -> dict:
    """Constant slot-time metadata for the robot cam_high history video (V3+).

    The history is a fixed strided grid ending at "now": slot k holds frame
    max(hist_min, t - (num_frames-1-k)*frame_stride). The stamps rendered from this
    metadata are SLOT times (k*frame_stride/fps seconds, oldest slot = 0.0), not
    content times: near the episode start clamped slots repeat the first frame but
    the ladder text stays identical, so every training sample -- and the serve-time
    frame deque primed with copies of the first frame -- renders the exact same
    timestamp text. (V3 default, 6 frames at stride 15 / 30 fps: slot times
    0.0/0.5/.../2.5 s, pair-averaged by the processor to <0.2>/<1.2>/<2.2 seconds>.)
    Serving must rebuild the identical ladder from the checkpoint stamp
    (image_history, num_frames, frame_stride).
    """
    return sampled_video_metadata(
        source_frame_count=(num_frames - 1) * frame_stride + 1,
        source_fps=fps,
        source_indices=[k * frame_stride for k in range(num_frames)],
    )


def prompt_digest(frames: Iterable[Image.Image]) -> str:
    """Short content digest for logs and client/server identity checks."""
    digest = hashlib.sha256()
    count = 0
    for frame in frames:
        rgb = frame.convert("RGB")
        digest.update(f"{rgb.width}x{rgb.height};".encode())
        digest.update(np.asarray(rgb, dtype=np.uint8).tobytes())
        count += 1
    if count == 0:
        raise ValueError("human prompt must contain at least one frame")
    return digest.hexdigest()[:16]


@dataclass(frozen=True)
class PromptRecord:
    session_id: str
    prompt_id: str
    frames: tuple[Image.Image, ...]
    source_frame_count: int
    source_indices: tuple[int, ...]
    digest: str
    created_at: float


class PromptStore:
    """Thread-safe LRU keyed by ``(session_id, prompt_id)``.

    A robot server normally has one active client, but a hard bound prevents abandoned
    prompts from accumulating raw camera frames forever during development.
    """

    def __init__(self, max_entries: int = 16):
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._records: OrderedDict[tuple[str, str], PromptRecord] = OrderedDict()
        self._lock = threading.Lock()

    def put(
        self,
        session_id: str,
        prompt_id: str,
        frames: Sequence[Image.Image],
        *,
        source_frame_count: int,
        source_indices: Sequence[int],
    ) -> tuple[PromptRecord, bool]:
        session_id = validate_prompt_key(session_id, "session_id")
        prompt_id = validate_prompt_key(prompt_id, "prompt_id")
        if not frames:
            raise ValueError("human prompt must contain at least one sampled frame")
        if len(frames) != len(source_indices):
            raise ValueError("source_indices must match the sampled frame count")
        # Validate the temporal provenance before accepting a record. The store does not
        # need an FPS itself, so use a harmless positive placeholder; serving supplies the
        # checkpoint-stamped source FPS when it builds processor metadata.
        sampled_video_metadata(source_frame_count, 1.0, source_indices)

        # Fully detach from upload buffers and lazy decoders before the request returns.
        stored = tuple(frame.convert("RGB").copy() for frame in frames)
        record = PromptRecord(
            session_id=session_id,
            prompt_id=prompt_id,
            frames=stored,
            source_frame_count=int(source_frame_count),
            source_indices=tuple(int(i) for i in source_indices),
            digest=prompt_digest(stored),
            created_at=time.time(),
        )
        key = (session_id, prompt_id)
        with self._lock:
            replaced = key in self._records
            self._records[key] = record
            self._records.move_to_end(key)
            while len(self._records) > self.max_entries:
                self._records.popitem(last=False)
        return record, replaced

    def get(self, session_id: str, prompt_id: str) -> PromptRecord:
        session_id = validate_prompt_key(session_id, "session_id")
        prompt_id = validate_prompt_key(prompt_id, "prompt_id")
        key = (session_id, prompt_id)
        with self._lock:
            record = self._records[key]
            self._records.move_to_end(key)
            return record

    def clear_session(self, session_id: str) -> int:
        session_id = validate_prompt_key(session_id, "session_id")
        with self._lock:
            keys = [key for key in self._records if key[0] == session_id]
            for key in keys:
                del self._records[key]
            return len(keys)

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)
