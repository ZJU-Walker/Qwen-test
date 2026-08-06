"""LeRobot-format dataset for flow-matching action-expert training on Qwen3-VL.

Mirrors the semantics of the openpi `pi05_trossen_memory` pipeline:
  - prompt is the time-aligned subtask label from `videos/chunk-000/subtask_labels.json`
    (fallback to a default prompt), pi0.5-style;
  - robot state is quantile-normalized to [-1, 1], discretized into 256 bins, and
    embedded in the text prompt ("Task: ..., State: 12 240 ...");
  - actions are optionally converted to deltas w.r.t. the current state (grippers stay
    absolute, mask 6,-1,6,-1 as in openpi's aloha/trossen configs), then
    quantile-normalized with stats computed from the dataset and cached to JSON;
  - images come from the cam_high video as a `num_frames`-frame history at
    `frame_stride`, matching the existing PrototypeRobotDataset.

Each item returns the usual Qwen3-VL tensors (input_ids, labels, position_ids,
pixel_values_videos, video_grid_thw) plus the normalized action chunk.
"""

import json
import os
import random
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_video

from qwenvl.data.ee_repr import EE_DIM, joints_to_ee
from qwenvl.data.data_processor import (
    IGNORE_INDEX,
    DataCollatorForSupervisedDataset,
    update_processor_pixels,
)
from qwenvl.data.rope2d import get_rope_index_3
from qwenvl.train.argument import DataArguments


@dataclass
class RobotDataArguments(DataArguments):
    # Comma-separated list of LeRobot dataset roots (each containing data/ and videos/).
    # Default matches the openpi pi05_trossen_memory config's merged dataset.
    robot_data_dirs: str = "/iris/projects/humanoid/trossen_data/0528_green_yellow_block_mem_merged_bk"
    camera: str = "cam_high"
    num_frames: int = 10
    frame_stride: int = 10
    # If False, feed ONLY the current cam_high frame as a single still image (no history
    # video); num_frames/frame_stride are then ignored. For ablating image-history context
    # (current-top + wrist = 2 images). The still is pre-resized to max_pixels like the wrist.
    image_history: bool = True
    # Extra single-frame wrist views at the CURRENT timestep t only (no history), added
    # as still images alongside the cam_high video. Comma-separated camera folder suffixes
    # (e.g. "cam_right_wrist" or "cam_right_wrist,cam_left_wrist"); empty = none.
    wrist_cameras: str = ""
    # Pixel budget for each wrist still. We pre-resize the frame to ~this many pixels
    # because apply_chat_template does NOT apply the image_processor's max_pixels to
    # pre-loaded PIL images (a raw 540x960 wrist frame would otherwise become ~420 tokens,
    # 9x a video frame). ~50176 -> ~45-64 tokens, comparable to one cam_high frame.
    wrist_max_pixels: int = 50176
    action_dim: int = 14
    action_horizon: int = 50
    # Which raw state/action dimensions to use (always the same for both). Python
    # slice/list syntax: "7:14" for the right arm, "0,2,5" for arbitrary dims,
    # None/"" for all. action_dim and delta_mask refer to the SELECTED dims.
    active_dims: Optional[str] = None
    use_delta_actions: bool = True
    # openpi aloha/trossen convention: joints are deltas w.r.t. current state, grippers
    # absolute. Applies to the selected dims (e.g. "6,-1" for a single 7-dof arm).
    delta_mask: str = "6,-1,6,-1"
    train_split: float = 0.75
    default_prompt: str = "waiting"
    norm_stats_path: Optional[str] = None
    # "joint": raw selected joint dims (historic behavior). "ee6d": convert the selected
    # 7-dim right arm [q0..q5, jaw] to the 10-dim end-effector representation
    # [xyz, 6D rotation (first two matrix columns), jaw] via forward kinematics at load
    # time (qwenvl/data/ee_repr.py). Requires action_dim 10 and delta_mask "9,-1"
    # (position + 6D naive-delta vs state, jaw absolute -- measured valid on this data,
    # see tests/probe_rotation_smoothness.py).
    action_space: str = "joint"
    # Skip episodes shorter than this many frames at scan time (0 = keep all). The 0731
    # set contains 1-4-frame aborted recordings that would otherwise be sampled as often
    # as full episodes.
    min_episode_len: int = 0
    # pi05-style train-time image augmentation, matched to what pi05 ACTUALLY runs
    # (openpi model.py preprocess_observation + augmax defaults): RandomCrop to 95% +
    # resize back + rotation U(-5,5) deg, always on, top-down camera only (history and
    # no-history stills); color jitter on ALL cameras fires with p=0.5 per camera draw
    # (augmax ColorJitter default -- pi05 does NOT jitter every image): brightness +
    # contrast + hue U(-0.1, 0.1). Saturation is omitted on purpose: augmax's saturation
    # jitter is a no-op (the adjusted value is computed and discarded), so real pi05
    # checkpoints never trained with it. Wrists get jitter only, like openpi. Parameters
    # are sampled ONCE per sample per camera and applied to every history frame
    # identically: a camera-pose/lighting shift is constant within a clip, and per-frame
    # parameter resampling would inject temporal shake the deployment camera never
    # produces.
    image_aug: bool = False
    image_aug_prob: float = 1.0
    # Feed the states at the image-history timesteps (all but the newest; its state IS the
    # current one) into the prompt as "Past states:", frame-aligned and edge-clamped exactly
    # like the frames (indices max(0, t - k*stride) -- near the episode start the first
    # state repeats). Input-shape contract: stamped into visual_budget.json; serve must match.
    state_history: bool = False
    # If True (knowledge-insulation co-training), the user turn asks a fixed question and
    # the assistant turn contains the time-aligned subtask label with LM supervision.
    # If False (default, pi0.5-style), the subtask label IS the prompt and there is no
    # language supervision (labels are fully masked).
    predict_subtask: bool = False
    subtask_question: str = "which colored block did the human hand point to?"
    # Gate out a leading subtask segment (by its label, e.g. "waiting") from episodes:
    # min_start moves to the segment's end+1, so its frames are never training timesteps
    # NOR history context -- unlike DAgger's is_intervention gate, image history and past
    # states are edge-clamped AT the cut (episode["hist_min"]), not at frame 0. Built for
    # the human-video-prompt setup, where the leading segment shows a human pointing at
    # the target block in the robot's own view: letting history reach into it would leak
    # the task and bypass the prompt video (and deployment has no pointing phase at all).
    # Labeled episodes whose first segment does NOT match are dropped with a warning (the
    # cut cannot be located); unlabeled episodes are untouched.
    skip_leading_subtask: str = ""
    # ---- human-video-prompt conditioning ----
    # "key=path" pairs, comma-separated: each LeRobot root holds human demonstration
    # videos for the tasks whose subtask label CONTAINS the key, e.g.
    #   "green=/data/green_human_prompt,yellow=/data/yellow_human_prompt".
    # When set, every sample prepends a randomly drawn same-task human demo clip as a
    # SECOND video ("Human demonstration:" ... "Robot view:" ...): fresh pairing on every
    # draw, so the model cannot memorize demo<->episode associations and must read the
    # demo's content. Combine with skip_leading_subtask when the robot episodes contain
    # an in-scene instruction phase (e.g. pointing) that would leak the task.
    human_prompt_dirs: str = ""
    # Prompt-clip sampling: every `human_prompt_stride`th frame -- keep it equal to
    # frame_stride so both videos share the M-RoPE seconds-per-frame truthfully -- with
    # the final frame (the grasp outcome) always included, uniformly re-spaced down to
    # `human_prompt_max_frames` when over. Input-shape contract: stamped into
    # visual_budget.json; serve-time prompt sampling must match.
    human_prompt_stride: int = 10
    human_prompt_max_frames: int = 12
    # Reserve the LAST N videos (sorted order) of each pool for evaluation: never drawn
    # during training, so the prompt-swap success test measures reading an UNSEEN demo,
    # not recognizing a memorized one.
    human_prompt_holdout: int = 4
    # FAST action tokens (arXiv:2501.09747) as an extra VLM supervision signal, per the
    # knowledge-insulation recipe (arXiv:2505.23705). Requires a tokenizer fit with
    # scripts/train_fast_tokenizer.py and train_vlm=True. The encoded action chunk is
    # appended to the sequence as new tokens and supervised with cross-entropy; the
    # continuous action expert is masked OUT of these positions (anti-shortcut).
    use_fast_tokens: bool = False
    fast_tokenizer_path: Optional[str] = None


def parse_active_dims(spec: Optional[str]) -> Optional[np.ndarray]:
    """"7:14" -> [7..13]; "0,2,7:10" -> [0,2,7,8,9]; None/"" -> None (all dims)."""
    if not spec:
        return None
    dims: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            start, end = part.split(":")
            dims.extend(range(int(start), int(end)))
        else:
            dims.append(int(part))
    return np.asarray(dims, dtype=np.int64)


def make_delta_mask(spec: str) -> np.ndarray:
    """"6,-1,6,-1" -> [T]*6 + [F] + [T]*6 + [F], as openpi's make_bool_mask."""
    mask: List[bool] = []
    for part in spec.split(","):
        n = int(part)
        mask.extend([n > 0] * abs(n))
    return np.asarray(mask, dtype=bool)


def quantile_normalize(x: np.ndarray, stats: Dict[str, list]) -> np.ndarray:
    """Exactly openpi's quantile norm (transforms.py). NOTE: dims that barely move in
    the dataset have q99 - q01 ~ 0, which amplifies sensor noise into huge normalized
    values that dominate the flow loss — exclude such dims with --active_dims instead
    (e.g. the stationary left arm in the block-mem data)."""
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


def quantile_unnormalize(x: np.ndarray, stats: Dict[str, list]) -> np.ndarray:
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def _discretize_state(normalized_state: np.ndarray, fixed_width: bool = False) -> str:
    """fixed_width: zero-pad every bin to 3 digits. Qwen tokenizes digit-by-digit, so
    plain str() makes the prompt LENGTH depend on the state VALUES -- harmless at 10
    numbers, but with state history (100 numbers) it swings ~90 tokens and breaks
    compiled-serving shape stability. State-history checkpoints therefore train and
    serve fixed-width; legacy checkpoints keep the exact format they trained with."""
    state = np.clip(normalized_state, -1.0, 1.0)
    discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    if fixed_width:
        return " ".join(f"{v:03d}" for v in discretized)
    return " ".join(map(str, discretized))


def format_robot_prompt(task: str, normalized_state: np.ndarray,
                        past_states: Optional[np.ndarray] = None) -> str:
    """pi0.5-style prompt with the discretized state. Must be identical at train/serve time.

    `past_states` (K, D), oldest first: the states at the SAME clamped timesteps as the
    image-history frames (all but the newest, whose state is the current one). Rendered
    before the current state so `State:` keeps its trained position next to the action
    region; each past state is discretized exactly like the current one, ';'-separated."""
    cleaned = task.strip().replace("_", " ").replace("\n", " ")
    if past_states is not None and len(past_states) > 0:
        # State-history format is fixed-width throughout (see _discretize_state): the
        # prompt token count is then CONSTANT, so compiled serving warms one shape.
        past_str = " ; ".join(_discretize_state(s, fixed_width=True) for s in past_states)
        return (f"Task: {cleaned}, Past states: {past_str}, "
                f"State: {_discretize_state(normalized_state, fixed_width=True)}")
    return f"Task: {cleaned}, State: {_discretize_state(normalized_state)}"


def make_action_chunk(
    actions: np.ndarray, state: np.ndarray, t: int, horizon: int, delta_mask: Optional[np.ndarray]
) -> np.ndarray:
    """Absolute action chunk actions[t:t+H] (edge-padded), optionally as deltas to state_t."""
    chunk = actions[t : t + horizon]
    if len(chunk) < horizon:
        chunk = np.concatenate([chunk, np.repeat(chunk[-1:], horizon - len(chunk), axis=0)], axis=0)
    chunk = chunk.astype(np.float32).copy()
    if delta_mask is not None:
        chunk[:, delta_mask] -= state[None, delta_mask]
    return chunk


def undo_delta_actions(chunk: np.ndarray, state: np.ndarray, delta_mask: np.ndarray) -> np.ndarray:
    """Inverse of the delta transform, for turning sampled actions back into joint targets."""
    out = chunk.copy()
    out[..., delta_mask] += state[None, delta_mask]
    return out


def apply_delta_actions(chunk: np.ndarray, state: np.ndarray, delta_mask: np.ndarray) -> np.ndarray:
    """Forward delta transform on an ABSOLUTE action chunk: express masked dims relative to
    `state` (the observation the model is conditioned on). Inverse of undo_delta_actions.
    Used by RTC to re-express a committed absolute action prefix in the model's action
    space relative to the CURRENT observation."""
    out = chunk.astype(np.float32).copy()
    out[..., delta_mask] -= state[None, delta_mask]
    return out


def compute_norm_stats(
    episodes: Sequence[Dict], horizon: int, delta_mask: Optional[np.ndarray],
    max_chunk_rows: int = 50_000_000,
) -> Dict[str, Dict[str, list]]:
    """q01/q99/mean/std for states and (delta-transformed) action chunks, openpi-style.

    Stats are computed over the transformed action values the model will actually see,
    i.e. after the delta conversion and over all chunk offsets. For large datasets the
    full chunk tensor (frames x horizon x dims) does not fit in RAM (e.g. the 3000-episode
    ABC conversion is ~9M frames -> 24+ GB), so timesteps are subsampled with a
    deterministic stride chosen to keep at most `max_chunk_rows` chunk rows; datasets
    below the cap (every existing trossen set) are computed exactly as before (stride 1).
    """
    total_frames = sum(len(ep["actions"]) for ep in episodes)
    stride = max(1, int(np.ceil(total_frames * horizon / max_chunk_rows)))
    if stride > 1:
        print(f"[compute_norm_stats] {total_frames} frames x horizon {horizon}: "
              f"subsampling every {stride}th timestep to bound memory")
    all_states = []
    all_action_values = []
    for ep_i, ep in enumerate(episodes):
        states, actions = ep["states"], ep["actions"]
        # DAgger episodes (min_start > 0): stats cover only the timesteps that can be
        # sampled for training -- the human-correction segment -- matching _build_item.
        start = int(ep.get("min_start", 0))
        all_states.append(states[start:])
        for t in range(start + (ep_i % stride), len(actions), stride):
            all_action_values.append(make_action_chunk(actions, states[t], t, horizon, delta_mask))
    states = np.concatenate(all_states, axis=0)
    action_values = np.concatenate(all_action_values, axis=0).reshape(-1, states.shape[-1])

    def _stats(x):
        return {
            "q01": np.quantile(x, 0.01, axis=0).tolist(),
            "q99": np.quantile(x, 0.99, axis=0).tolist(),
            "mean": x.mean(axis=0).tolist(),
            "std": x.std(axis=0).tolist(),
        }

    return {"state": _stats(states), "actions": _stats(action_values)}


class RobotFlowMatchingDataset(Dataset):
    def __init__(self, processor, data_args: RobotDataArguments):
        super().__init__()
        if getattr(data_args, "model_type", "qwen3vl") != "qwen3vl":
            raise ValueError("RobotFlowMatchingDataset only supports Qwen3-VL")
        self.get_rope_index = get_rope_index_3
        self.data_args = data_args
        self.horizon = data_args.action_horizon
        self.num_frames = data_args.num_frames
        self.stride = data_args.frame_stride
        self.active_dims = parse_active_dims(data_args.active_dims)
        self.delta_mask = make_delta_mask(data_args.delta_mask) if data_args.use_delta_actions else None
        self.wrist_cameras = [c.strip() for c in data_args.wrist_cameras.split(",") if c.strip()]

        if data_args.action_space not in ("joint", "ee6d"):
            raise ValueError(f"unknown action_space {data_args.action_space!r}")
        if data_args.state_history and not data_args.image_history:
            raise ValueError("--state_history is frame-aligned to the image history; "
                             "it requires --image_history True")
        num_dims = len(self.active_dims) if self.active_dims is not None else None
        if data_args.action_space == "ee6d":
            # The FK conversion consumes the SELECTED 7 joint dims and emits EE_DIM;
            # everything downstream (action_dim, delta_mask) lives in EE space.
            if num_dims != 7:
                raise ValueError(
                    f"action_space=ee6d requires active_dims selecting the 7-dim right arm "
                    f"(got {num_dims}); FK is defined for [q0..q5, jaw]."
                )
            num_dims = EE_DIM
        if num_dims is not None and num_dims != data_args.action_dim:
            raise ValueError(
                f"active_dims '{data_args.active_dims}' (action_space="
                f"{data_args.action_space}) implies {num_dims} dims but action_dim is "
                f"{data_args.action_dim}; they must match (and the model's expert "
                "action_dim comes from action_dim)."
            )
        if self.delta_mask is not None and num_dims is not None and len(self.delta_mask) != num_dims:
            raise ValueError(
                f"delta_mask '{data_args.delta_mask}' covers {len(self.delta_mask)} dims but "
                f"the {data_args.action_space} representation has {num_dims}; the mask "
                "applies to the REPRESENTATION dims (e.g. '6,-1' for joints, '9,-1' for ee6d)."
            )

        self.episodes = self._scan_episodes(data_args)
        if not self.episodes:
            raise ValueError(f"No usable episodes found under: {data_args.robot_data_dirs}")
        print(f"[RobotFlowMatchingDataset] {len(self.episodes)} training episodes")

        self.human_prompt_pools = self._scan_human_prompts(data_args)

        # fps for windowed single-frame wrist decode (avoids full-decoding the video).
        first_root = Path(data_args.robot_data_dirs.split(",")[0].strip())
        info_path = first_root / "meta" / "info.json"
        self.fps = float(json.load(open(info_path)).get("fps", 30)) if info_path.exists() else 30.0

        self.norm_stats = self._load_or_compute_norm_stats(data_args)

        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)

        self.use_fast = data_args.use_fast_tokens
        if self.use_fast:
            self._init_fast_tokenizer(data_args)

        # Input dump (sanity check): QWEN_DUMP_MODEL_INPUTS=<dir> makes rank 0 write the
        # first QWEN_DUMP_MODEL_INPUTS_N items per dataloader process as PNGs reconstructed
        # from pixel_values (exactly what the model sees) + a compression report, then the
        # hook goes dead. See qwenvl/data/input_inspect.py. Off (zero overhead) by default.
        self._dump_dir = os.environ.get("QWEN_DUMP_MODEL_INPUTS", "")
        self._dump_left = (
            int(os.environ.get("QWEN_DUMP_MODEL_INPUTS_N", "2"))
            if self._dump_dir and int(os.environ.get("RANK", "0")) == 0
            else 0
        )

    def _init_fast_tokenizer(self, data_args: RobotDataArguments):
        """Load the fitted FAST tokenizer and register its output tokens (plus two
        boundary markers) as NEW tokens on the Qwen tokenizer. Records the id offsets
        so _build_item can map FAST ids -> Qwen ids. The training script must resize the
        model embeddings to len(tokenizer) afterwards (see self.vlm_vocab_size)."""
        from transformers import AutoProcessor

        if not data_args.fast_tokenizer_path:
            raise ValueError(
                "use_fast_tokens=True requires --fast_tokenizer_path "
                "(fit one with scripts/train_fast_tokenizer.py)."
            )
        self.fast_tok = AutoProcessor.from_pretrained(
            data_args.fast_tokenizer_path, trust_remote_code=True
        )
        meta_path = Path(data_args.fast_tokenizer_path) / "fast_fit_meta.json"
        meta = json.load(open(meta_path)) if meta_path.exists() else {}
        self.fast_vocab_size = int(meta.get("vocab_size", getattr(self.fast_tok, "vocab_size", 1024)))
        # Guard against a tokenizer fit on a different action space than we train on.
        for key, ours in [("action_dim", data_args.action_dim), ("action_horizon", data_args.action_horizon),
                          ("active_dims", data_args.active_dims), ("delta_mask", data_args.delta_mask)]:
            if key in meta and str(meta[key]) != str(ours):
                raise ValueError(
                    f"FAST tokenizer was fit with {key}={meta[key]} but training uses {ours}; refit it."
                )

        markers = ["<|action_start|>", "<|action_end|>"]
        action_tokens = [f"<|action_{i}|>" for i in range(self.fast_vocab_size)]
        self.tokenizer.add_tokens(markers + action_tokens)
        self.fast_start_id = self.tokenizer.convert_tokens_to_ids("<|action_start|>")
        self.fast_end_id = self.tokenizer.convert_tokens_to_ids("<|action_end|>")
        self.fast_base_id = self.tokenizer.convert_tokens_to_ids("<|action_0|>")
        self.vlm_vocab_size = len(self.tokenizer)  # training script resizes embeddings to this
        print(
            f"[RobotFlowMatchingDataset] FAST enabled: vocab {self.fast_vocab_size}, "
            f"base id {self.fast_base_id}, new tokenizer size {self.vlm_vocab_size}"
        )

    # ------------------------------------------------------------------
    # Human-prompt video pools (see RobotDataArguments.human_prompt_dirs)
    # ------------------------------------------------------------------
    def _scan_human_prompts(self, data_args: RobotDataArguments):
        """Build {key: [video paths]} pools of human demo clips and verify every robot
        episode's subtask labels match exactly one pool key. Returns None when off."""
        if not data_args.human_prompt_dirs:
            return None
        pools = {}
        for spec in data_args.human_prompt_dirs.split(","):
            spec = spec.strip()
            if not spec:
                continue
            key, _, root_str = spec.partition("=")
            key, root = key.strip(), Path(root_str.strip())
            if not root_str or not key:
                raise ValueError(f"human_prompt_dirs entry {spec!r} is not 'key=path'")
            vids = []
            for chunk_dir in sorted((root / "videos").glob("chunk-*")):
                cam_dir = chunk_dir / f"observation.images.{data_args.camera}"
                vids.extend(sorted(cam_dir.glob("episode_*.mp4")))
            if not vids:
                raise ValueError(f"no {data_args.camera} videos under {root}")
            holdout = data_args.human_prompt_holdout
            train_vids = vids[:-holdout] if 0 < holdout < len(vids) else vids
            pools[key] = [str(v) for v in train_vids]
            print(f"[RobotFlowMatchingDataset] human-prompt pool '{key}': "
                  f"{len(train_vids)} train clips ({len(vids) - len(train_vids)} held out) "
                  f"from {root}")
        self.human_prompt_pools = pools  # _human_prompt_key reads it during validation
        unmatched = [Path(ep["video_path"]).name for ep in self.episodes
                     if self._human_prompt_key(ep) is None]
        if unmatched:
            raise ValueError(
                f"{len(unmatched)} episodes have no subtask label containing any human-"
                f"prompt pool key {sorted(pools)} (first: {unmatched[:3]}); cannot pair "
                "a demo video. Fix the keys or the labels.")
        return pools

    def _human_prompt_key(self, episode: Dict) -> Optional[str]:
        for segment in episode.get("subtasks", []):
            for key in self.human_prompt_pools:
                if key in segment["task"]:
                    return key
        return None

    def _extract_human_prompt(self, episode: Dict) -> List[Image.Image]:
        """A same-task human demo clip, freshly drawn per call. Full-clip decode is fine:
        prompt recordings are a few seconds. Stride-sampled with the final frame (the
        grasp outcome) forced in; uniformly re-spaced when over max_frames."""
        path = random.choice(self.human_prompt_pools[self._human_prompt_key(episode)])
        self._last_prompt_path = path  # provenance for the QWEN_NAN_DEBUG ring buffer
        video, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
        total = len(video)
        idxs = list(range(0, total, self.data_args.human_prompt_stride))
        if idxs[-1] != total - 1:
            idxs.append(total - 1)
        max_frames = self.data_args.human_prompt_max_frames
        if len(idxs) > max_frames:
            sel = np.linspace(0, len(idxs) - 1, max_frames)
            idxs = [idxs[int(round(s))] for s in sel]
        return [Image.fromarray(video[i].numpy()) for i in idxs]

    # ------------------------------------------------------------------
    # Episode discovery
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_video(root: Path, chunk_name: str, camera: str, video_name: str) -> Optional[Path]:
        """Raw AV1 (read_video decodes it) preferred; h264_videos/ fallback; else None."""
        cam_dir = root / "videos" / chunk_name / f"observation.images.{camera}"
        for candidate in (cam_dir / video_name, cam_dir / "h264_videos" / video_name):
            if candidate.exists():
                return candidate
        return None

    def _scan_episodes(self, data_args: RobotDataArguments) -> List[Dict]:
        episodes = []
        for root_str in data_args.robot_data_dirs.split(","):
            root = Path(root_str.strip())
            if not root.exists():
                print(f"[RobotFlowMatchingDataset] skipping missing root {root}")
                continue

            chunk_dirs = sorted((root / "data").glob("chunk-*"))
            subtask_labels = {}
            instructions = {}
            per_root = []
            for chunk_dir in chunk_dirs:
                chunk_name = chunk_dir.name
                labels_path = root / "videos" / chunk_name / "subtask_labels.json"
                if labels_path.exists():
                    with open(labels_path) as f:
                        subtask_labels.update(json.load(f))
                # Optional per-episode task instructions (multi-task datasets, e.g. the
                # ABC-130k conversion). Keyed by video basename like subtask_labels.json.
                # Absent -> episode["instruction"] is None -> prompt falls back to the
                # global --subtask_question, so single-task datasets are unaffected.
                instructions_path = root / "videos" / chunk_name / "instructions.json"
                if instructions_path.exists():
                    with open(instructions_path) as f:
                        instructions.update(json.load(f))

                for parquet_path in sorted(chunk_dir.glob("episode_*.parquet")):
                    video_name = parquet_path.stem + ".mp4"
                    # PyAV (libdav1d) decodes the raw AV1 LeRobot videos directly; h264
                    # fallback if present. No convert_av1.sh step required.
                    video_path = self._resolve_video(root, chunk_name, data_args.camera, video_name)
                    if video_path is None:
                        print(f"[RobotFlowMatchingDataset] no video for {parquet_path}, skipping")
                        continue

                    wrist_paths = []
                    missing_wrist = False
                    for cam in self.wrist_cameras:
                        wp = self._resolve_video(root, chunk_name, cam, video_name)
                        if wp is None:
                            print(f"[RobotFlowMatchingDataset] no {cam} video for {parquet_path}, skipping episode")
                            missing_wrist = True
                            break
                        wrist_paths.append(str(wp))
                    if missing_wrist:
                        continue

                    # DAgger recordings carry a per-frame human-intervention flag. Frames
                    # before the first intervention are the policy's own (erroneous)
                    # rollout: usable as history context, but never as a training
                    # timestep (see min_start in _build_item / compute_norm_stats).
                    has_iv = "is_intervention" in pq.read_schema(parquet_path).names
                    df = pd.read_parquet(
                        parquet_path,
                        columns=["observation.state", "action"] + (["is_intervention"] if has_iv else []),
                    )
                    states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
                    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
                    min_start = 0
                    if has_iv:
                        flags = np.stack(df["is_intervention"].to_numpy()).astype(np.float32).ravel() > 0.5
                        min_start = int(flags.argmax()) if flags.any() else 0
                    # Leading-segment gate (see RobotDataArguments.skip_leading_subtask):
                    # unlike the DAgger gate above, hist_min ALSO clamps history there.
                    hist_min = 0
                    segs = subtask_labels.get(video_name, [])
                    if data_args.skip_leading_subtask and segs:
                        if (segs[0]["task"] == data_args.skip_leading_subtask
                                and len(segs) > 1):
                            min_start = max(min_start, int(segs[0]["end"]) + 1)
                            hist_min = min_start
                        else:
                            print(f"[RobotFlowMatchingDataset] dropping {video_name}: "
                                  f"labels do not start with a "
                                  f"'{data_args.skip_leading_subtask}' segment, cannot "
                                  f"locate the cut")
                            continue
                    if self.active_dims is not None:
                        # Slice once at load time: everything downstream (delta transform,
                        # norm stats, prompt state, chunks) lives in the selected space.
                        states = states[:, self.active_dims]
                        actions = actions[:, self.active_dims]
                    if data_args.min_episode_len and len(states) < data_args.min_episode_len:
                        print(f"[RobotFlowMatchingDataset] skipping {parquet_path.stem}: "
                              f"{len(states)} < min_episode_len {data_args.min_episode_len}")
                        continue
                    if data_args.action_space == "ee6d":
                        # Joints -> [xyz, 6D rotation, jaw] once at load; every consumer
                        # (deltas, norm stats, prompt state, FAST) then lives in EE space.
                        states = joints_to_ee(states)
                        actions = joints_to_ee(actions)
                    per_root.append(
                        {
                            "states": states,
                            "actions": actions,
                            "video_path": str(video_path),
                            "wrist_paths": wrist_paths,
                            "subtasks": segs,
                            "instruction": instructions.get(video_name),
                            "min_start": min_start,
                            "hist_min": hist_min,
                        }
                    )

            # Same 75% split convention as PrototypeRobotDataset (sorted, first fraction).
            split_idx = int(len(per_root) * data_args.train_split)
            episodes.extend(per_root[:split_idx])
        return episodes

    def _check_norm_stats_meta(self, meta: Dict, data_args: RobotDataArguments, path: str):
        """A frozen stats artifact must describe the SAME action space this run uses;
        a different source dataset / split only warns (expected when serving)."""
        structural = {
            "horizon": self.horizon,
            "use_delta_actions": data_args.use_delta_actions,
            "delta_mask": data_args.delta_mask,
            "active_dims": data_args.active_dims,
        }
        # Action space: legacy artifacts predate the field and are all joint-space.
        theirs_space = meta.get("action_space", "joint")
        if theirs_space != data_args.action_space:
            raise ValueError(
                f"norm stats at {path} are for action_space={theirs_space!r} but this run "
                f"uses {data_args.action_space!r} -- wrong representation, refusing."
            )
        missing = [k for k in structural if k not in meta]
        if missing:
            # Trusting a file whose action space can't be verified would defeat the point
            # of the check -- normalization mismatches degrade the policy silently.
            raise ValueError(
                f"norm stats at {path} lack meta key(s) {missing} so their action space "
                "cannot be verified -- refusing to trust them. Refit with "
                "scripts/train_fast_tokenizer.py (which stamps a full meta block)."
            )
        for key, ours in structural.items():
            theirs = meta[key]
            same = theirs == ours
            if not same:
                # Spec strings may differ while meaning the same thing ('' vs None
                # active_dims, '6,-1' with stray spaces) -- compare the parsed forms.
                try:
                    if key == "active_dims":
                        a, b = parse_active_dims(theirs), parse_active_dims(ours)
                        same = (a is None) == (b is None) and (a is None or np.array_equal(a, b))
                    elif key == "delta_mask":
                        same = np.array_equal(make_delta_mask(theirs), make_delta_mask(ours))
                except (ValueError, TypeError, AttributeError):
                    same = False
            if not same:
                raise ValueError(
                    f"norm stats at {path} were computed with {key}={theirs!r} but this "
                    f"run uses {ours!r} -- wrong action space, refusing to continue."
                )
        for key in ("robot_data_dirs", "train_split"):
            if key in meta and meta[key] != getattr(data_args, key):
                print(
                    f"[RobotFlowMatchingDataset] note: norm stats {path} were computed over "
                    f"{key}={meta[key]!r} (this run: {getattr(data_args, key)!r}) -- fine when "
                    "serving a frozen artifact; refit the FAST tokenizer + stats if this is a "
                    "NEW training dataset."
                )

    def _load_or_compute_norm_stats(self, data_args: RobotDataArguments) -> Dict:
        # Frozen-artifact stats, trusted as-is (never recomputed, never written back):
        #   1. an existing file at --norm_stats_path (serving passes the checkpoint's copy);
        #   2. else norm_stats.json saved next to the FAST tokenizer by
        #      scripts/train_fast_tokenizer.py -- stats and tokenizer are one artifact
        #      (FAST is fit inside the normalized space the stats define).
        # This keeps stats attached to model artifacts, not dataset dirs, so a new
        # training mix can never silently overwrite the file an older served model reads.
        explicit = Path(data_args.norm_stats_path) if data_args.norm_stats_path else None
        candidates = [(explicit, "--norm_stats_path")]
        if explicit is None and data_args.fast_tokenizer_path:
            candidates.append((Path(data_args.fast_tokenizer_path) / "norm_stats.json", "FAST tokenizer dir"))
        for candidate, origin in candidates:
            if candidate is not None and candidate.exists():
                with open(candidate) as f:
                    stats = json.load(f)
                self._check_norm_stats_meta(stats.get("meta", {}), data_args, str(candidate))
                print(f"[RobotFlowMatchingDataset] norm stats ({origin}): {candidate}")
                return stats

        # Legacy cache in the first dataset dir (or an explicit path that doesn't exist
        # yet): reuse when the meta matches exactly, else compute and (over)write.
        if explicit is not None:
            stats_path = explicit
        else:
            first_root = Path(data_args.robot_data_dirs.split(",")[0].strip())
            stats_path = first_root / "qwen_action_expert_norm_stats.json"

        meta = {
            "horizon": self.horizon,
            "use_delta_actions": data_args.use_delta_actions,
            "delta_mask": data_args.delta_mask,
            "active_dims": data_args.active_dims,
            "robot_data_dirs": data_args.robot_data_dirs,
            "train_split": data_args.train_split,
        }
        # Fingerprint DAgger min_start gating so pre-gating caches are not reused for a
        # gated mix; only added when nonzero so existing plain-dataset caches stay valid.
        total_min_start = sum(int(ep.get("min_start", 0)) for ep in self.episodes)
        if total_min_start:
            meta["min_start_frames"] = total_min_start
        # Same back-compat convention for the newer fields: only stamped when non-default.
        if data_args.action_space != "joint":
            meta["action_space"] = data_args.action_space
        if data_args.min_episode_len:
            meta["min_episode_len"] = data_args.min_episode_len
        if stats_path.exists():
            with open(stats_path) as f:
                cached = json.load(f)
            if cached.get("meta") == meta:
                print(f"[RobotFlowMatchingDataset] loaded norm stats from {stats_path}")
                return cached
            # NEVER overwrite an existing stats file on a meta mismatch: a deployed model
            # may resolve its stats to this exact file (the pre-relocation convention), and
            # os.replace-ing it with a new mix's stats would silently change what that
            # model serves. Loud error > silent clobber.
            raise ValueError(
                f"norm stats at {stats_path} were computed for a different config\n"
                f"  cached meta: {cached.get('meta')}\n"
                f"  this run:    {meta}\n"
                "Refusing to overwrite a file an already-deployed model may read. For a new "
                "data mix, refit the FAST tokenizer (scripts/train_fast_tokenizer.py now "
                "saves norm_stats.json next to it, which this loader prefers) or pass "
                "--norm_stats_path pointing at a NEW file. Delete the cached file only if "
                "you are certain nothing serves from it."
            )

        # Multi-rank runs: only global rank 0 computes (the compute walks every episode
        # and is memory-heavy at large scale); other ranks poll for rank 0's atomic write.
        rank = int(os.environ.get("RANK", "0"))
        if rank != 0:
            print(f"[RobotFlowMatchingDataset] rank {rank} waiting for norm stats {stats_path}")
            deadline = time.time() + 3600
            while time.time() < deadline:
                time.sleep(5)
                if stats_path.exists():
                    with open(stats_path) as f:
                        cached = json.load(f)
                    if cached.get("meta") == meta:
                        return cached
            raise RuntimeError(f"rank {rank}: timed out waiting for norm stats {stats_path}")

        print(f"[RobotFlowMatchingDataset] computing norm stats -> {stats_path}")
        # Fail on an unwritable target BEFORE the minutes-long compute (a crash at the
        # write would also leave non-zero ranks polling until their 1h timeout).
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats = compute_norm_stats(self.episodes, self.horizon, self.delta_mask)
        stats["meta"] = meta
        # Atomic write so concurrent ranks don't read a partial file.
        fd, tmp = tempfile.mkstemp(dir=str(stats_path.parent), suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(stats, f, indent=2)
        os.replace(tmp, stats_path)
        return stats

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.episodes)

    @property
    def lengths(self):
        return [1] * len(self.episodes)

    @property
    def modality_lengths(self):
        return [1] * len(self.episodes)

    @property
    def pre_calculated_length(self):
        return np.array([1] * len(self.episodes))

    def _extract_frames(self, episode: Dict, t: int) -> List[Image.Image]:
        """A `num_frames`-frame history ending at t, spaced by `stride` (oldest->newest),
        clamped at the episode boundaries.

        Decodes only the [t-(N-1)*stride, t] window (~N*stride frames) instead of the whole
        video: read_video with start_pts/end_pts returns exactly the contiguous frames
        lo..t, so global frame g maps to window-local index g-lo. Clamped duplicates near
        the episode start reuse the same frame.

        Episodes gated by skip_leading_subtask carry hist_min > 0: the low clamp moves
        from frame 0 to hist_min so history can never reach into the gated-out segment
        (near the cut the first frame repeats, same as the episode-start behavior)."""
        hist_min = int(episode.get("hist_min", 0))
        lo = max(hist_min, t - (self.num_frames - 1) * self.stride)
        video, _, _ = read_video(
            episode["video_path"], start_pts=lo / self.fps, end_pts=(t + 0.9) / self.fps,
            pts_unit="sec", output_format="THWC",
        )
        n = len(video)
        if n == 0:  # empty window -> full-decode fallback
            video, _, _ = read_video(episode["video_path"], pts_unit="sec", output_format="THWC")
            total = len(video)
            indices = [min(max(t - i * self.stride, hist_min), total - 1) for i in reversed(range(self.num_frames))]
            return [Image.fromarray(video[i].numpy()) for i in indices]
        frames = []
        for i in reversed(range(self.num_frames)):
            g = max(t - i * self.stride, hist_min)  # desired global frame (low-clamped)
            local = min(max(g - lo, 0), n - 1)  # map into window (high-clamp to its end)
            frames.append(Image.fromarray(video[local].numpy()))
        return frames

    def _extract_single_frame(self, path: str, t: int, budget: int) -> Image.Image:
        """Single frame at the current timestep t (no history), pre-resized to <= budget
        pixels (aspect preserved, downscale only). Windowed decode (~30ms) instead of the
        whole video. Used for wrist stills AND, in no-history mode, the current cam_high still
        (apply_chat_template does NOT apply max_pixels to pre-loaded PIL images, so we resize)."""
        video, _, _ = read_video(
            path, start_pts=t / self.fps, end_pts=(t + 0.9) / self.fps,
            pts_unit="sec", output_format="THWC",
        )
        if len(video) == 0:  # t past the end / empty window -> full-decode fallback
            video, _, _ = read_video(path, pts_unit="sec", output_format="THWC")
            frame = video[min(max(t, 0), len(video) - 1)]
        else:
            frame = video[0]
        img = Image.fromarray(frame.numpy())
        w, h = img.size
        # GATE 1 of 2 for stills (the wrist, and the top frame in no-history mode). This is
        # OUR resize, before the processor ever sees the image; GATE 2 is the processor's own
        # --max_pixels ceiling. Because this one runs first and shrinks the image, raising
        # only --max_pixels changes nothing -- the image is already under the ceiling. To give
        # the wrist more resolution, raise --wrist_max_pixels AND --max_pixels together.
        # (History frames never pass through here: they go to the video processor raw.)
        if w * h > budget:
            scale = (budget / (w * h)) ** 0.5
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
        return img

    def _extract_wrist_images(self, episode: Dict, t: int) -> List[Image.Image]:
        """Single pre-resized frame at t from each wrist video (no history)."""
        budget = self.data_args.wrist_max_pixels
        return [self._extract_single_frame(p, t, budget) for p in episode.get("wrist_paths", [])]

    def _augment(self, frames: List[Image.Image], geometric: bool) -> List[Image.Image]:
        """pi05-style augmentation with ONE parameter draw applied to every frame in
        `frames` (see RobotDataArguments.image_aug for the per-clip rationale).
        geometric=True (top-down camera): RandomCrop 95% + resize back + rotate U(-5,5)
        + jitter. geometric=False (wrists, like openpi): jitter only."""
        import torchvision.transforms.functional as TF
        from torchvision.transforms import InterpolationMode

        out = frames
        if geometric:
            w, h = out[0].size
            cw, ch = int(round(w * 0.95)), int(round(h * 0.95))
            x0 = random.randint(0, w - cw)
            y0 = random.randint(0, h - ch)
            angle = random.uniform(-5.0, 5.0)
            out = [TF.resize(TF.crop(f, y0, x0, ch, cw), [h, w],
                             interpolation=InterpolationMode.BILINEAR) for f in out]
            out = [TF.rotate(f, angle, interpolation=InterpolationMode.BILINEAR) for f in out]
        # openpi/augmax ColorJitter, matched to pi05's EFFECTIVE behavior: fires with
        # p=0.5 per camera draw (augmax default -- pi05 does not jitter every image);
        # brightness/contrast factor ranges [1-x, 1+x] plus the augmax default hue shift
        # U(-0.1, 0.1); saturation omitted (augmax's saturation jitter is a no-op, the
        # result is discarded, so pi05 never trained with it). One draw shared by all
        # frames of this camera.
        if random.random() < 0.5:
            b = random.uniform(0.7, 1.3)
            c = random.uniform(0.6, 1.4)
            h = random.uniform(-0.1, 0.1)
            out = [TF.adjust_hue(TF.adjust_contrast(TF.adjust_brightness(f, b), c), h)
                   for f in out]
        return out

    def _subtask_at(self, episode: Dict, t: int) -> str:
        for segment in episode["subtasks"]:
            if segment["start"] <= t <= segment["end"]:
                return segment["task"]
        return self.data_args.default_prompt

    def _build_item(self, idx: int) -> Dict[str, torch.Tensor]:
        episode = self.episodes[idx]
        num_steps = len(episode["states"])
        # min_start > 0 on DAgger episodes: only the human-correction segment may be a
        # training timestep. _extract_frames still gets the raw t, so the image history
        # reaches back past min_start into the policy's own mistake -- by design.
        t = random.randint(int(episode.get("min_start", 0)), num_steps - 1)

        state = episode["states"][t]
        actions = make_action_chunk(episode["actions"], state, t, self.horizon, self.delta_mask)
        norm_state = quantile_normalize(state, self.norm_stats["state"])
        norm_actions = quantile_normalize(actions, self.norm_stats["actions"])
        # States at the image-history timesteps (oldest first, newest frame's state == the
        # current state above): SAME indices and edge-clamping as _extract_frames.
        norm_past = None
        if self.data_args.state_history:
            hist_min = int(episode.get("hist_min", 0))
            idxs = [max(hist_min, t - k * self.stride) for k in range(self.num_frames - 1, 0, -1)]
            norm_past = quantile_normalize(episode["states"][idxs], self.norm_stats["state"])

        # One decision per SAMPLE; each camera then gets its own parameter draw.
        aug_on = (self.data_args.image_aug and not getattr(self, "_aug_disabled", False)
                  and random.random() < self.data_args.image_aug_prob)

        wrist_images = self._extract_wrist_images(episode, t)
        if aug_on:
            wrist_images = [self._augment([im], geometric=False)[0] for im in wrist_images]
        subtask = self._subtask_at(episode, t)

        if self.data_args.predict_subtask:
            # The prompt is the episode's task instruction when the dataset carries one
            # (videos/chunk-*/instructions.json, e.g. the ABC-130k conversion), else the
            # fixed --subtask_question; the subtask is the assistant answer to predict.
            task_text = episode.get("instruction") or self.data_args.subtask_question
            # Mixed supervision: an episode with NO subtask labels contributes no language
            # signal (no assistant turn; the VLM is supervised through FAST alone).
            # Supervising the "waiting" fallback instead would teach the VLM a false
            # description of the scene on every unlabeled sample. Labeled episodes are
            # untouched, so datasets with and without labels mix freely in one run.
            assistant_text = subtask if episode["subtasks"] else None
        else:
            # Subtask-input mode: the subtask IS the prompt; no subtask prediction (the VLM is
            # supervised only through FAST). At inference the subtask is provided by the caller.
            task_text = subtask
            assistant_text = None
        prompt = format_robot_prompt(task_text, norm_state, past_states=norm_past)

        # Top-down cam_high (history video, or a single current still if image_history=False),
        # then the current-timestep wrist still(s), then the text.
        if self.data_args.image_history:
            top_frames = self._extract_frames(episode, t)
            if aug_on:
                top_frames = self._augment(top_frames, geometric=True)
            content = [{"type": "video", "video": top_frames}]
        else:
            top_img = self._extract_single_frame(episode["video_path"], t, self.data_args.max_pixels)
            if aug_on:
                top_img = self._augment([top_img], geometric=True)[0]
            top_frames = [top_img]
            content = [{"type": "image", "image": top_img}]
        content += [{"type": "image", "image": img} for img in wrist_images]
        content.append({"type": "text", "text": prompt})
        if self.human_prompt_pools is not None:
            # Freshly drawn same-task human demo as a FIRST video, with structural text
            # markers so the two videos are unambiguous. Independent augmentation draw
            # (same per-clip convention as the other cameras). Serve must build the
            # identical structure (inference.templatize).
            prompt_frames = self._extract_human_prompt(episode)
            if aug_on:
                prompt_frames = self._augment(prompt_frames, geometric=True)
            content = [{"type": "text", "text": "Human demonstration:"},
                       {"type": "video", "video": prompt_frames},
                       {"type": "text", "text": "Robot view:"}] + content
        messages = [{"role": "user", "content": content}]
        if assistant_text is not None:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})

        data_dict = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=assistant_text is None,
            do_sample_frames=False, # NOTE: this is important! If not set, it will default to True and subsample the images
        )

        input_ids = data_dict["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.tensor(input_ids).unsqueeze(0)
            data_dict["input_ids"] = input_ids

        labels = torch.full_like(input_ids, IGNORE_INDEX)
        # subtask_token_mask marks the ENTIRE assistant turn (<|im_start|>assistant\n ...
        # <|im_end|>\n). The model uses it only when trained with expert_attends_subtask=
        # False ("subtask insulation"): the expert then conditions on images+state alone and
        # the subtask stays a pure VLM co-training signal -- which lets inference run the
        # expert without waiting for subtask generation.
        subtask_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if assistant_text is not None:
            # Unmask assistant answers (same convention as preprocess_qwen_visual):
            # token 77091 is "assistant" in the <|im_start|>assistant\n header, and the
            # answer runs up to and including <|im_end|>\n (151645 + newline).
            ids = input_ids[0].tolist()
            pos = 0
            while pos < len(ids):
                # Require the full <|im_start|>assistant bigram (151644, 77091): a bare
                # 77091 also appears when user text contains punctuation-adjacent
                # "assistant" (e.g. '(assistant)'), and matching it alone would unmask
                # user-prompt tokens as labels AND insulate the expert from part of the
                # user turn (incl. the state string) -- silently, per episode.
                if ids[pos] == 77091 and pos > 0 and ids[pos - 1] == 151644:
                    ans_start = pos + 2
                    ans_end = ans_start
                    while ans_end < len(ids) and ids[ans_end] != 151645:
                        ans_end += 1
                    if ans_end < len(ids):
                        labels[0, ans_start : ans_end + 2] = input_ids[0, ans_start : ans_end + 2]
                        # pos-1 is the assistant turn's <|im_start|>; span ends after the
                        # trailing newline (ans_end+1), matching the label span above.
                        subtask_token_mask[0, pos - 1 : ans_end + 2] = True
                        pos = ans_end
                pos += 1
        else:
            # No assistant turn (subtask-input mode, or an unlabeled episode under mixed
            # supervision): the sequence still ends with the 3-token generation header
            # <|im_start|>assistant\n. Insulated serving masks that header from the expert
            # (inference.subtask_token_mask_for), so mark it here too -- otherwise these
            # training samples would let the expert attend tokens serving hides from it.
            # The FAST tokens appended below stay unmasked via the padding of this mask.
            ids = input_ids[0].tolist()
            for i in range(len(ids) - 1):
                if ids[i] == 151644 and ids[i + 1] == 77091:
                    subtask_token_mask[0, i:] = True
                    break
        # Append FAST action tokens (supervised with cross-entropy) BEFORE computing
        # position_ids so positions cover the whole sequence. fast_token_mask marks the
        # appended region, which the model excludes from the continuous expert's attention.
        fast_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self.use_fast:
            input_ids, labels, fast_token_mask = self._append_fast_tokens(input_ids, labels, norm_actions)
            data_dict["input_ids"] = input_ids
            # FAST tokens are not part of the assistant subtask turn
            pad = torch.zeros(1, input_ids.shape[1] - subtask_token_mask.shape[1], dtype=torch.bool)
            subtask_token_mask = torch.cat([subtask_token_mask, pad], dim=1)
        data_dict["labels"] = labels
        data_dict["fast_token_mask"] = fast_token_mask
        data_dict["subtask_token_mask"] = subtask_token_mask

        video_grid_thw = data_dict.get("video_grid_thw")
        second_per_grid_ts = None
        if video_grid_thw is not None:
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size / self.processor.video_processor.fps
            ] * len(video_grid_thw)

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            input_ids,
            image_grid_thw=data_dict.get("image_grid_thw"),
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
        )
        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [input_ids.shape[1]]

        data_dict["actions"] = torch.from_numpy(np.ascontiguousarray(norm_actions)).float()

        if os.environ.get("QWEN_NAN_DEBUG"):
            # Provenance ring buffer for the nan trap in train_action_expert.compute_loss.
            # Only meaningful with dataloader_num_workers=0 (workers are separate
            # processes; their buffers are invisible to the trainer).
            from collections import deque
            if not hasattr(self, "_recent_items"):
                self._recent_items = deque(maxlen=64)
            self._recent_items.append({
                "video": episode["video_path"], "t": int(t),
                "prompt_clip": getattr(self, "_last_prompt_path", None),
                "seq_len": int(input_ids.shape[1]),
                "fast_len": int(fast_token_mask.sum()),
            })

        if self._dump_left > 0:
            self._dump_left -= 1
            try:  # a debug dump must never break training
                from qwenvl.data.input_inspect import dump_item
                dump_item(self, data_dict, top_frames, wrist_images, subtask, t, idx, self._dump_dir)
            except Exception as e:  # noqa: BLE001
                print(f"[input-dump] non-fatal failure: {e}")

        return data_dict

    def _append_fast_tokens(self, input_ids, labels, norm_actions):
        """Encode the normalized action chunk to FAST ids, map them to the new Qwen token
        ids, and append `<|action_start|> <fast...> <|action_end|>` to the sequence.

        Labels supervise the FAST tokens and the end marker (so the model learns to
        terminate); the start marker is the given "prompt" for action decoding (-100).
        Returns (input_ids, labels, fast_token_mask), each (1, L + postfix_len)."""
        fast_ids = self.fast_tok(norm_actions[None].astype(np.float32))[0]  # list[int] in [0, vocab)
        postfix = [self.fast_start_id] + [self.fast_base_id + int(i) for i in fast_ids] + [self.fast_end_id]
        postfix = torch.tensor(postfix, dtype=input_ids.dtype).unsqueeze(0)

        # start marker not supervised; FAST tokens + end marker supervised.
        postfix_labels = postfix.clone()
        postfix_labels[0, 0] = IGNORE_INDEX

        input_ids = torch.cat([input_ids, postfix], dim=1)
        labels = torch.cat([labels, postfix_labels], dim=1)
        fast_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        fast_token_mask[0, -postfix.shape[1] :] = True
        return input_ids, labels, fast_token_mask

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_retries = 3
        for attempt in range(num_retries):
            try:
                return self._build_item(i)
            except Exception as e:  # noqa: BLE001
                print(f"[Try #{attempt}] Failed to fetch robot sample {i}: {e}")
                i = random.randint(0, len(self.episodes) - 1)
        return self._build_item(i)


@dataclass
class RobotActionDataCollator(DataCollatorForSupervisedDataset):
    tokenizer: object = None

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(instances)
        batch["actions"] = torch.stack([inst["actions"] for inst in instances])
        # Pad the per-token bool masks to the batched input_ids length (right pad = False:
        # padded positions are neither FAST nor subtask tokens). The base collator pads
        # input_ids to the max length in the batch, so match that.
        seq_len = batch["input_ids"].shape[1]
        for key in ("fast_token_mask", "subtask_token_mask"):
            if key in instances[0]:
                masks = []
                for inst in instances:
                    m = inst[key].squeeze(0)
                    if m.shape[0] < seq_len:
                        m = torch.cat([m, torch.zeros(seq_len - m.shape[0], dtype=torch.bool)])
                    masks.append(m[:seq_len])
                batch[key] = torch.stack(masks)
        return batch


def make_robot_data_module(processor, data_args: RobotDataArguments) -> Dict:
    train_dataset = RobotFlowMatchingDataset(processor, data_args)
    data_collator = RobotActionDataCollator(processor.tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)


# ============================================================================
# Manual inspection / verification. Run as a MODULE (-m) so the qwenvl package
# imports resolve (running the file path directly fails on `import qwenvl...`):
#
#   cd qwen-vl-finetune && python -m qwenvl.data.robot_data
#
# This block ONLY runs when the file is executed directly. Training imports this
# module (it does not run it as __main__), so this does NOT interfere with a
# training run -- there is no need to comment it out. It reproduces
# _build_item() step by step at a fixed timestep so every intermediate the
# dataset computes but does not return is visible, then calls the real ds[i] to
# show the final returned dict. Edit the hard-coded config to match your run.
# ============================================================================
if __name__ == "__main__":
    import os

    os.environ.setdefault(
        "HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface"
    )
    from transformers import AutoProcessor

    # ---- hard-coded config: matches scripts/train_action_expert_4b.sh ----
    MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
    ROBOT_DATA_DIRS = "/iris/projects/humanoid/trossen_data/0528_green_yellow_block_mem_merged_bk"
    SAMPLE_INDEX = 0          # which episode
    T = 100                   # which timestep within the episode (fixed for reproducibility)

    data_args = RobotDataArguments()
    data_args.model_type = "qwen3vl"
    data_args.robot_data_dirs = ROBOT_DATA_DIRS  # <-- the dataset path lives here
    data_args.active_dims = "7:14"
    data_args.action_dim = 7
    data_args.delta_mask = "6,-1"
    data_args.action_horizon = 50
    data_args.train_split = 1.0
    data_args.predict_subtask = True

    print("Loading processor + dataset...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, cache_dir=os.environ["HF_HOME"])
    ds = RobotFlowMatchingDataset(processor, data_args)

    print("\n===== DATASET-LEVEL =====")
    print(f"data dir        : {ROBOT_DATA_DIRS}")
    print(f"num episodes    : {len(ds)}")
    print(f"active_dims     : {ds.active_dims}")
    print(f"delta_mask      : {ds.delta_mask}")
    print(f"horizon         : {ds.horizon}")
    print(f"frames / stride : {ds.num_frames} / {ds.stride}")
    print(f"actions q01     : {ds.norm_stats['actions']['q01']}")
    print(f"actions q99     : {ds.norm_stats['actions']['q99']}")

    # ---- reproduce _build_item at a fixed t so intermediates are visible ----
    episode = ds.episodes[SAMPLE_INDEX]
    num_steps = len(episode["states"])
    t = min(T, num_steps - 1)

    state = episode["states"][t]                                                   # (7,) raw right-arm state
    actions = make_action_chunk(episode["actions"], state, t, ds.horizon, ds.delta_mask)  # (50, 7) deltas
    norm_state = quantile_normalize(state, ds.norm_stats["state"])                 # (7,) ~[-1, 1]
    norm_actions = quantile_normalize(actions, ds.norm_stats["actions"])           # (50, 7)
    subtask = ds._subtask_at(episode, t)                                           # time-aligned label
    prompt = format_robot_prompt(data_args.subtask_question, norm_state)           # text the VLM sees

    # round-trip: normalize -> unnormalize -> undo delta should recover raw joints
    recovered = undo_delta_actions(
        quantile_unnormalize(norm_actions, ds.norm_stats["actions"]), state, ds.delta_mask
    )
    raw_target = episode["actions"][t : t + ds.horizon]
    rt_ok = np.allclose(recovered[: len(raw_target)], raw_target, atol=1e-4)

    print(f"\n===== SAMPLE {SAMPLE_INDEX} @ t={t}/{num_steps}  ({os.path.basename(episode['video_path'])}) =====")
    print(f"raw state       : {np.array2string(state, precision=3)}")
    print(f"norm_state      : {np.array2string(norm_state, precision=3)}")
    print(f"actions shape   : {actions.shape}")
    print(f"norm_actions rng: [{norm_actions.min():.3f}, {norm_actions.max():.3f}]")
    print(f"subtask label   : {subtask!r}")
    print(f"prompt          : {prompt!r}")
    print(f"round-trip recovers raw joints: {rt_ok}")

    # ---- the real returned dict (note: ds[i] samples its OWN random t) ----
    item = ds[SAMPLE_INDEX]
    supervised = item["input_ids"][0][item["labels"][0] != -100]
    print("\n===== RETURNED ITEM (ds[i], random t) =====")
    for k, v in item.items():
        shape = tuple(v.shape) if isinstance(v, torch.Tensor) else v
        print(f"  {k:20s}: {type(v).__name__} {shape}")
    print(f"supervised tokens : {processor.tokenizer.decode(supervised)!r}")
    print(f"full text (300ch) : {processor.tokenizer.decode(item['input_ids'][0])[:300]!r}")

    # Uncomment to poke around interactively (any variable above is in scope):
    # breakpoint()
