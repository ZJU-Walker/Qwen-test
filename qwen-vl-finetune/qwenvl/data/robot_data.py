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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_video

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
    # If True (knowledge-insulation co-training), the user turn asks a fixed question and
    # the assistant turn contains the time-aligned subtask label with LM supervision.
    # If False (default, pi0.5-style), the subtask label IS the prompt and there is no
    # language supervision (labels are fully masked).
    predict_subtask: bool = False
    subtask_question: str = "which colored block did the human hand point to?"
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


def format_robot_prompt(task: str, normalized_state: np.ndarray) -> str:
    """pi0.5-style prompt with the discretized state. Must be identical at train/serve time."""
    state = np.clip(normalized_state, -1.0, 1.0)
    discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized))
    cleaned = task.strip().replace("_", " ").replace("\n", " ")
    return f"Task: {cleaned}, State: {state_str}"


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
    episodes: Sequence[Dict], horizon: int, delta_mask: Optional[np.ndarray]
) -> Dict[str, Dict[str, list]]:
    """q01/q99/mean/std for states and (delta-transformed) action chunks, openpi-style.

    Stats are computed over the transformed action values the model will actually see,
    i.e. after the delta conversion and over all chunk offsets.
    """
    all_states = []
    all_action_values = []
    for ep in episodes:
        states, actions = ep["states"], ep["actions"]
        all_states.append(states)
        for t in range(len(actions)):
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

        num_dims = len(self.active_dims) if self.active_dims is not None else None
        if num_dims is not None and num_dims != data_args.action_dim:
            raise ValueError(
                f"active_dims '{data_args.active_dims}' selects {num_dims} dims but "
                f"action_dim is {data_args.action_dim}; they must match (and the model's "
                "expert action_dim comes from action_dim)."
            )
        if self.delta_mask is not None and num_dims is not None and len(self.delta_mask) != num_dims:
            raise ValueError(
                f"delta_mask '{data_args.delta_mask}' covers {len(self.delta_mask)} dims but "
                f"active_dims selects {num_dims}; the mask applies to the SELECTED dims "
                "(e.g. '6,-1' for one 7-dof arm)."
            )

        self.episodes = self._scan_episodes(data_args)
        if not self.episodes:
            raise ValueError(f"No usable episodes found under: {data_args.robot_data_dirs}")
        print(f"[RobotFlowMatchingDataset] {len(self.episodes)} training episodes")

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
            per_root = []
            for chunk_dir in chunk_dirs:
                chunk_name = chunk_dir.name
                labels_path = root / "videos" / chunk_name / "subtask_labels.json"
                if labels_path.exists():
                    with open(labels_path) as f:
                        subtask_labels.update(json.load(f))

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

                    df = pd.read_parquet(parquet_path, columns=["observation.state", "action"])
                    states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
                    actions = np.stack(df["action"].to_numpy()).astype(np.float32)
                    if self.active_dims is not None:
                        # Slice once at load time: everything downstream (delta transform,
                        # norm stats, prompt state, chunks) lives in the selected space.
                        states = states[:, self.active_dims]
                        actions = actions[:, self.active_dims]
                    per_root.append(
                        {
                            "states": states,
                            "actions": actions,
                            "video_path": str(video_path),
                            "wrist_paths": wrist_paths,
                            "subtasks": subtask_labels.get(video_name, []),
                        }
                    )

            # Same 75% split convention as PrototypeRobotDataset (sorted, first fraction).
            split_idx = int(len(per_root) * data_args.train_split)
            episodes.extend(per_root[:split_idx])
        return episodes

    def _load_or_compute_norm_stats(self, data_args: RobotDataArguments) -> Dict:
        if data_args.norm_stats_path:
            stats_path = Path(data_args.norm_stats_path)
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
        if stats_path.exists():
            with open(stats_path) as f:
                cached = json.load(f)
            if cached.get("meta") == meta:
                print(f"[RobotFlowMatchingDataset] loaded norm stats from {stats_path}")
                return cached

        print(f"[RobotFlowMatchingDataset] computing norm stats -> {stats_path}")
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
        the episode start reuse the same frame."""
        lo = max(0, t - (self.num_frames - 1) * self.stride)
        video, _, _ = read_video(
            episode["video_path"], start_pts=lo / self.fps, end_pts=(t + 0.9) / self.fps,
            pts_unit="sec", output_format="THWC",
        )
        n = len(video)
        if n == 0:  # empty window -> full-decode fallback
            video, _, _ = read_video(episode["video_path"], pts_unit="sec", output_format="THWC")
            total = len(video)
            indices = [min(max(t - i * self.stride, 0), total - 1) for i in reversed(range(self.num_frames))]
            return [Image.fromarray(video[i].numpy()) for i in indices]
        frames = []
        for i in reversed(range(self.num_frames)):
            g = max(t - i * self.stride, 0)  # desired global frame (low-clamped)
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
        if w * h > budget:
            scale = (budget / (w * h)) ** 0.5
            img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.BILINEAR)
        return img

    def _extract_wrist_images(self, episode: Dict, t: int) -> List[Image.Image]:
        """Single pre-resized frame at t from each wrist video (no history)."""
        budget = self.data_args.wrist_max_pixels
        return [self._extract_single_frame(p, t, budget) for p in episode.get("wrist_paths", [])]

    def _subtask_at(self, episode: Dict, t: int) -> str:
        for segment in episode["subtasks"]:
            if segment["start"] <= t <= segment["end"]:
                return segment["task"]
        return self.data_args.default_prompt

    def _build_item(self, idx: int) -> Dict[str, torch.Tensor]:
        episode = self.episodes[idx]
        num_steps = len(episode["states"])
        t = random.randint(0, num_steps - 1)

        state = episode["states"][t]
        actions = make_action_chunk(episode["actions"], state, t, self.horizon, self.delta_mask)
        norm_state = quantile_normalize(state, self.norm_stats["state"])
        norm_actions = quantile_normalize(actions, self.norm_stats["actions"])

        wrist_images = self._extract_wrist_images(episode, t)
        subtask = self._subtask_at(episode, t)

        if self.data_args.predict_subtask:
            # The prompt is a fixed question; the subtask is the assistant answer to predict.
            task_text = self.data_args.subtask_question
            assistant_text = subtask
        else:
            # Subtask-input mode: the subtask IS the prompt; no subtask prediction (the VLM is
            # supervised only through FAST). At inference the subtask is provided by the caller.
            task_text = subtask
            assistant_text = None
        prompt = format_robot_prompt(task_text, norm_state)

        # Top-down cam_high (history video, or a single current still if image_history=False),
        # then the current-timestep wrist still(s), then the text.
        if self.data_args.image_history:
            top_frames = self._extract_frames(episode, t)
            content = [{"type": "video", "video": top_frames}]
        else:
            top_img = self._extract_single_frame(episode["video_path"], t, self.data_args.max_pixels)
            top_frames = [top_img]
            content = [{"type": "image", "image": top_img}]
        content += [{"type": "image", "image": img} for img in wrist_images]
        content.append({"type": "text", "text": prompt})
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
                if ids[pos] == 77091:
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
