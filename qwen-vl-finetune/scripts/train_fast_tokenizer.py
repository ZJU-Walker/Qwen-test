"""Fit a custom FAST action tokenizer (arXiv:2501.09747) on our robot data.

The FAST tokenizer must be trained on the *exact* action representation the model
predicts: delta-transformed (joints relative to state, grippers absolute) and
quantile-normalized to ~[-1, 1], restricted to `active_dims`. We reuse the dataset's
own transform helpers so this stays byte-for-byte consistent with training.

Run once, from qwen-vl-finetune/ in the qwen3vl env:

    python -m scripts.train_fast_tokenizer \
        --robot_data_dirs /iris/projects/humanoid/trossen_data/0528_green_yellow_block_mem_merged_bk \
        --active_dims 7:14 --delta_mask 6,-1 --action_horizon 50 \
        --output_dir /iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen

Then pass --fast_tokenizer_path <output_dir> to the action-expert training script.

The norm stats are computed here too and saved as <output_dir>/norm_stats.json: the
tokenizer and the stats are one artifact (FAST is fit in the normalized space the stats
define), and training/serving load them from the tokenizer dir or the checkpoint --
never from a dataset dir (see robot_data._load_or_compute_norm_stats).

DAgger datasets (an `is_intervention` parquet column) contribute only their
human-correction segments, matching the training sampler's min_start gate.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

import numpy as np
import pandas as pd

from qwenvl.data.robot_data import (
    compute_norm_stats,
    make_action_chunk,
    make_delta_mask,
    parse_active_dims,
    quantile_normalize,
)


def scan_state_actions(robot_data_dirs: str, active_dims, train_split: float,
                       action_space: str = "joint", min_episode_len: int = 0,
                       skip_leading_subtask: str = ""):
    """Load (states, actions) per episode. No video needed for tokenizer fitting.

    DAgger episodes (an `is_intervention` parquet column) get `min_start` = the first
    intervention frame: only the human-correction segment feeds the stats and the FAST
    fit, mirroring the training dataset's sampling gate (robot_data.py).

    skip_leading_subtask mirrors the same-named RobotDataArguments gate: episodes whose
    subtask labels start with the given segment (e.g. "waiting") get min_start = its
    end+1; labeled episodes that don't match are dropped, unlabeled ones untouched."""
    import json

    import pyarrow.parquet as papq

    episodes = []
    for root_str in robot_data_dirs.split(","):
        root = Path(root_str.strip())
        if not root.exists():
            print(f"skipping missing root {root}")
            continue
        subtask_labels = {}
        for labels_path in sorted((root / "videos").glob("chunk-*/subtask_labels.json")):
            with open(labels_path) as f:
                subtask_labels.update(json.load(f))
        per_root = []
        for chunk_dir in sorted((root / "data").glob("chunk-*")):
            for pq in sorted(chunk_dir.glob("episode_*.parquet")):
                has_iv = "is_intervention" in papq.read_schema(pq).names
                df = pd.read_parquet(
                    pq, columns=["observation.state", "action"] + (["is_intervention"] if has_iv else [])
                )
                states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
                actions = np.stack(df["action"].to_numpy()).astype(np.float32)
                min_start = 0
                if has_iv:
                    flags = np.stack(df["is_intervention"].to_numpy()).astype(np.float32).ravel() > 0.5
                    min_start = int(flags.argmax()) if flags.any() else 0
                segs = subtask_labels.get(pq.stem + ".mp4", [])
                if skip_leading_subtask and segs:
                    if segs[0]["task"] == skip_leading_subtask and len(segs) > 1:
                        min_start = max(min_start, int(segs[0]["end"]) + 1)
                    else:
                        print(f"dropping {pq.stem}: labels do not start with a "
                              f"'{skip_leading_subtask}' segment, cannot locate the cut")
                        continue
                if active_dims is not None:
                    states, actions = states[:, active_dims], actions[:, active_dims]
                if min_episode_len and len(states) < min_episode_len:
                    continue
                if action_space == "ee6d":
                    from qwenvl.data.ee_repr import joints_to_ee
                    states, actions = joints_to_ee(states), joints_to_ee(actions)
                per_root.append({"states": states, "actions": actions, "min_start": min_start})
        split_idx = int(len(per_root) * train_split)
        episodes.extend(per_root[:split_idx])
    return episodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot_data_dirs", default="/iris/projects/humanoid/trossen_data/0528_green_yellow_block_mem_merged_bk")
    ap.add_argument("--active_dims", default="7:14")
    ap.add_argument("--delta_mask", default="6,-1")
    ap.add_argument("--use_delta_actions", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--action_horizon", type=int, default=50)
    ap.add_argument("--train_split", type=float, default=1.0)
    ap.add_argument("--norm_stats_path", default=None,
                    help="force-load an EXISTING stats file instead of computing fresh "
                         "(legacy reproduction only; normally leave unset)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--action_space", default="joint", choices=["joint", "ee6d"],
                    help="ee6d: convert the selected 7 joint dims to the 10-dim EE "
                         "representation before stats/fit (see qwenvl/data/ee_repr.py)")
    ap.add_argument("--min_episode_len", type=int, default=0,
                    help="skip episodes shorter than this (match the training filter)")
    ap.add_argument("--skip_leading_subtask", default="",
                    help="gate out a leading subtask segment by label (e.g. 'waiting'); "
                         "must match the training run's --skip_leading_subtask so the "
                         "stats/fit cover exactly the trainable timesteps")
    ap.add_argument("--vocab_size", type=int, default=1024)
    ap.add_argument("--scale", type=float, default=10.0, help="FAST DCT scale (tokenizer default 10); higher = finer quantization / more tokens")
    ap.add_argument("--max_chunks", type=int, default=500_000,
                    help="subsample timesteps to at most this many chunks (a 1k-token "
                         "BPE fit does not need all ~16M chunks of a large dataset, and "
                         "building them all would hold tens of GB in RAM)")
    args = ap.parse_args()

    active_dims = parse_active_dims(args.active_dims)
    delta_mask = make_delta_mask(args.delta_mask) if args.use_delta_actions else None

    episodes = scan_state_actions(args.robot_data_dirs, active_dims, args.train_split,
                                  args.action_space, args.min_episode_len,
                                  args.skip_leading_subtask)
    if not episodes:
        raise ValueError(f"No episodes under {args.robot_data_dirs}")
    print(f"scanned {len(episodes)} episodes")

    # Norm stats: computed fresh over exactly the episodes/frames FAST is fit on, and
    # saved into --output_dir alongside the tokenizer -- they are ONE artifact (FAST is
    # fit inside the normalized space the stats define). Training and serving read the
    # stats back from the tokenizer dir / checkpoint, never from a dataset dir, so a new
    # data mix can't overwrite the stats an older served model depends on.
    # --norm_stats_path force-loads an existing file instead (legacy reproduction only).
    import json

    if args.norm_stats_path:
        norm_stats = json.load(open(args.norm_stats_path))
        if "meta" not in norm_stats:
            # The file is re-saved into --output_dir as a TRUSTED artifact; without a meta
            # block its action space could never be verified downstream (the loader in
            # robot_data.py refuses meta-less artifacts for exactly that reason).
            raise ValueError(
                f"{args.norm_stats_path} has no 'meta' block and cannot be minted into a "
                "trusted artifact. Drop --norm_stats_path to compute stats fresh instead."
            )
        print(f"loaded norm stats from {args.norm_stats_path}")
    else:
        print("computing norm stats over the fit episodes")
        norm_stats = compute_norm_stats(episodes, args.action_horizon, delta_mask)
        norm_stats["meta"] = {
            "horizon": args.action_horizon,
            "use_delta_actions": args.use_delta_actions,
            "delta_mask": args.delta_mask,
            "active_dims": args.active_dims,
            "robot_data_dirs": args.robot_data_dirs,
            "train_split": args.train_split,
        }
        if args.action_space != "joint":
            norm_stats["meta"]["action_space"] = args.action_space
        if args.min_episode_len:
            norm_stats["meta"]["min_episode_len"] = args.min_episode_len
        total_min_start = sum(int(ep.get("min_start", 0)) for ep in episodes)
        if total_min_start:
            norm_stats["meta"]["min_start_frames"] = total_min_start

    # Build normalized action chunks; deterministically subsample timesteps on
    # large datasets (see --max_chunks).
    total = sum(len(ep["actions"]) for ep in episodes)
    stride = max(1, -(-total // args.max_chunks))  # ceil div
    if stride > 1:
        print(f"{total} timesteps: taking every {stride}th chunk (~{total // stride})")
    chunks = []
    for ep_i, ep in enumerate(episodes):
        start = int(ep.get("min_start", 0))
        for t in range(start + (ep_i % stride), len(ep["actions"]), stride):
            chunk = make_action_chunk(ep["actions"], ep["states"][t], t, args.action_horizon, delta_mask)
            chunks.append(quantile_normalize(chunk, norm_stats["actions"]))
    chunks = np.stack(chunks).astype(np.float32)  # (N, horizon, action_dim)
    print(f"fitting FAST on {chunks.shape} action chunks; value range [{chunks.min():.2f}, {chunks.max():.2f}]")

    from transformers import AutoProcessor

    base = AutoProcessor.from_pretrained(
        "physical-intelligence/fast", trust_remote_code=True, cache_dir=os.environ["HF_HOME"]
    )
    tok = base.fit(chunks, scale=args.scale, vocab_size=args.vocab_size)

    # Report the token-length distribution -- this sizes the sequence budget.
    enc = tok(chunks[:: max(1, len(chunks) // 2000)])  # subsample for speed
    lens = np.array([len(e) for e in enc])
    print(f"FAST token lengths: mean {lens.mean():.1f}, p50 {np.percentile(lens,50):.0f}, "
          f"p95 {np.percentile(lens,95):.0f}, max {lens.max()}  (vocab_size={args.vocab_size})")

    # Sanity: round-trip reconstruction error.
    dec = np.asarray(tok.decode(enc, time_horizon=args.action_horizon, action_dim=chunks.shape[-1]))
    ref = chunks[:: max(1, len(chunks) // 2000)][: len(dec)]
    print(f"round-trip mean abs error (normalized space): {np.abs(dec - ref).mean():.4f}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(args.output_dir)
    # The stats travel WITH the tokenizer: robot_data.py loads
    # <fast_tokenizer_path>/norm_stats.json as a trusted frozen artifact.
    json.dump(norm_stats, open(Path(args.output_dir) / "norm_stats.json", "w"), indent=2)
    # Stash the fit config so training can sanity-check compatibility.
    meta = {
        "vocab_size": args.vocab_size,
        "scale": args.scale,
        "action_horizon": args.action_horizon,
        "action_dim": int(chunks.shape[-1]),
        "active_dims": args.active_dims,
        "delta_mask": args.delta_mask,
        "action_space": args.action_space,
        "max_fast_len": int(lens.max()),
    }
    json.dump(meta, open(Path(args.output_dir) / "fast_fit_meta.json", "w"), indent=2)
    print(f"saved FAST tokenizer + fast_fit_meta.json to {args.output_dir}")


if __name__ == "__main__":
    main()
