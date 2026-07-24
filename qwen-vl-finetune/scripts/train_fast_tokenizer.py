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


def scan_state_actions(robot_data_dirs: str, active_dims, train_split: float):
    """Load (states, actions) per episode. No video needed for tokenizer fitting."""
    episodes = []
    for root_str in robot_data_dirs.split(","):
        root = Path(root_str.strip())
        if not root.exists():
            print(f"skipping missing root {root}")
            continue
        per_root = []
        for chunk_dir in sorted((root / "data").glob("chunk-*")):
            for pq in sorted(chunk_dir.glob("episode_*.parquet")):
                df = pd.read_parquet(pq, columns=["observation.state", "action"])
                states = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
                actions = np.stack(df["action"].to_numpy()).astype(np.float32)
                if active_dims is not None:
                    states, actions = states[:, active_dims], actions[:, active_dims]
                per_root.append({"states": states, "actions": actions})
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
    ap.add_argument("--norm_stats_path", default=None, help="defaults to <first root>/qwen_action_expert_norm_stats.json")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--vocab_size", type=int, default=1024)
    ap.add_argument("--scale", type=float, default=10.0, help="FAST DCT scale (tokenizer default 10); higher = finer quantization / more tokens")
    args = ap.parse_args()

    active_dims = parse_active_dims(args.active_dims)
    delta_mask = make_delta_mask(args.delta_mask) if args.use_delta_actions else None

    episodes = scan_state_actions(args.robot_data_dirs, active_dims, args.train_split)
    if not episodes:
        raise ValueError(f"No episodes under {args.robot_data_dirs}")
    print(f"scanned {len(episodes)} episodes")

    # Norm stats: reuse the cached file if present so FAST sees exactly the training space.
    import json

    stats_path = Path(args.norm_stats_path) if args.norm_stats_path else (
        Path(args.robot_data_dirs.split(",")[0].strip()) / "qwen_action_expert_norm_stats.json"
    )
    if stats_path.exists():
        norm_stats = json.load(open(stats_path))
        print(f"loaded norm stats from {stats_path}")
    else:
        print(f"norm stats not found at {stats_path}; computing (should match the dataset's)")
        norm_stats = compute_norm_stats(episodes, args.action_horizon, delta_mask)

    # Build every normalized action chunk the model would ever train on.
    chunks = []
    for ep in episodes:
        for t in range(len(ep["actions"])):
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
    # Stash the fit config so training can sanity-check compatibility.
    meta = {
        "vocab_size": args.vocab_size,
        "scale": args.scale,
        "action_horizon": args.action_horizon,
        "action_dim": int(chunks.shape[-1]),
        "active_dims": args.active_dims,
        "delta_mask": args.delta_mask,
        "max_fast_len": int(lens.max()),
    }
    json.dump(meta, open(Path(args.output_dir) / "fast_fit_meta.json", "w"), indent=2)
    print(f"saved FAST tokenizer + fast_fit_meta.json to {args.output_dir}")


if __name__ == "__main__":
    main()
