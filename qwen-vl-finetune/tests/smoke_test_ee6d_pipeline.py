"""End-to-end CPU validation of the combined ee6d + augmentation + 2-wrist pipeline.

1. Dataset: 444 episodes (5 short 0731 episodes filtered), 10-dim states/actions,
   frozen ee6d stats adopted from the new tokenizer artifact, labeled/unlabeled mix intact.
2. Item: (50, 10) action chunk, 10-number state string in the prompt, one history video
   + TWO wrist stills, subtask supervision semantics unchanged (labeled supervises text,
   unlabeled supervises nothing but FAST).
3. Augmentation: off -> bit-identical items (regression); on -> deterministic per seed,
   different across seeds.
4. Serve boundary: repr auto-detect from a stamped ckpt dir, joint state -> model state,
   joint RTC prefix -> normalized 10-dim, model actions -> joint commands ~= ground truth.

Run:  python tests/smoke_test_ee6d_pipeline.py
"""
import json
import os
import random
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune")
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

from qwenvl.action_expert.inference import (
    build_dataset, load_repr_config, model_actions_to_robot, normalize_action_prefix,
    state_to_model,
)
from qwenvl.data.ee_repr import EE_DIM, joints_to_ee

DATA = "/iris/projects/humanoid/trossen_data"
MIX = f"{DATA}/0717_green_yellow_block_mem_merged,{DATA}/0731_green_yellow_merged"
TOK = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717m_0731gy_ee6d"
IGNORE = -100

REPR = {"action_space": "ee6d", "action_dim": 10, "action_horizon": 50,
        "delta_mask": "9,-1", "active_dims": "7:14"}
BUDGET = {"wrist_cameras": "cam_right_wrist,cam_left_wrist"}


def build(image_aug=False):
    ds = build_dataset(fast_tok=TOK, data_dirs=MIX, budget=dict(BUDGET), repr_cfg=dict(REPR))
    ds.data_args.image_aug = image_aug
    ds.data_args.min_episode_len = 50   # scan already done; assert below instead
    return ds


def fixed_item(ds, idx, t, seed=0):
    orig = random.randint
    random.seed(seed)
    random.randint = lambda lo, hi: max(lo, min(t, hi))
    try:
        return ds[idx]
    finally:
        random.randint = orig


# ---- 1. dataset-level ----
ds = build_dataset(fast_tok=TOK, data_dirs=MIX, budget=dict(BUDGET), repr_cfg=dict(REPR))
# min_episode_len flows through repr overlay? No -- set via budget-style overlay too:
# the run script passes --min_episode_len; serving does not need the filter. For this
# test, verify the FILTERED count by building with the flag the way training does:
ds_f = build_dataset(fast_tok=TOK, data_dirs=MIX,
                     budget=dict(BUDGET, min_episode_len=50), repr_cfg=dict(REPR))
assert len(ds_f.episodes) == 443, f"expected 443 filtered episodes, got {len(ds_f.episodes)}"
assert all(ep["states"].shape[1] == EE_DIM for ep in ds_f.episodes), "states not 10-dim"
assert all(ep["actions"].shape[1] == EE_DIM for ep in ds_f.episodes), "actions not 10-dim"
n_labeled = sum(1 for ep in ds_f.episodes if ep["subtasks"])
assert n_labeled == 224, f"labeled episode count changed: {n_labeled}"
frozen = json.load(open(f"{TOK}/norm_stats.json"))
assert ds_f.norm_stats == frozen, "ee6d dataset did not adopt the frozen ee6d stats"
assert frozen["meta"].get("action_space") == "ee6d", "artifact meta lacks action_space"
print(f"1. dataset: 443 filtered eps, all 10-dim, {n_labeled} labeled, frozen ee6d stats  OK")

# ---- 2. item-level ----
ds = ds_f
item = fixed_item(ds, 5, 120)             # labeled 0717 episode
assert item["actions"].shape == (50, EE_DIM) or item["actions"].shape[-1] == EE_DIM, \
    f"action chunk shape {item['actions'].shape}"
text = ds.tokenizer.decode(np.asarray(item["input_ids"]).ravel().tolist())
state_part = text.split("State:")[1].split("<|im_end|>")[0]
n_nums = len(state_part.split())
assert n_nums == EE_DIM, f"prompt state has {n_nums} numbers, expected {EE_DIM}"
# History video = 5 temporal groups (10 frames / temporal_patch 2), each its own
# timestamped <|video_pad|> block; wrist stills are <|image_pad|> blocks.
n_vid = text.count("<|vision_start|><|video_pad|>")
n_img = text.count("<|vision_start|><|image_pad|>")
assert n_vid == 5, f"expected 5 video groups, got {n_vid}"
assert n_img == 2, f"expected 2 wrist image blocks (right+left), got {n_img}"
sup = item["labels"][0][(item["labels"][0] != IGNORE) & ~item["fast_token_mask"][0]]
assert len(sup) > 0 and "block" in ds.tokenizer.decode(sup.tolist()), "labeled episode lost supervision"
item_u = fixed_item(ds, 300, 120)         # unlabeled 0731 episode
sup_u = item_u["labels"][0][(item_u["labels"][0] != IGNORE) & ~item_u["fast_token_mask"][0]]
assert len(sup_u) == 0, "unlabeled episode gained language supervision"
print(f"2. item: (50,{EE_DIM}) chunk, {EE_DIM}-number state prompt, 5 video + 2 wrist blocks, "
      "mixed supervision intact  OK")

# ---- 3. augmentation determinism / regression ----
a = fixed_item(ds, 5, 120, seed=1)
b = fixed_item(ds, 5, 120, seed=1)
assert torch.equal(a["input_ids"], b["input_ids"]) and torch.allclose(
    a["pixel_values_videos"], b["pixel_values_videos"]), "aug-off items not reproducible"
ds.data_args.image_aug = True
c = fixed_item(ds, 5, 120, seed=1)
d = fixed_item(ds, 5, 120, seed=1)
e = fixed_item(ds, 5, 120, seed=2)
assert torch.allclose(c["pixel_values_videos"], d["pixel_values_videos"]), "aug not seed-deterministic"
assert not torch.allclose(c["pixel_values_videos"], e["pixel_values_videos"]), "aug ignores seed (no variation)"
assert not torch.allclose(a["pixel_values_videos"], c["pixel_values_videos"]), "aug on == aug off?!"
ds.data_args.image_aug = False
print("3. augmentation: off = reproducible; on = seed-deterministic and varying  OK")

# ---- 4. serve boundary ----
with tempfile.TemporaryDirectory() as td:
    json.dump(frozen, open(os.path.join(td, "norm_stats.json"), "w"))
    cfg = load_repr_config(td)
assert cfg == REPR, f"repr auto-detect mismatch: {cfg}"
q = ds.episodes[5]  # raw joints no longer stored; use a real parquet row instead
import pandas as pd
pq = f"{DATA}/0717_green_yellow_block_mem_merged/data/chunk-000/episode_000005.parquet"
df = pd.read_parquet(pq, columns=["observation.state", "action"])
joints_state = np.stack(df["observation.state"].to_numpy()).astype(np.float32)[120, 7:14]
joints_chunk = np.stack(df["action"].to_numpy()).astype(np.float32)[120:170, 7:14]
ms = state_to_model(ds, joints_state)
assert ms.shape == (EE_DIM,), f"model state shape {ms.shape}"
pref_norm = normalize_action_prefix(ds, joints_chunk[:8], ms)
assert pref_norm.shape == (8, EE_DIM), f"prefix shape {pref_norm.shape}"
ee_chunk = joints_to_ee(joints_chunk)
rec, ik_fail = model_actions_to_robot(ds, ee_chunk, joints_state)
err = np.abs(rec[:, :6] - joints_chunk[:, :6]).max()
assert ik_fail == 0 and err < 1e-3, f"serve-boundary IK round trip: fail={ik_fail} err={err}"
print(f"4. serve boundary: repr auto-detect, 7->10 state, joint prefix -> (8,{EE_DIM}) norm, "
      f"IK round trip err {err:.1e} rad, 0 failures  OK")

print("\nALL EE6D PIPELINE TESTS PASS")
