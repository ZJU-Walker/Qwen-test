"""Quantify two precision suspects on the REAL 0731gy checkpoint-9000:
1. dtype: bf16 serving (current) vs full-fp32 reference -- same noise, same obs.
2. integration: 10 Euler steps (serving default) vs 50 steps, per dtype.
Both runs use SDPA attention so kernel choice doesn't pollute the dtype comparison.
Errors reported in normalized units, radians, approx end-effector mm (0.30 m lever arm),
and as a fraction of each joint's q01..q99 motion range.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune")
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

from qwenvl.action_expert.inference import (
    ACTION_DIM, ACTION_HORIZON, build_dataset, load_model, load_norm_stats_path,
    load_visual_budget, make_prompt, predict_expert, subtask_token_mask_for, templatize,
)

CKPT = ("/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/"
        "qwen3_4b_ae_hist_subpred_0717m_0731gy_rtc10_subinsul_L18_vis16_constlr/checkpoint-9000")
TOK = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717merged_0731gy"
MIX = ("/iris/projects/humanoid/trossen_data/0717_green_yellow_block_mem_merged,"
       "/iris/projects/humanoid/trossen_data/0731_green_yellow_merged")
DEVICE = "cuda"

ds = build_dataset(fast_tok=TOK, data_dirs=MIX, budget=load_visual_budget(CKPT),
                   norm_stats_path=load_norm_stats_path(CKPT))
model = load_model(CKPT, ds, DEVICE, expert_attends_subtask=False)
# Force SDPA everywhere so the bf16-vs-fp32 comparison isn't polluted by FA2-vs-SDPA
# kernel differences (FA2 does not support fp32 at all).
for cfg in (model.vlm.config, getattr(model.vlm, "model", model.vlm).config):
    cfg._attn_implementation = "sdpa"

rng = np.random.default_rng(0)
frames = [Image.fromarray(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)) for _ in range(10)]
wrist = [Image.fromarray(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8))]
state = np.asarray(ds.norm_stats["state"]["q99"], dtype=np.float32)
noise = torch.randn(1, ACTION_HORIZON, ACTION_DIM, generator=torch.Generator().manual_seed(0))

q01 = np.asarray(ds.norm_stats["actions"]["q01"]); q99 = np.asarray(ds.norm_stats["actions"]["q99"])
rng_per_dim = q99 - q01  # normalized-actions range per dim (delta space for joints)


def run(steps):
    mm = templatize(ds, frames, wrist, make_prompt(ds, state), None, DEVICE)
    stask = subtask_token_mask_for(mm["input_ids"])
    with torch.inference_mode():
        return predict_expert(model, mm, ds, state, noise.clone().to(DEVICE), steps,
                              subtask_token_mask=stask)


def report(tag, a, b):
    d = np.abs(a - b)              # absolute joint units (radians for arm dims)
    rad = d[:, :6]                 # 6 arm joints (dim 7 = gripper)
    print(f"{tag}: max {rad.max():.5f} rad ({np.degrees(rad.max()):.3f} deg, "
          f"~{rad.max() * 300:.2f} mm at 0.30 m)   mean {rad.mean():.5f} rad (~{rad.mean() * 300:.2f} mm)   "
          f"gripper max {d[:, 6].max():.5f}")


a10 = run(10)
a50 = run(50)
print("== bf16 (serving config, SDPA) ==")
report("  10 vs 50 Euler steps        ", a10, a50)

model.float()
b10 = run(10)
b50 = run(50)
print("== fp32 reference (SDPA) ==")
report("  10 vs 50 Euler steps (fp32) ", b10, b50)
print("== dtype effect ==")
report("  bf16 vs fp32 @ 10 steps     ", a10, b10)
report("  bf16 vs fp32 @ 50 steps     ", a50, b50)

# context: motion range per arm joint in raw units
motion = np.abs(np.asarray(ds.norm_stats["state"]["q99"]) - np.asarray(ds.norm_stats["state"]["q01"]))[:6]
print(f"context: arm-joint q01..q99 motion ranges (rad): {np.array2string(motion, precision=3)}")
print(f"dtype error as % of motion range: max {100 * np.abs(a10 - b10)[:, :6].max() / motion.max():.2f}%")
print("DONE")
