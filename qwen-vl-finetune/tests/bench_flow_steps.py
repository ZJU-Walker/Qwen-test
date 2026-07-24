"""Phase-1: flow-ODE Euler steps 10 vs 5 (vs 3) -- GT-chunk MSE comparison on real episodes.
Teacher-forces the GT subtask so the ONLY variable is the number of flow steps."""
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune")
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

from qwenvl.action_expert.inference import (
    ACTION_DIM, ACTION_HORIZON, build_dataset, load_model, make_prompt, predict_expert, templatize,
)

DEVICE = "cuda"
CKPT_GLOB = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/qwen3_4b_ae_hist_subpred_0714merged_rtc25/checkpoint-*/pytorch_model.bin"
FAST_TOK = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0714merged"
DATA_DIRS = "/iris/projects/humanoid/trossen_data/0714_green_yellow_block_mem_merged"
N_EPISODES = 20
STEP_COUNTS = [10, 5, 3]

ckpts = sorted(glob.glob(CKPT_GLOB), key=lambda p: int(p.split("checkpoint-")[1].split("/")[0]))
ckpt = os.path.dirname(ckpts[-1])
ds = build_dataset(fast_tok=FAST_TOK, data_dirs=DATA_DIRS, image_history=True, predict_subtask=True)
model = load_model(ckpt, ds, DEVICE)

mses = {k: [] for k in STEP_COUNTS}
with torch.inference_mode():
    for idx in range(N_EPISODES):
        ep = ds.episodes[idx]
        t = (idx * 37) % max(1, len(ep["states"]) - ACTION_HORIZON)  # deterministic spread
        state = ep["states"][t]
        gt = ep["actions"][t: t + ACTION_HORIZON].astype(np.float32)
        if len(gt) < ACTION_HORIZON:
            gt = np.concatenate([gt, np.repeat(gt[-1:], ACTION_HORIZON - len(gt), axis=0)])
        frames = ds._extract_frames(ep, t)
        wrist = ds._extract_wrist_images(ep, t)
        subtask = ds._subtask_at(ep, t)
        prompt = make_prompt(ds, state)
        mm = templatize(ds, frames, wrist, prompt, subtask, DEVICE)
        noise = torch.randn(1, ACTION_HORIZON, ACTION_DIM, device=DEVICE,
                            generator=torch.Generator(device=DEVICE).manual_seed(idx))
        for k in STEP_COUNTS:
            act = predict_expert(model, mm, ds, state, noise.clone(), k)
            mses[k].append(float(np.mean((act - gt) ** 2)))
        print(f"[ep {idx}] " + "  ".join(f"{k}steps={mses[k][-1]:.4f}" for k in STEP_COUNTS))

print("\n=== mean GT-chunk MSE over episodes ===")
for k in STEP_COUNTS:
    print(f"  {k:>2} flow steps: {np.mean(mses[k]):.5f}  (median {np.median(mses[k]):.5f})")
