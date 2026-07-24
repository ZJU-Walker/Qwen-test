"""Phase-1 validation: single-prefill (generate_subtask_cached + prefix_key_values fast path)
vs the legacy double-prefill path, on REAL episode observations.

Checks per episode:
  1. subtask text identical (greedy decode must match generate())
  2. input_ids_full token-EXACT vs templatize(..., assistant_text=subtask)  <- validates the
     trailing-"\n" cache-extension assumption structurally
  3. expert actions match with the same noise (tolerance for bf16 kernel-path differences)
  4. wall-clock comparison old vs new
"""
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune")
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

from qwenvl.action_expert.inference import (
    ACTION_DIM, ACTION_HORIZON,
    build_dataset, generate_subtask, generate_subtask_cached, load_model, make_prompt,
    predict_expert, templatize,
)

DEVICE = "cuda"
CKPT_GLOB = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/qwen3_4b_ae_hist_subpred_0714merged_rtc25/checkpoint-*/pytorch_model.bin"
FAST_TOK = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0714merged"
DATA_DIRS = "/iris/projects/humanoid/trossen_data/0714_green_yellow_block_mem_merged"
N_EPISODES = 5
ACTION_TOL = 2e-2  # bf16 prefill-in-one-pass vs incremental-decode kernel differences

ckpts = sorted(glob.glob(CKPT_GLOB), key=lambda p: int(p.split("checkpoint-")[1].split("/")[0]))
ckpt = os.path.dirname(ckpts[-1])
print(f"checkpoint: {ckpt}")

ds = build_dataset(fast_tok=FAST_TOK, data_dirs=DATA_DIRS, image_history=True, predict_subtask=True)
model = load_model(ckpt, ds, DEVICE)
im_end_id = ds.tokenizer.convert_tokens_to_ids("<|im_end|>")

ok = True
with torch.inference_mode():
    for idx in range(N_EPISODES):
        ep = ds.episodes[idx]
        t = len(ep["states"]) // 2
        state = ep["states"][t]
        frames = ds._extract_frames(ep, t)
        wrist = ds._extract_wrist_images(ep, t)
        prompt = make_prompt(ds, state)
        noise = torch.randn(1, ACTION_HORIZON, ACTION_DIM, device=DEVICE,
                            generator=torch.Generator(device=DEVICE).manual_seed(idx))

        # ---- legacy double-prefill path ----
        torch.cuda.synchronize(); t0 = time.perf_counter()
        mm_gen = templatize(ds, frames, wrist, prompt, None, DEVICE)
        sub_old = generate_subtask(model, mm_gen, ds, im_end_id, 48)
        mm_old = templatize(ds, frames, wrist, prompt, sub_old, DEVICE)
        act_old = predict_expert(model, mm_old, ds, state, noise.clone(), 10)
        torch.cuda.synchronize(); old_ms = (time.perf_counter() - t0) * 1e3

        # ---- new single-prefill path ----
        torch.cuda.synchronize(); t0 = time.perf_counter()
        mm_gen2 = templatize(ds, frames, wrist, prompt, None, DEVICE)
        sub_new, cache, ids_full, pos_full = generate_subtask_cached(model, mm_gen2, ds, im_end_id, 48)
        act_new = predict_expert(model, None, ds, state, noise.clone(), 10,
                                 prefix_cache=(cache, ids_full, pos_full))
        torch.cuda.synchronize(); new_ms = (time.perf_counter() - t0) * 1e3

        # ---- checks ----
        sub_match = sub_old == sub_new
        ids_ref = mm_old["input_ids"]
        tok_exact = ids_full.shape == ids_ref.shape and bool(torch.equal(ids_full, ids_ref))
        pos_exact = bool(torch.equal(pos_full, mm_old["position_ids"]))
        adiff = float(np.max(np.abs(act_old - act_new)))
        line_ok = sub_match and tok_exact and pos_exact and adiff < ACTION_TOL
        ok = ok and line_ok
        print(f"[ep {idx}] subtask={'MATCH' if sub_match else f'DIFF ({sub_old!r} vs {sub_new!r})'} | "
              f"tokens={'EXACT' if tok_exact else 'MISMATCH'} | positions={'EXACT' if pos_exact else 'MISMATCH'} | "
              f"action maxdiff={adiff:.2e} | old={old_ms:.0f} ms new={new_ms:.0f} ms "
              f"({'OK' if line_ok else 'FAIL'})")

print("\nALL SINGLE-PREFILL CHECKS PASS" if ok else "\nFAILURES -- do not deploy the cached path")
sys.exit(0 if ok else 1)
