"""Per-phase latency breakdown of the /infer request path, joint vs ee6d checkpoints.

Mirrors qwen_action_expert_server.py's insulated mode-c flow (single-prefill cache reuse):
  jpeg decode/resize -> state_to_model -> templatize -> generate_subtask_cached (prefill +
  subtask decode) -> predict_expert (expert prefill ~0 via cache + flow loop) -> IK.
Also times the subtask-skip variant (--subtask_every 0: no decode, prefill inside the
expert call) and the CPU-side RTC prefix conversion.

Run ON the GPU node (shares the GPU with training -- memory hard-capped so an OOM hits
this process, never the training run):
  CUDA_VISIBLE_DEVICES=1 python tests/bench_infer_phases.py
"""

import io
import json
import os
import statistics
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qwenvl.action_expert.inference import (  # noqa: E402
    build_dataset, generate_subtask_cached, load_model, load_norm_stats_path,
    load_repr_config, load_visual_budget, make_prompt, model_actions_to_robot,
    predict_expert, state_to_model, subtask_token_mask_for, templatize,
)

ROOT = "/iris/projects/humanoid/ke/Qwen3-VL"
DATA_DIRS = ("/iris/projects/humanoid/trossen_data/0717_green_yellow_block_mem_merged,"
             "/iris/projects/humanoid/trossen_data/0731_green_yellow_merged")
CONFIGS = [
    dict(name="joint (ckpt-23000, 1 wrist, dim 7)",
         ckpt=f"{ROOT}/checkpoints/qwen3_4b_ae_hist_subpred_0717m_0731gy_rtc10_subinsul_L18_vis16_constlr/checkpoint-23000",
         fast_tok=f"{ROOT}/checkpoints/fast_tokenizer_trossen_0717merged_0731gy"),
    dict(name="ee6d (ckpt-18000, 2 wrists, dim 10)",
         ckpt=f"{ROOT}/checkpoints/qwen3_4b_ae_hist_subpred_0717m_0731gy_ee6d_rtc10_subinsul_L18_vis16_constlr/checkpoint-18000",
         fast_tok=f"{ROOT}/checkpoints/fast_tokenizer_trossen_0717m_0731gy_ee6d"),
]
REPS, WARMUP, NSTEPS = 15, 3, 10


def resize_wrist(img, budget):
    w, h = img.size
    if w * h > budget:
        s = (budget / (w * h)) ** 0.5
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    return img


def med(xs):
    return statistics.median(xs)


def bench(cfg):
    print(f"\n================ {cfg['name']} ================", flush=True)
    budget = load_visual_budget(cfg["ckpt"])
    # Stamp absent (pre-fix joint checkpoint) -> still insulated: BOTH benchmarked runs
    # trained with EXPERT_ATTENDS_SUBTASK=False (the _subinsul suffix).
    insulated = not budget.pop("expert_attends_subtask", False)
    repr_cfg = load_repr_config(cfg["ckpt"])
    ds = build_dataset(fast_tok=cfg["fast_tok"], data_dirs=DATA_DIRS, budget=budget,
                       norm_stats_path=load_norm_stats_path(cfg["ckpt"]), repr_cfg=repr_cfg)
    model = load_model(cfg["ckpt"], ds, "cuda", expert_attends_subtask=False, use_ema=True)
    im_end = ds.tokenizer.convert_tokens_to_ids("<|im_end|>")
    n_wrists = len(ds.wrist_cameras)
    print(f"  action_space={repr_cfg['action_space']} dim={repr_cfg['action_dim']} wrists={n_wrists} "
          f"insulated={insulated or '(assumed)'}", flush=True)

    rng = np.random.default_rng(0)
    jpegs = []
    for _ in range(ds.num_frames + n_wrists):
        buf = io.BytesIO()
        Image.fromarray(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)).save(buf, "JPEG", quality=90)
        jpegs.append(buf.getvalue())
    state7 = np.zeros(7, dtype=np.float32)
    prefix7 = np.zeros((8, 7), dtype=np.float32)

    T = {k: [] for k in ("jpeg", "templatize", "gen_prefill", "decode", "flow", "expert_prefill_cached",
                         "skip_total", "ik", "state_conv")}
    dec_tokens = 0
    for rep in range(WARMUP + REPS):
        rec = rep >= WARMUP

        t = time.perf_counter()
        frames = [Image.open(io.BytesIO(b)).convert("RGB") for b in jpegs[:ds.num_frames]]
        wrists = [resize_wrist(Image.open(io.BytesIO(b)).convert("RGB"), ds.data_args.wrist_max_pixels)
                  for b in jpegs[ds.num_frames:]]
        if rec: T["jpeg"].append((time.perf_counter() - t) * 1e3)

        t = time.perf_counter()
        model_state = state_to_model(ds, state7)
        if rec: T["state_conv"].append((time.perf_counter() - t) * 1e3)

        prompt = make_prompt(ds, model_state)
        t = time.perf_counter()
        mm_gen = templatize(ds, frames, wrists, prompt, None, "cuda")
        if rec: T["templatize"].append((time.perf_counter() - t) * 1e3)

        # --- default serving path: single prefill + subtask decode + cached expert ---
        tm = {}
        with torch.inference_mode():
            _sub, cache, ids_full, pos_full = generate_subtask_cached(model, mm_gen, ds, im_end, 16, timings=tm)
        mm = dict(input_ids=ids_full, position_ids=pos_full, pixel_values=None,
                  image_grid_thw=None, pixel_values_videos=None, video_grid_thw=None)
        stask = subtask_token_mask_for(ids_full) if insulated else None
        tm2 = {}
        with torch.inference_mode():
            actions = predict_expert(model, mm, ds, model_state, None, NSTEPS,
                                     action_prefix_abs=prefix7, prefix_cache=(cache, ids_full, pos_full),
                                     subtask_token_mask=stask, timings=tm2)
        if rec:
            T["gen_prefill"].append(tm.get("gen_prefill_ms", 0.0))
            T["decode"].append(tm.get("decode_ms", 0.0))
            dec_tokens = tm.get("decode_tokens", dec_tokens)
            T["expert_prefill_cached"].append(tm2.get("expert_prefill_ms", 0.0))
            T["flow"].append(tm2.get("flow_loop_ms", 0.0))

        t = time.perf_counter()
        joints, ik_fail = model_actions_to_robot(ds, actions, state7, ik_max_iters=60)
        if rec: T["ik"].append((time.perf_counter() - t) * 1e3)

        # --- subtask-skip path (--subtask_every 0): no decode, prefill inside the expert ---
        mm_skip = templatize(ds, frames, wrists, prompt, None, "cuda")
        stask2 = subtask_token_mask_for(mm_skip["input_ids"]) if insulated else None
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.inference_mode():
            predict_expert(model, mm_skip, ds, model_state, None, NSTEPS,
                           action_prefix_abs=prefix7, subtask_token_mask=stask2)
        torch.cuda.synchronize()
        if rec: T["skip_total"].append((time.perf_counter() - t) * 1e3)

    print(f"  seq_len={ids_full.shape[-1]} tokens | subtask decode length={dec_tokens} tok | ik_failures={ik_fail}")
    order = [("jpeg decode + wrist resize (CPU)", "jpeg"), ("state->model conv (CPU)", "state_conv"),
             ("templatize incl. vision preproc (CPU)", "templatize"), ("VLM+ViT prefill (GPU)", "gen_prefill"),
             ("subtask decode (GPU, autoregressive)", "decode"), ("expert prefill w/ cache reuse (GPU)", "expert_prefill_cached"),
             ("flow loop 10 steps (GPU)", "flow"), ("IK chunk->joints (CPU)", "ik")]
    total = 0.0
    for label, k in order:
        m = med(T[k]); total += m
        print(f"  {label:44s} {m:8.1f} ms")
    print(f"  {'TOTAL (default path, sum of medians)':44s} {total:8.1f} ms  (~{1000/total:.1f} Hz)")
    skip = med(T["skip_total"]) + med(T["jpeg"]) + med(T["templatize"]) + med(T["ik"]) + med(T["state_conv"])
    print(f"  {'TOTAL (subtask_every=0 skip path)':44s} {skip:8.1f} ms  (~{1000/skip:.1f} Hz)")

    del model
    torch.cuda.empty_cache()
    return None


if __name__ == "__main__":
    torch.cuda.set_per_process_memory_fraction(0.30, 0)  # ~43 GB cap: we OOM before training does
    for cfg in CONFIGS:
        bench(cfg)
    print("\nDONE", flush=True)
