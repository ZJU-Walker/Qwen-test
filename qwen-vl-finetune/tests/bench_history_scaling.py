"""Phase-0 benchmark: per-phase inference latency vs history length (10/25/50/100 frames).

Uses the REAL checkpoint + processor so token counts and kernels are authentic; frames are
random (content doesn't affect timing). For each num_frames, measures (2nd of 2 runs, after
warmup):
  templatize_ms      CPU processor work (resize/tokenize)
  prefill_ms         one full ViT+LM prefill (what _prefix_forward costs)
  vision_ms          the ViT share of that prefill (forward hooks)
  decode16_ms        16 forced decode steps -> per-token autoregressive cost
  flow_loop_ms       10 expert Euler steps
  seq_len            total prefix tokens
"""
import glob
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune")
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

from qwenvl.action_expert.inference import (
    build_dataset, load_model, make_prompt, predict_expert, resize_wrist, templatize,
)

DEVICE = "cuda"
CKPT_GLOB = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/qwen3_4b_ae_hist_subpred_0714merged_rtc25/checkpoint-*/pytorch_model.bin"
FAST_TOK = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0714merged"
DATA_DIRS = "/iris/projects/humanoid/trossen_data/0714_green_yellow_block_mem_merged"
FRAME_COUNTS = [10, 25, 50, 100]
DECODE_TOKENS = 16

ckpts = sorted(glob.glob(CKPT_GLOB), key=lambda p: int(p.split("checkpoint-")[1].split("/")[0]))
ckpt = os.path.dirname(ckpts[-1])
print(f"checkpoint: {ckpt}")

ds = build_dataset(fast_tok=FAST_TOK, data_dirs=DATA_DIRS, image_history=True, predict_subtask=True)
model = load_model(ckpt, ds, DEVICE)
im_end_id = ds.tokenizer.convert_tokens_to_ids("<|im_end|>")

vision_ms = {"t": 0.0}
def _pre(m, a):
    torch.cuda.synchronize(); vision_ms["t0"] = time.perf_counter()
def _post(m, a, o):
    torch.cuda.synchronize(); vision_ms["t"] += (time.perf_counter() - vision_ms["t0"]) * 1e3
model.vlm.visual.register_forward_pre_hook(_pre)
model.vlm.visual.register_forward_hook(_post)

rng = np.random.default_rng(0)
state = rng.standard_normal(7).astype(np.float32)
wrist = resize_wrist(Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)))
prompt = make_prompt(ds, state)

rows = []
with torch.inference_mode():
    for nf in FRAME_COUNTS:
        frames = [Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)) for _ in range(nf)]
        rec = {}
        for run in range(2):  # run 0 = warmup, run 1 = measured
            t0 = time.perf_counter()
            mm = templatize(ds, frames, [wrist], prompt, None, DEVICE)
            rec["templatize_ms"] = (time.perf_counter() - t0) * 1e3
            rec["seq_len"] = mm["input_ids"].shape[1]

            # full prefill (ViT + LM) == what _prefix_forward / generate's prefill costs
            vision_ms["t"] = 0.0
            torch.cuda.synchronize(); t0 = time.perf_counter()
            model._prefix_forward(mm["input_ids"], torch.ones_like(mm["input_ids"]),
                                  mm["position_ids"], mm["pixel_values"], mm["image_grid_thw"],
                                  mm["pixel_values_videos"], mm["video_grid_thw"])
            torch.cuda.synchronize()
            rec["prefill_ms"] = (time.perf_counter() - t0) * 1e3
            rec["vision_ms"] = vision_ms["t"]

            # forced 16-token decode (per-token autoregressive cost at this context length)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            model.vlm.generate(
                input_ids=mm["input_ids"], attention_mask=torch.ones_like(mm["input_ids"]),
                pixel_values=mm["pixel_values"], image_grid_thw=mm["image_grid_thw"],
                pixel_values_videos=mm["pixel_values_videos"], video_grid_thw=mm["video_grid_thw"],
                min_new_tokens=DECODE_TOKENS, max_new_tokens=DECODE_TOKENS,
                do_sample=False, num_beams=1,
                pad_token_id=ds.tokenizer.pad_token_id or im_end_id,
            )
            torch.cuda.synchronize()
            gen_total = (time.perf_counter() - t0) * 1e3
            rec["decode16_ms"] = gen_total - rec["prefill_ms"]  # generate's prefill ~= measured prefill

            # expert: prefix forward + 10 Euler steps (timings dict splits them)
            tms = {}
            predict_expert(model, mm, ds, state, None, 10, timings=tms)
            rec["flow_loop_ms"] = tms["flow_loop_ms"]
        rec["nf"] = nf
        rows.append(rec)
        print(f"[{nf:3d} frames] seq={rec['seq_len']:5d}  templatize={rec['templatize_ms']:7.1f}  "
              f"prefill={rec['prefill_ms']:6.1f} (vision {rec['vision_ms']:6.1f})  "
              f"decode16={rec['decode16_ms']:6.1f}  flow10={rec['flow_loop_ms']:6.1f}")

print("\n=== summary (ms; mode-c request today = templatize*2 + prefill*2 + decode + flow) ===")
print(f"{'frames':>7} {'seq':>6} {'templz':>8} {'prefill':>8} {'vision':>8} {'per-tok':>8} {'flow10':>8} {'~mode-c total':>14}")
for r in rows:
    per_tok = r["decode16_ms"] / DECODE_TOKENS
    total = 2 * r["templatize_ms"] + 2 * r["prefill_ms"] + per_tok * 6 + r["flow_loop_ms"]
    print(f"{r['nf']:>7} {r['seq_len']:>6} {r['templatize_ms']:>8.1f} {r['prefill_ms']:>8.1f} "
          f"{r['vision_ms']:>8.1f} {per_tok:>8.1f} {r['flow_loop_ms']:>8.1f} {total:>14.1f}")
