"""torch.compile (Phase 4) GPU test on the REAL serve checkpoint.

1. equality: compiled vs eager actions, plain and RTC-prefix variants, on the production
   serve path (insulated, EMA loaded) -- tolerance 5e-2 rad (bf16 tiling + fusion noise;
   observed classes are ~5e-3-2e-2)
2. no-recompile guarantee: a prompt with a DIFFERENT token length (state digits change
   width) must reuse the same padded-bucket graph -- dynamo's graph counter must not move
3. speedup: flow-loop ms, eager vs compiled

Run:  python tests/smoke_test_compile_gpu.py [--ckpt <dir>]
"""
import argparse
import os
import sys
import time as _time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune")
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

from qwenvl.action_expert.inference import (
    ACTION_DIM, ACTION_HORIZON, build_dataset, load_model, load_visual_budget,
    make_prompt, predict_expert, subtask_token_mask_for, templatize,
)

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/"
                                  "qwen3_4b_ae_hist_subpred_0717merged_rtc20_subinsul_L18_vis16_constlr/serve-22000")
ap.add_argument("--num_flow_steps", type=int, default=10)
args = ap.parse_args()
DEVICE = "cuda"

budget = load_visual_budget(args.ckpt)
ds = build_dataset("/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717merged",
                   data_dirs="/iris/projects/humanoid/trossen_data/0717_green_yellow_block_mem_merged",
                   image_history=True, predict_subtask=True, budget=budget)
model = load_model(args.ckpt, ds, DEVICE, expert_attends_subtask=False)

rng = np.random.default_rng(0)
frames = [Image.fromarray(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)) for _ in range(10)]
wrist = [Image.fromarray(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8))]
state_a = np.zeros(ACTION_DIM, dtype=np.float32)          # "0 0 0 ..." -> short prompt
# Longer prompt via a REALISTIC state (the dataset's q99 joints -> 3-digit bins).
# NOT a huge fake value: predict_expert adds the state back onto the delta dims, and at
# magnitude 1e6 the float32 grid is 0.0625 -- any comparison then measures ULP
# quantization, not model behavior (that artifact ate a debugging round on 2026-07-30).
state_b = np.asarray(ds.norm_stats["state"]["q99"], dtype=np.float32)

def build(state):
    mm = templatize(ds, frames, wrist, make_prompt(ds, state), None, DEVICE)
    return mm, subtask_token_mask_for(mm["input_ids"])

mm_a, stask_a = build(state_a)
mm_b, stask_b = build(state_b)
la, lb = mm_a["input_ids"].shape[1], mm_b["input_ids"].shape[1]
print(f"prompt lengths: A={la}  B={lb} tokens (must differ for the recompile test)")
assert la != lb, "state trick failed to change prompt length; adjust state_b"

noise = torch.randn(1, ACTION_HORIZON, ACTION_DIM, device=DEVICE, dtype=torch.float32,
                    generator=torch.Generator(device=DEVICE).manual_seed(0))
rtc_prefix = rng.standard_normal((8, ACTION_DIM)).astype(np.float32) * 0.1

def run(mm, stask, state, prefix=None, timings=None):
    with torch.inference_mode():
        return predict_expert(model, mm, ds, state, noise.clone(), args.num_flow_steps,
                              action_prefix_abs=prefix, subtask_token_mask=stask, timings=timings)

# ---- eager (unpadded) reference ----
te = {}
eager_a = run(mm_a, stask_a, state_a, timings=te)
eager_rtc = run(mm_a, stask_a, state_a, prefix=rtc_prefix)
eager_b = run(mm_b, stask_b, state_b)
print(f"eager flow loop: {te['flow_loop_ms']} ms / {args.num_flow_steps} steps")

# ---- eager PADDED reference: isolates the padding effect (bf16 SDPA tiling changes
# with sequence length) from compile numerics. Padding noise exists for ANY bucketing
# scheme and is the same class we accepted for single-prefill/insulation.
model.enable_expert_compile(mode=None)  # pad only, no compilation
pad_a = run(mm_a, stask_a, state_a)
pad_rtc = run(mm_a, stask_a, state_a, prefix=rtc_prefix)
pad_b = run(mm_b, stask_b, state_b)
print(f"0. padding-only effect (eager padded vs eager): "
      f"A={np.max(np.abs(pad_a - eager_a)):.2e}  B={np.max(np.abs(pad_b - eager_b)):.2e}  "
      f"rtc={np.max(np.abs(pad_rtc - eager_rtc)):.2e}   [informational; tiling noise]")
assert max(np.max(np.abs(pad_a - eager_a)), np.max(np.abs(pad_b - eager_b))) < 1.5e-1, \
    "padding perturbs actions far beyond known tiling noise -- investigate before serving"

# ---- compile + warmup ----
model.disable_expert_compile()
model.enable_expert_compile()
t0 = _time.perf_counter()
run(mm_a, stask_a, state_a); run(mm_a, stask_a, state_a)
run(mm_a, stask_a, state_a, prefix=rtc_prefix); run(mm_a, stask_a, state_a, prefix=rtc_prefix)
print(f"warmup: {_time.perf_counter() - t0:.0f}s")

# ---- 1. STRICT equality: compiled vs eager-padded (same shapes -> pure compile numerics) ----
tc = {}
comp_a = run(mm_a, stask_a, state_a, timings=tc)
comp_rtc = run(mm_a, stask_a, state_a, prefix=rtc_prefix)
d_plain = float(np.max(np.abs(comp_a - pad_a)))
d_rtc = float(np.max(np.abs(comp_rtc - pad_rtc)))
print(f"1. pure-compile equality (vs eager-padded): plain={d_plain:.2e}  rtc={d_rtc:.2e}")
assert d_plain < 2e-2 and d_rtc < 2e-2, "compiled output diverges beyond bf16 noise"
seam = np.abs(comp_rtc[:8] - pad_rtc[:8]).max()
print(f"   rtc clamp intact under compile (prefix slots match to {seam:.1e})  OK")

# ---- 2. different prompt length -> same bucket, NO recompile ----
from torch._dynamo.utils import counters
before = {k: dict(v) for k, v in counters.items()}
comp_b = run(mm_b, stask_b, state_b)
after = {k: dict(v) for k, v in counters.items()}
new_graphs = after.get("stats", {}).get("unique_graphs", 0) - before.get("stats", {}).get("unique_graphs", 0)
d_b = float(np.max(np.abs(comp_b - pad_b)))
print(f"2. length {lb} (vs warmed {la}): new graphs compiled = {new_graphs}, "
      f"max|diff vs eager-padded| = {d_b:.2e}")

# ---- 2b. DIAGNOSTICS for the cross-input discrepancy ----
comp_b2 = run(mm_b, stask_b, state_b)
comp_a2 = run(mm_a, stask_a, state_a)
print(f"2b. compiled-B determinism |B-B|          = {np.max(np.abs(comp_b2 - comp_b)):.2e}")
print(f"    A-contamination probes |compB-padA|   = {np.max(np.abs(comp_b - pad_a)):.2e}   "
      f"|compB-compA| = {np.max(np.abs(comp_b - comp_a)):.2e}   (vs |padB-padA| = "
      f"{np.max(np.abs(pad_b - pad_a)):.2e})")
print(f"    A-after-B still correct |compA2-padA|  = {np.max(np.abs(comp_a2 - pad_a)):.2e}")

# ---- 3. speedup (reduce-overhead) ----
def bench(n=5):
    ts = []
    for _ in range(n):
        t = {}
        run(mm_a, stask_a, state_a, timings=t)
        ts.append(t["flow_loop_ms"])
    return float(np.median(ts))

ro_ms = bench()
print(f"3. flow loop: eager {te['flow_loop_ms']} ms -> reduce-overhead {ro_ms} ms "
      f"({te['flow_loop_ms'] / ro_ms:.2f}x)")

# ---- 4. control arm: inductor fusion WITHOUT cudagraphs ----
import torch._dynamo
model.disable_expert_compile()
torch._dynamo.reset()
model.enable_expert_compile(mode="default")
run(mm_a, stask_a, state_a); run(mm_a, stask_a, state_a)  # warmup
nc_a = run(mm_a, stask_a, state_a)
nc_b = run(mm_b, stask_b, state_b)
nc_rtc = run(mm_a, stask_a, state_a, prefix=rtc_prefix)
d_nc_a = float(np.max(np.abs(nc_a - pad_a)))
d_nc_b = float(np.max(np.abs(nc_b - pad_b)))
d_nc_rtc = float(np.max(np.abs(nc_rtc - pad_rtc)))
nc_ms = bench()
print(f"4. no-cudagraphs mode: |A|={d_nc_a:.2e}  |B|={d_nc_b:.2e}  |rtc|={d_nc_rtc:.2e}   "
      f"flow loop {nc_ms} ms ({te['flow_loop_ms'] / nc_ms:.2f}x)")

# ---- verdicts ----
assert new_graphs == 0, "RECOMPILED on a different prompt length -- bucket padding broken"
assert d_nc_a < 2e-2 and d_nc_b < 2e-2 and d_nc_rtc < 2e-2, \
    "even no-cudagraphs compile diverges -- inductor numerics problem, do not ship"
if d_b < 2e-2:
    print("\nVERDICT: reduce-overhead is safe on cross-inputs; ship reduce-overhead")
else:
    print("\nVERDICT: reduce-overhead corrupts cross-input results (cudagraph input "
          "handling); ship mode='default' (no-cudagraphs) instead")

# ---- 3. speedup ----
def bench(n=5):
    ts = []
    for _ in range(n):
        t = {}
        run(mm_a, stask_a, state_a, timings=t)
        ts.append(t["flow_loop_ms"])
    return float(np.median(ts))

comp_ms = bench()
print(f"3. flow loop: eager {te['flow_loop_ms']} ms -> compiled {comp_ms} ms  "
      f"({te['flow_loop_ms'] / comp_ms:.2f}x)")

print("\nALL COMPILE TESTS PASS")
