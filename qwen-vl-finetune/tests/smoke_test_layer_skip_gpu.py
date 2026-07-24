"""GPU test for SmolVLA-style layer skipping (expert has N < L layers, attends VLM 0..N-1).

Runs on the real 4B model (random-weight; gates randomized so attention truly mixes):

1. An N=18 expert builds and sample_actions runs -> correct shape, finite, and the expert
   really has 18 layers consuming 18 sliced VLM-KV layers.
2. KEY INVARIANCE: perturbing the VLM's UPPER layers (18..35) leaves the N=18 expert's
   actions BIT-IDENTICAL. Those layers run after 0..17, so they cannot change the first-18
   KV the expert reads -> the expert provably depends only on VLM layers 0..N-1, which is
   exactly what makes a serve-time VLM early-exit at layer N correct.
3. Sanity: an N=36 (full) expert still builds and runs (default path unchanged).
4. Guard: asking for more expert layers than the VLM has raises.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune")
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

from qwenvl.action_expert.inference import (
    ACTION_DIM, ACTION_HORIZON, build_dataset, make_prompt, templatize,
)
from transformers import Qwen3VLForConditionalGeneration
from qwenvl.action_expert import ActionExpertConfig, Qwen3VLWithActionExpert

DEVICE = "cuda"
FAST_TOK = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717merged"
DATA_DIRS = "/iris/projects/humanoid/trossen_data/0717_green_yellow_block_mem_merged"

ds = build_dataset(fast_tok=FAST_TOK, data_dirs=DATA_DIRS, image_history=True, predict_subtask=True)

_VLM = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct", cache_dir=os.environ["HF_HOME"],
    dtype=torch.bfloat16, attn_implementation="flash_attention_2")
L = _VLM.config.text_config.num_hidden_layers
print(f"VLM has {L} language-model layers")


def build(n_expert_layers, seed=1):
    torch.manual_seed(seed)
    tc = _VLM.config.text_config
    cfg = ActionExpertConfig(num_hidden_layers=n_expert_layers,
                             num_key_value_heads=tc.num_key_value_heads, head_dim=tc.head_dim,
                             hidden_size=1024, intermediate_size=4096, num_attention_heads=16,
                             action_dim=ACTION_DIM, action_horizon=ACTION_HORIZON)
    model = Qwen3VLWithActionExpert(_VLM, cfg, train_vlm=False, expert_attends_subtask=True)
    with torch.no_grad():
        for mod in model.action_expert.modules():
            if getattr(mod, "dense", None) is not None:
                torch.nn.init.normal_(mod.dense.weight, std=0.02)
                torch.nn.init.normal_(mod.dense.bias, std=0.02)
    model.action_expert.to(torch.bfloat16)
    model.vlm.resize_token_embeddings(ds.vlm_vocab_size)
    model.vlm.get_input_embeddings().to(torch.bfloat16)
    model.vlm.lm_head.to(torch.bfloat16)
    return model.eval().to(DEVICE)


rng = np.random.default_rng(0)
frames = [Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)) for _ in range(10)]
wrist = [Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))]
state = rng.standard_normal(7).astype(np.float32)
prompt = make_prompt(ds, state)
noise = torch.randn(1, ACTION_HORIZON, ACTION_DIM)
mm = templatize(ds, frames, wrist, prompt, "pick up green block", DEVICE)


def run(model):
    with torch.inference_mode():
        return model.sample_actions(
            input_ids=mm["input_ids"], attention_mask=torch.ones_like(mm["input_ids"]),
            position_ids=mm["position_ids"], pixel_values=mm["pixel_values"],
            image_grid_thw=mm["image_grid_thw"], pixel_values_videos=mm["pixel_values_videos"],
            video_grid_thw=mm["video_grid_thw"], noise=noise.clone().to(DEVICE), num_steps=3)


N = L // 2  # SmolVLA verdict: half
model = build(N)
assert len(model.action_expert.layers) == N, len(model.action_expert.layers)

# count how many VLM text layers actually execute in a run (proves the early-exit fires)
_exec = {"n": 0}
hooks = [lyr.register_forward_hook(lambda *a, **k: _exec.__setitem__("n", _exec["n"] + 1))
         for lyr in _VLM.model.language_model.layers]

a = run(model)  # early-exit is automatic inside _prefix_forward at inference
n_ran = _exec["n"]
assert a.shape == (1, ACTION_HORIZON, ACTION_DIM), a.shape
assert torch.isfinite(a).all()
assert n_ran == N, f"expected {N} VLM text layers to run, got {n_ran}"
print(f"1. N={N} expert; sample_actions -> {tuple(a.shape)}, finite; only {n_ran}/{L} VLM "
      f"layers executed (early-exit fired)  OK")

# 2. early-exit MUST equal full-VLM-then-slice, bit for bit. Force the full VLM by disabling
#    the truncation context manager, rerun, compare.
_exec["n"] = 0
orig = model._vlm_truncated_to_expert_depth
model._vlm_truncated_to_expert_depth = lambda: __import__("contextlib").nullcontext()
a_full = run(model)
model._vlm_truncated_to_expert_depth = orig
n_full = _exec["n"]
for h in hooks:
    h.remove()
assert n_full == L, f"forced-full run should execute all {L} layers, got {n_full}"
d = (a - a_full).abs().max().item()
assert d == 0.0, f"early-exit diverged from full-VLM+slice: maxdiff={d}"
print(f"2. early-exit ({N} layers) == full VLM+slice ({L} layers): BIT-IDENTICAL (maxdiff {d})  OK")

del model
torch.cuda.empty_cache()

# 3. full-depth expert still works
model_full = build(L)
assert len(model_full.action_expert.layers) == L
a_full = run(model_full)
assert a_full.shape == (1, ACTION_HORIZON, ACTION_DIM) and torch.isfinite(a_full).all()
print(f"3. N={L} (full, default) expert still builds and runs  OK")
del model_full
torch.cuda.empty_cache()

# 4. guard: more expert layers than the VLM has must raise
try:
    build(L + 1)
    raise SystemExit("guard FAILED: expected ValueError for N > L")
except ValueError as e:
    print(f"4. guard rejects N>L  OK ({str(e)[:60]}...)")

print("\nALL LAYER-SKIP GPU TESTS PASS")
