"""Integration test for the fast mode-c serve path (implicit HL + SmolVLA layer skip),
mirroring the server's insulated-skip branch WITHOUT needing a trained checkpoint or HTTP.

Reproduces exactly what qwen_action_expert_server.py does on an insulated, layer-skipped
request with --subtask_every 0:
  templatize(generation prompt, no subtask) -> subtask_token_mask_for -> predict_expert(
      prefix_cache=None) -> absolute-joint actions,
and asserts (a) actions have the right shape and are finite, (b) only N VLM layers execute
(the early-exit fired on the real serve path), (c) an RTC action_prefix is honored (clamped).
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune")
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

from qwenvl.action_expert.inference import (
    ACTION_DIM, ACTION_HORIZON, build_dataset, make_prompt, predict_expert,
    subtask_token_mask_for, templatize,
)
from transformers import Qwen3VLForConditionalGeneration
from qwenvl.action_expert import ActionExpertConfig, Qwen3VLWithActionExpert

DEVICE = "cuda"
FAST_TOK = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717merged"
DATA_DIRS = "/iris/projects/humanoid/trossen_data/0717_green_yellow_block_mem_merged"
N = 18

ds = build_dataset(fast_tok=FAST_TOK, data_dirs=DATA_DIRS, image_history=True, predict_subtask=True)

vlm = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct", cache_dir=os.environ["HF_HOME"],
    dtype=torch.bfloat16, attn_implementation="flash_attention_2")
tc = vlm.config.text_config
L = tc.num_hidden_layers
torch.manual_seed(0)
cfg = ActionExpertConfig(num_hidden_layers=N, num_key_value_heads=tc.num_key_value_heads,
                         head_dim=tc.head_dim, hidden_size=1024, intermediate_size=4096,
                         num_attention_heads=16, action_dim=ACTION_DIM, action_horizon=ACTION_HORIZON)
# insulated = the implicit-HL checkpoint (expert ignores the subtask turn); RTC-trained
model = Qwen3VLWithActionExpert(vlm, cfg, train_vlm=True, expert_attends_subtask=False,
                                rtc_prefix_max_length=20)
with torch.no_grad():
    for mod in model.action_expert.modules():
        if getattr(mod, "dense", None) is not None:
            torch.nn.init.normal_(mod.dense.weight, std=0.02)
            torch.nn.init.normal_(mod.dense.bias, std=0.02)
model.action_expert.to(torch.bfloat16)
model.vlm.resize_token_embeddings(ds.vlm_vocab_size)
model.vlm.get_input_embeddings().to(torch.bfloat16)
model.vlm.lm_head.to(torch.bfloat16)
model = model.eval().to(DEVICE)
print(f"insulated layer-skipped model: expert {N} layers, VLM {L} layers")

rng = np.random.default_rng(0)
frames = [Image.fromarray(rng.integers(0, 255, (480, 640, 3), dtype=np.uint8)) for _ in range(10)]
wrist = [Image.fromarray(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))]
state = rng.standard_normal(7).astype(np.float32)

# --- exactly the server's insulated skip branch (subtask_every == 0) ---
_exec = {"n": 0}
hooks = [lyr.register_forward_hook(lambda *a, **k: _exec.__setitem__("n", _exec["n"] + 1))
         for lyr in vlm.model.language_model.layers]

prompt = make_prompt(ds, state)                                  # fixed question, no subtask
mm = templatize(ds, frames, wrist, prompt, None, DEVICE)         # generation prompt
stask_mask = subtask_token_mask_for(mm["input_ids"])             # insulated: mask the header
with torch.inference_mode():
    actions = predict_expert(model, mm, ds, state, None, 10, prefix_cache=None,
                             subtask_token_mask=stask_mask)
n_ran = _exec["n"]
assert actions.shape == (ACTION_HORIZON, ACTION_DIM), actions.shape
assert np.isfinite(actions).all()
assert n_ran == N, f"expected {N} VLM layers on the serve path, got {n_ran}"
print(f"1. serve path -> actions {actions.shape}, finite; {n_ran}/{L} VLM layers ran "
      f"(early-exit fired on the real /infer path)  OK")

# --- RTC: an absolute action_prefix must be clamped into the first d slots ---
d = 6
prefix_abs = np.repeat(state[None], d, axis=0)  # hold current pose for d ticks (absolute joints)
_exec["n"] = 0
with torch.inference_mode():
    actions_rtc = predict_expert(model, mm, ds, state, None, 10,
                                 action_prefix_abs=prefix_abs, prefix_cache=None,
                                 subtask_token_mask=stask_mask)
for h in hooks:
    h.remove()
# first d actions should equal the (absolute) prefix we clamped in
seam = np.abs(actions_rtc[:d] - prefix_abs).max()
assert seam < 1e-2, f"RTC prefix not honored: max|actions[:d]-prefix|={seam}"
assert _exec["n"] == N, f"early-exit must also fire with an RTC prefix, got {_exec['n']}"
print(f"2. RTC action_prefix honored (max|clamp|={seam:.1e}) and early-exit still fired  OK")

print("\nALL LAYER-SKIP SERVE-PATH TESTS PASS")
