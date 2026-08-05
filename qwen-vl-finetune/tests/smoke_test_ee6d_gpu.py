"""GPU check for the ee6d pipeline (batch-1, fits in ~12GB beside a training job).

1. Training forward on a REAL labeled ee6d item (action_dim 10, 2 wrists, insulated,
   RTC off, fixed noise/time): loss finite.
2. Same on an unlabeled 0731 item.
3. Insulation invariance in EE space: scramble the generation-header tokens of the
   unlabeled item -> expert v_t (1, 50, 10) bit-identical; scramble one attended state
   token -> changes.
"""
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune")
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from qwenvl.action_expert.inference import ACTION_HORIZON, build_dataset

DEVICE = "cuda"
DATA = "/iris/projects/humanoid/trossen_data"
MIX = f"{DATA}/0717_green_yellow_block_mem_merged,{DATA}/0731_green_yellow_merged"
TOK = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717m_0731gy_ee6d"
REPR = {"action_space": "ee6d", "action_dim": 10, "action_horizon": 50,
        "delta_mask": "9,-1", "active_dims": "7:14"}

torch.manual_seed(0)
ds = build_dataset(fast_tok=TOK, data_dirs=MIX, repr_cfg=dict(REPR),
                   budget={"wrist_cameras": "cam_right_wrist,cam_left_wrist",
                           "min_episode_len": 50})

from transformers import Qwen3VLForConditionalGeneration
from qwenvl.action_expert import ActionExpertConfig, Qwen3VLWithActionExpert

vlm = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct", cache_dir=os.environ["HF_HOME"],
    dtype=torch.bfloat16, attn_implementation="sdpa")
tc = vlm.config.text_config
cfg = ActionExpertConfig(num_hidden_layers=18, num_key_value_heads=tc.num_key_value_heads,
                         head_dim=tc.head_dim, hidden_size=1024, intermediate_size=4096,
                         num_attention_heads=16, action_dim=10, action_horizon=ACTION_HORIZON)
model = Qwen3VLWithActionExpert(vlm, cfg, train_vlm=False, expert_attends_subtask=False)
torch.manual_seed(1)
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

MODEL_KEYS = ("input_ids", "attention_mask", "position_ids", "pixel_values",
              "image_grid_thw", "pixel_values_videos", "video_grid_thw")


def fixed_t_item(idx, t):
    orig = random.randint
    random.randint = lambda lo, hi: max(lo, min(t, hi))
    try:
        return ds[idx]
    finally:
        random.randint = orig


def to_dev(item):
    return {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in item.items()}


noise = torch.randn(1, ACTION_HORIZON, 10, device=DEVICE)
tstep = torch.full((1,), 0.7, device=DEVICE)

for name, idx in (("labeled/0717", 5), ("unlabeled/0731", 300)):
    it = to_dev(fixed_t_item(idx, 120))
    kw = {k: it[k] for k in MODEL_KEYS if it.get(k) is not None}
    if not torch.is_tensor(kw.get("attention_mask")):
        kw["attention_mask"] = torch.ones_like(kw["input_ids"])
    with torch.inference_mode():
        out = model(actions=it["actions"].unsqueeze(0) if it["actions"].ndim == 2 else it["actions"],
                    labels=None, fast_token_mask=it["fast_token_mask"],
                    subtask_token_mask=it["subtask_token_mask"], noise=noise.clone(),
                    time=tstep.clone(), **kw)
    assert torch.isfinite(out.loss), f"{name}: non-finite loss"
    print(f"1/2. {name}: training forward loss {out.loss.item():.4f} (finite, action_dim 10)  OK")

# ---- 3. insulation invariance on the unlabeled ee6d item ----
U = to_dev(fixed_t_item(300, 120))
kwU = {k: U[k] for k in MODEL_KEYS if U.get(k) is not None}
if not torch.is_tensor(kwU.get("attention_mask")):
    kwU["attention_mask"] = torch.ones_like(kwU["input_ids"])


def v_t(ids):
    with torch.inference_mode():
        kv, _, _ = model._prefix_forward(ids, kwU["attention_mask"], kwU["position_ids"],
                                         kwU.get("pixel_values"), kwU.get("image_grid_thw"),
                                         kwU.get("pixel_values_videos"), kwU.get("video_grid_thw"),
                                         labels=None, fast_token_mask=U["fast_token_mask"])
        return model._expert_forward(noise, tstep, kv, kwU["attention_mask"], kwU["position_ids"],
                                     U["fast_token_mask"], U["subtask_token_mask"])


base = v_t(kwU["input_ids"])
assert base.shape[-1] == 10, f"expert output dim {base.shape}"
g = torch.Generator().manual_seed(7)
junk = torch.randint(1000, 30000, kwU["input_ids"].shape, generator=g).to(kwU["input_ids"])
hdr = U["subtask_token_mask"]
d_hdr = (v_t(torch.where(hdr, junk, kwU["input_ids"])) - base).abs().max().item()
assert d_hdr == 0.0, f"ee6d expert SAW the scrambled header ({d_hdr})"
ids = kwU["input_ids"][0]
im_end = (ids == 151645).nonzero().ravel()[0]
ctrl = torch.zeros_like(hdr)
ctrl[0, im_end - 1] = True
d_ctrl = (v_t(torch.where(ctrl, junk, kwU["input_ids"])) - base).abs().max().item()
assert d_ctrl > 1e-4, f"control failed ({d_ctrl})"
print(f"3. insulation in EE space: header scramble -> 0.0; state-token scramble -> {d_ctrl:.1e}  OK")

# ---- 4. FAST region scramble on the unlabeled item -> invariant ----
d_fast = (v_t(torch.where(U["fast_token_mask"], junk, kwU["input_ids"])) - base).abs().max().item()
assert d_fast == 0.0, f"ee6d expert SAW scrambled FAST tokens ({d_fast})"
print(f"4. FAST scramble -> v_t bit-identical (0.0)  OK")

# ---- 5. labeled item: full assistant turn scrambled -> invariant; counterfactual ----
L = to_dev(fixed_t_item(5, 120))
kwL = {k: L[k] for k in MODEL_KEYS if L.get(k) is not None}
if not torch.is_tensor(kwL.get("attention_mask")):
    kwL["attention_mask"] = torch.ones_like(kwL["input_ids"])


def v_t_L(ids):
    with torch.inference_mode():
        kv, _, _ = model._prefix_forward(ids, kwL["attention_mask"], kwL["position_ids"],
                                         kwL.get("pixel_values"), kwL.get("image_grid_thw"),
                                         kwL.get("pixel_values_videos"), kwL.get("video_grid_thw"),
                                         labels=None, fast_token_mask=L["fast_token_mask"])
        return model._expert_forward(noise, tstep, kv, kwL["attention_mask"], kwL["position_ids"],
                                     L["fast_token_mask"], L["subtask_token_mask"])


junkL = torch.randint(1000, 30000, kwL["input_ids"].shape,
                      generator=torch.Generator().manual_seed(7)).to(kwL["input_ids"])
baseL = v_t_L(kwL["input_ids"])
d_turn = (v_t_L(torch.where(L["subtask_token_mask"], junkL, kwL["input_ids"])) - baseL).abs().max().item()
assert d_turn == 0.0, f"expert SAW the scrambled subtask turn ({d_turn})"
model.expert_attends_subtask = True
b_att = v_t_L(kwL["input_ids"])
d_cf = (v_t_L(torch.where(L["subtask_token_mask"], junkL, kwL["input_ids"])) - b_att).abs().max().item()
model.expert_attends_subtask = False
assert d_cf > 1e-4, f"counterfactual failed ({d_cf})"
print(f"5. labeled: turn scramble -> 0.0; insulation-off counterfactual -> {d_cf:.1e}  OK")

# ---- 6. mixed batch through the REAL collator (10-dim masks stay aligned) ----
from qwenvl.data.robot_data import RobotActionDataCollator
coll = RobotActionDataCollator(ds.tokenizer)
cpuL, cpuU = fixed_t_item(5, 120), fixed_t_item(300, 120)
batch = to_dev(coll([cpuL, cpuU]))
for i, solo in enumerate((cpuL, cpuU)):
    n = solo["input_ids"].shape[1]
    for key in ("input_ids", "subtask_token_mask", "fast_token_mask"):
        assert torch.equal(batch[key][i, :n].cpu(), solo[key][0]), f"collator misaligned {key} (sample {i})"
        if key != "input_ids":
            assert not batch[key][i, n:].any(), f"{key} leaked into padding"
    assert not batch["attention_mask"][i, n:].any(), "padding not attention-masked"
assert batch["actions"].shape[-1] == 10, f"collated actions dim {batch['actions'].shape}"
with torch.inference_mode():
    outb = model(actions=batch["actions"], labels=None,
                 fast_token_mask=batch["fast_token_mask"],
                 subtask_token_mask=batch["subtask_token_mask"],
                 noise=torch.cat([noise, noise]), time=torch.cat([tstep, tstep]),
                 **{k: batch[k] for k in MODEL_KEYS if batch.get(k) is not None})
assert torch.isfinite(outb.loss), "mixed ee6d batch loss non-finite"
print(f"6. mixed batch [labeled+unlabeled]: masks aligned at 10 dims, loss {outb.loss.item():.3f}  OK")

# ---- 7. full serve chain: templatize (2 wrists) -> predict_expert -> IK to joints ----
from PIL import Image as PILImage
from qwenvl.action_expert.inference import (
    make_prompt, model_actions_to_robot, predict_expert, state_to_model, subtask_token_mask_for,
    templatize,
)
rng = np.random.default_rng(0)
frames = [PILImage.fromarray(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)) for _ in range(10)]
wr = [PILImage.fromarray(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)) for _ in range(2)]
q_now = np.array([0.1, 0.9, 0.6, 0.2, -0.3, 0.1, 0.02], dtype=np.float32)
ms = state_to_model(ds, q_now)
mm = templatize(ds, frames, wr, make_prompt(ds, ms), None, DEVICE)
smask = subtask_token_mask_for(mm["input_ids"])
serve_noise = torch.randn(1, ACTION_HORIZON, 10, device=DEVICE,
                          generator=torch.Generator(device=DEVICE).manual_seed(0))
with torch.inference_mode():
    acts = predict_expert(model, mm, ds, ms, serve_noise.clone(), 3, subtask_token_mask=smask)
assert acts.shape == (ACTION_HORIZON, 10), f"serve chunk shape {acts.shape}"
ids2 = torch.where(smask, torch.randint(1000, 30000, mm["input_ids"].shape,
                                        generator=torch.Generator().manual_seed(7)).to(mm["input_ids"]),
                   mm["input_ids"])
mm2 = dict(mm); mm2["input_ids"] = ids2
with torch.inference_mode():
    acts2 = predict_expert(model, mm2, ds, ms, serve_noise.clone(), 3,
                           subtask_token_mask=subtask_token_mask_for(mm["input_ids"]))
d_serve = np.abs(acts - acts2).max()
assert d_serve == 0.0, f"serve-path insulation leak ({d_serve})"
joints, n_ikfail = model_actions_to_robot(ds, acts, q_now)
assert joints.shape == (ACTION_HORIZON, 7), f"robot actions shape {joints.shape}"
print(f"7. serve chain: (50,10) EE chunk, header-scramble invariant (0.0), IK -> (50,7) "
      f"joints ({n_ikfail} unconverged on RANDOM-weight outputs -- held, no crash)  OK")

print("\nALL EE6D GPU CHECKS PASS")
