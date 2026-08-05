"""HARSH attention-wiring proof for mixed subtask supervision (GPU, real 4B, real data).

Every probe is a perturbation argument on REAL dataset items from the 0717(labeled) +
0731(unlabeled) mix: corrupt token ids at positions the expert must NOT see -> the expert
output tensor must be BIT-IDENTICAL (0.0); corrupt one token it MUST see -> it must
change. Same-length sequences throughout, so bf16 kernel tiling cannot blur the result.

Why id-scrambling is a valid probe: the masked spans (assistant turn / generation header)
are the LAST tokens before the FAST region, and FAST is also masked -- so a change there
can only reach the expert THROUGH the masked positions' KV. Bit-equality proves the mask.

1. UNLABELED (0731) training item: scramble the 3-token generation header -> v_t identical.
2. UNLABELED: scramble the entire FAST region -> v_t identical.
3. UNLABELED control: scramble ONE attended state-digit token -> v_t CHANGES (detector works).
4. UNLABELED counterfactual: flip expert_attends_subtask=True on the SAME weights ->
   header scramble now CHANGES v_t (the mask, not coincidence, was doing the work).
5. LABELED (0717) training item: scramble the whole assistant turn (header+answer+end)
   -> v_t identical; control token -> changes.
6. MIXED BATCH: collate [labeled, unlabeled] (different lengths, real collator); per-sample
   masks land on the same token ids as solo; batched v_t matches solo runs (bf16 tiling tol).
7. SERVE path: insulated sample_actions on the generation-header form; scramble header ->
   actions identical.

Fresh-expert gotcha handled: adaRMS gates are zero-init (branches gated to 0, probes would
trivially pass), so .dense weights are randomized -- attention genuinely mixes.

Run (GPU): python tests/smoke_test_mixed_attention_gpu.py
"""
import os
import random
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune")
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

from qwenvl.action_expert.inference import (
    ACTION_DIM, ACTION_HORIZON, build_dataset, make_prompt, subtask_token_mask_for, templatize,
)
from qwenvl.data.robot_data import RobotActionDataCollator

DEVICE = "cuda"
DATA = "/iris/projects/humanoid/trossen_data"
NEW_TOK = "/iris/projects/humanoid/ke/Qwen3-VL/checkpoints/fast_tokenizer_trossen_0717merged_0731gy"
MIX = f"{DATA}/0717_green_yellow_block_mem_merged,{DATA}/0731_green_yellow_merged"

torch.manual_seed(0)
ds = build_dataset(fast_tok=NEW_TOK, data_dirs=MIX)

# ---- real 4B VLM + fresh expert with randomized adaRMS gates (see docstring) ----
from transformers import Qwen3VLForConditionalGeneration
from qwenvl.action_expert import ActionExpertConfig, Qwen3VLWithActionExpert

vlm = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-4B-Instruct", cache_dir=os.environ["HF_HOME"],
    dtype=torch.bfloat16, attn_implementation="flash_attention_2")
tc = vlm.config.text_config
cfg = ActionExpertConfig(num_hidden_layers=18, num_key_value_heads=tc.num_key_value_heads,
                         head_dim=tc.head_dim, hidden_size=1024, intermediate_size=4096,
                         num_attention_heads=16, action_dim=ACTION_DIM, action_horizon=ACTION_HORIZON)
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

x_t = torch.randn(1, ACTION_HORIZON, ACTION_DIM, device=DEVICE)
tstep = torch.full((1,), 0.7, device=DEVICE)

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


def expert_v_t(item, ids_override=None, batch_x_t=None, batch_tstep=None):
    """Raw expert output on a (possibly id-scrambled) prefix. Bit-exact comparable."""
    kw = {k: item[k] for k in MODEL_KEYS if item.get(k) is not None}
    # Raw items carry attention_mask as a list (the collator normally rebuilds it);
    # solo items are unpadded so all-ones is exact. Batched items keep their real mask.
    if not torch.is_tensor(kw.get("attention_mask")):
        kw["attention_mask"] = torch.ones_like(kw["input_ids"])
    if ids_override is not None:
        kw["input_ids"] = ids_override
    with torch.inference_mode():
        kv, _, _ = model._prefix_forward(
            kw["input_ids"], kw["attention_mask"], kw["position_ids"],
            kw.get("pixel_values"), kw.get("image_grid_thw"),
            kw.get("pixel_values_videos"), kw.get("video_grid_thw"),
            labels=None, fast_token_mask=item["fast_token_mask"])
        return model._expert_forward(
            batch_x_t if batch_x_t is not None else x_t,
            batch_tstep if batch_tstep is not None else tstep,
            kv, kw["attention_mask"], kw["position_ids"],
            item["fast_token_mask"], item["subtask_token_mask"])


def scrambled(item, pos_mask, seed=7):
    """input_ids with positions in pos_mask replaced by arbitrary in-vocab text tokens."""
    g = torch.Generator().manual_seed(seed)
    junk = torch.randint(1000, 30000, item["input_ids"].shape, generator=g).to(item["input_ids"])
    return torch.where(pos_mask, junk, item["input_ids"])


def state_digit_pos(item):
    """Position of one attended token: the LAST supervisable text token of the user turn
    (a state digit right before <|im_end|>), guaranteed outside all masks."""
    ids = item["input_ids"][0]
    sub, fast = item["subtask_token_mask"][0], item["fast_token_mask"][0]
    im_ends = (ids == 151645).nonzero().ravel()
    user_end = int(im_ends[0])                       # first <|im_end|> closes the user turn
    p = user_end - 1
    assert not sub[p] and not fast[p], "picked a masked token as control?!"
    return p


# ================= UNLABELED (0731) =================
U = to_dev(fixed_t_item(224 + 5, 120))
assert not (U["labels"][0] != -100)[~U["fast_token_mask"][0]].any(), "unlabeled item has language labels?!"
base_u = expert_v_t(U)

hdr = U["subtask_token_mask"]
assert int(hdr.sum()) == 3, f"unlabeled mask should be the 3-token header, got {int(hdr.sum())}"
d = (expert_v_t(U, ids_override=scrambled(U, hdr)) - base_u).abs().max().item()
assert d == 0.0, f"expert SAW the scrambled generation header! maxdiff={d}"
print(f"1. unlabeled: header scrambled -> v_t bit-identical (maxdiff {d})  OK")

d = (expert_v_t(U, ids_override=scrambled(U, U["fast_token_mask"])) - base_u).abs().max().item()
assert d == 0.0, f"expert SAW scrambled FAST tokens! maxdiff={d}"
print(f"2. unlabeled: FAST region scrambled -> v_t bit-identical (maxdiff {d})  OK")

ctrl = torch.zeros_like(hdr)
ctrl[0, state_digit_pos(U)] = True
d = (expert_v_t(U, ids_override=scrambled(U, ctrl)) - base_u).abs().max().item()
assert d > 1e-4, f"control failed: expert ignored an attended state token (maxdiff={d})"
print(f"3. unlabeled control: one state digit scrambled -> v_t CHANGES (maxdiff {d:.2e})  OK")

model.expert_attends_subtask = True
base_att = expert_v_t(U)
d = (expert_v_t(U, ids_override=scrambled(U, hdr)) - base_att).abs().max().item()
model.expert_attends_subtask = False
assert d > 1e-4, f"counterfactual failed: non-insulated expert also ignored the header ({d})"
print(f"4. counterfactual (insulation OFF, same weights): header scramble CHANGES v_t ({d:.2e})  OK")

# ================= LABELED (0717) =================
L = to_dev(fixed_t_item(5, 120))
n_turn = int(L["subtask_token_mask"].sum())
assert n_turn >= 8, f"labeled mask should span the whole assistant turn, got {n_turn}"
base_l = expert_v_t(L)
d = (expert_v_t(L, ids_override=scrambled(L, L["subtask_token_mask"])) - base_l).abs().max().item()
assert d == 0.0, f"expert SAW the scrambled subtask turn! maxdiff={d}"
ctrl = torch.zeros_like(L["subtask_token_mask"])
ctrl[0, state_digit_pos(L)] = True
d2 = (expert_v_t(L, ids_override=scrambled(L, ctrl)) - base_l).abs().max().item()
assert d2 > 1e-4, "labeled control failed"
print(f"5. labeled: {n_turn}-token assistant turn scrambled -> bit-identical (0.0); control changes ({d2:.2e})  OK")

# ================= MIXED BATCH =================
collator = RobotActionDataCollator(ds.tokenizer)
cpu_L, cpu_U = fixed_t_item(5, 120), fixed_t_item(224 + 5, 120)
batch = to_dev(collator([cpu_L, cpu_U]))
seq = batch["input_ids"].shape[1]
for i, solo in enumerate((cpu_L, cpu_U)):
    n = solo["input_ids"].shape[1]
    for key in ("input_ids",):
        assert torch.equal(batch[key][i, :n].cpu(), solo[key][0]), f"collator moved tokens (sample {i})"
    for key in ("subtask_token_mask", "fast_token_mask"):
        assert torch.equal(batch[key][i, :n].cpu(), solo[key][0]), f"collator misaligned {key} (sample {i})"
        assert not batch[key][i, n:].any(), f"{key} leaked into padding (sample {i})"
    assert not batch["attention_mask"][i, n:].any(), f"padding not attention-masked (sample {i})"
bx = torch.cat([x_t, x_t]); bt = torch.cat([tstep, tstep])
vb = expert_v_t(batch, batch_x_t=bx, batch_tstep=bt)
d_l = (vb[0:1] - base_l).abs().max().item()
d_u = (vb[1:2] - base_u).abs().max().item()
assert d_l < 2e-2 and d_u < 2e-2, f"batched != solo beyond bf16 tiling noise: L={d_l:.2e} U={d_u:.2e}"
print(f"6. mixed batch [labeled+unlabeled]: masks land on identical tokens; padding masked; "
      f"batched v_t ~= solo (L {d_l:.1e}, U {d_u:.1e}, bf16 tiling)  OK")

# ================= SERVE PATH =================
rng = np.random.default_rng(0)
frames = [Image.fromarray(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8)) for _ in range(10)]
wrist = [Image.fromarray(rng.integers(0, 255, (540, 960, 3), dtype=np.uint8))]
mm = templatize(ds, frames, wrist, make_prompt(ds, np.zeros(ACTION_DIM, dtype=np.float32)), None, DEVICE)
smask = subtask_token_mask_for(mm["input_ids"])
assert int(smask.sum()) == 3, "serve mask should cover exactly the 3-token header"
noise = torch.randn(1, ACTION_HORIZON, ACTION_DIM, device=DEVICE)


def serve(ids):
    with torch.inference_mode():
        return model.sample_actions(
            input_ids=ids, attention_mask=torch.ones_like(ids), position_ids=mm["position_ids"],
            pixel_values=mm["pixel_values"], image_grid_thw=mm["image_grid_thw"],
            pixel_values_videos=mm["pixel_values_videos"], video_grid_thw=mm["video_grid_thw"],
            noise=noise.clone(), num_steps=3, subtask_token_mask=smask)


a = serve(mm["input_ids"])
g = torch.Generator().manual_seed(7)
junk = torch.randint(1000, 30000, mm["input_ids"].shape, generator=g).to(mm["input_ids"])
b = serve(torch.where(smask, junk, mm["input_ids"]))
d = (a - b).abs().max().item()
assert d == 0.0, f"SERVING expert saw the scrambled header! maxdiff={d}"
print(f"7. serve path: header scrambled -> actions bit-identical (maxdiff {d})  OK")

print("\nALL MIXED-ATTENTION WIRING PROOFS PASS")
