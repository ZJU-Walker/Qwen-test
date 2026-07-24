"""GPU end-to-end RTC test: real Qwen3-VL-4B + (random-init) action expert.

Checks the full model mechanics (weights don't matter for these):
  1. per-token timestep (B, ah) with all-equal entries == per-sample (B,) forward  (v_t match)
  2. training forward: RTC OFF unchanged; RTC ON -> finite loss, and a hand-check that the
     postfix-only reweighting matches a manual recomputation with forced prefix lengths
  3. sample_actions: no-prefix path unchanged shape/finite; with action_prefix -> the first d
     outputs EQUAL the prefix (hard clamp) and the postfix differs from the no-prefix run;
     prefix_length=0 degenerates to (numerically) the no-prefix result
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune")
os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")

from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

from qwenvl.action_expert import ActionExpertConfig, Qwen3VLWithActionExpert

DEVICE = "cuda"
MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
AH, AD = 50, 7
torch.manual_seed(0)

print("Loading Qwen3-VL-4B (bf16)...")
vlm = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_PATH, cache_dir=os.environ["HF_HOME"], dtype=torch.bfloat16, attn_implementation="flash_attention_2",
)
tc = vlm.config.text_config
cfg = ActionExpertConfig(
    num_hidden_layers=tc.num_hidden_layers, num_key_value_heads=tc.num_key_value_heads,
    head_dim=tc.head_dim, hidden_size=1024, intermediate_size=4096, num_attention_heads=16,
    action_dim=AD, action_horizon=AH,
)
model = Qwen3VLWithActionExpert(vlm, cfg, train_vlm=False)
# The adaRMS conditioning projections are ZERO-init (gates = 0 -> attention/MLP branches
# contribute nothing at init, so a fresh expert is a position-wise map that trivially ignores
# the prefix and makes the per-token-vs-per-sample check degenerate). Randomize them so the
# expert behaves like a trained one mechanically: nonzero gates -> real attention mixing.
with torch.no_grad():
    for m in model.action_expert.modules():
        if getattr(m, "dense", None) is not None:
            torch.nn.init.normal_(m.dense.weight, std=0.02)
            torch.nn.init.normal_(m.dense.bias, std=0.02)
model.action_expert.to(torch.bfloat16)
model = model.eval().to(DEVICE)

# Text-only prefix (the expert only needs a VLM KV cache; no images required for mechanics).
tok = AutoTokenizer.from_pretrained(MODEL_PATH, cache_dir=os.environ["HF_HOME"])
B = 2
ids = tok(["Task: pick up yellow block, State: 1 2 3"] * B, return_tensors="pt").input_ids.to(DEVICE)
mask = torch.ones_like(ids)
pos = torch.arange(ids.shape[1], device=DEVICE)[None].expand(B, -1)

with torch.no_grad():
    pkv, _, _ = model._prefix_forward(ids, mask, pos, None, None, None, None)

    # ---- 1. per-sample vs per-token timestep equivalence through the REAL expert ----
    x_t = torch.randn(B, AH, AD, device=DEVICE)
    t = torch.rand(B, device=DEVICE)
    v_sample = model._expert_forward(x_t, t, pkv, mask, pos, None)
    v_token = model._expert_forward(x_t, t[:, None].expand(B, AH), pkv, mask, pos, None)
    diff = (v_sample - v_token).abs().max().item()
    assert diff < 1e-3, f"per-token != per-sample: {diff}"
    print(f"1. per-token (B,ah) == per-sample (B,) expert forward  OK (max diff {diff:.1e})")

    # ---- 2. training forward ----
    actions = torch.randn(B, AH, AD, device=DEVICE)
    noise = torch.randn(B, AH, AD, device=DEVICE)
    time_fixed = torch.full((B,), 0.7, device=DEVICE)

    out_off = model(input_ids=ids, actions=actions, attention_mask=mask, position_ids=pos,
                    noise=noise, time=time_fixed)
    # RTC off: flow_loss must equal the plain MSE recomputed by hand
    x_manual = 0.7 * noise + 0.3 * actions
    v_manual = model._expert_forward(x_manual, time_fixed, pkv, mask, pos, None)
    manual = torch.nn.functional.mse_loss(noise - actions, v_manual)
    d_off = abs(out_off.flow_loss.item() - manual.item())
    assert d_off < 1e-4, f"RTC-off loss changed: {d_off}"
    print(f"2a. RTC OFF: forward flow_loss matches manual MSE  OK (diff {d_off:.1e})")

    # RTC on: force a KNOWN prefix length by pinning min == max == 10, then hand-recompute.
    model.rtc_prefix_min_length = 10
    model.rtc_prefix_max_length = 10
    out_on = model(input_ids=ids, actions=actions, attention_mask=mask, position_ids=pos,
                   noise=noise, time=time_fixed)
    assert torch.isfinite(out_on.loss), "RTC-on loss not finite"
    d = 10
    x_rtc = x_manual.clone(); x_rtc[:, :d] = actions[:, :d]
    t_tok = time_fixed[:, None].expand(B, AH).clone(); t_tok[:, :d] = 0.0
    v_rtc = model._expert_forward(x_rtc, t_tok, pkv, mask, pos, None)
    per_tok = (((noise - actions) - v_rtc) ** 2).mean(-1)
    # the reweighted (zero-prefix, ah/postfix_count-scaled) mean == plain postfix mean per sample
    manual_on = per_tok[:, d:].mean(-1).mean()
    d_on = abs(out_on.flow_loss.item() - manual_on.item())
    assert d_on < 1e-4, f"RTC-on loss mismatch vs manual postfix mean: {d_on}"
    print(f"2b. RTC ON (d=10 forced): loss == manual postfix-only mean  OK (diff {d_on:.1e})")
    model.rtc_prefix_min_length = 0
    model.rtc_prefix_max_length = 0

    # ---- 3. sample_actions ----
    noise_s = torch.randn(B, AH, AD, device=DEVICE)
    base = model.sample_actions(input_ids=ids, attention_mask=mask, position_ids=pos,
                                noise=noise_s.clone(), num_steps=4)
    assert base.shape == (B, AH, AD) and torch.isfinite(base).all()

    d = 25
    prefix = torch.randn(B, d, AD, device=DEVICE)
    rtc = model.sample_actions(input_ids=ids, attention_mask=mask, position_ids=pos,
                               noise=noise_s.clone(), num_steps=4, action_prefix=prefix)
    assert torch.equal(rtc[:, :d], prefix), "prefix not clamped exactly"
    assert torch.isfinite(rtc).all()
    assert not torch.allclose(rtc[:, d:], base[:, d:], atol=1e-3), "postfix ignored the prefix?"
    print(f"3a. sample_actions clamps the d={d} prefix EXACTLY; postfix responds to it  OK")

    deg = model.sample_actions(input_ids=ids, attention_mask=mask, position_ids=pos,
                               noise=noise_s.clone(), num_steps=4,
                               action_prefix=prefix, prefix_length=0)
    dd = (deg - base).abs().max().item()
    assert dd < 1e-2, f"prefix_length=0 != no-prefix: {dd}"
    print(f"3b. prefix_length=0 degenerates to the no-prefix result  OK (max diff {dd:.1e})")

print("\nALL GPU RTC TESTS PASS")
