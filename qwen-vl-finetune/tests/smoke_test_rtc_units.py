"""CPU unit tests for the RTC math (no VLM, no GPU)."""
import sys

import numpy as np
import torch

sys.path.insert(0, "/iris/projects/humanoid/ke/Qwen3-VL/qwen-vl-finetune")

from qwenvl.action_expert.modeling_action_expert import AdaRMSNorm
from qwenvl.action_expert.modeling_qwen3vl_with_expert import (
    _CLEAN_TIMESTEP,
    _clamp_action_prefix,
    _rtc_prefix_mask,
    create_sinusoidal_pos_embedding,
)
from qwenvl.data.robot_data import apply_delta_actions, make_delta_mask, quantile_normalize, quantile_unnormalize, undo_delta_actions

torch.manual_seed(0)
np.random.seed(0)
B, AH, AD, DIM = 3, 50, 7, 64

# ---- 1. sinusoidal embedding: (B,) backward-compat + (B, ah) consistency ----
t1 = torch.rand(B)
emb1 = create_sinusoidal_pos_embedding(t1, DIM, 4e-3, 4.0)
assert emb1.shape == (B, DIM)
# old formula, reimplemented literally
fraction = torch.linspace(0.0, 1.0, DIM // 2)
period = 4e-3 * (4.0 / 4e-3) ** fraction
scaling = 1.0 / period * 2 * torch.pi
old = torch.cat([torch.sin(scaling[None] * t1[:, None]), torch.cos(scaling[None] * t1[:, None])], dim=1)
assert torch.allclose(emb1, old, atol=1e-6), "regression: (B,) embedding changed"
# (B, ah) with all tokens sharing the sample's t must equal the (B,) embedding per row
t2 = t1[:, None].expand(B, AH)
emb2 = create_sinusoidal_pos_embedding(t2, DIM, 4e-3, 4.0)
assert emb2.shape == (B, AH, DIM)
assert torch.allclose(emb2, emb1[:, None, :].expand(B, AH, DIM), atol=1e-6)
print("1. sinusoidal embedding: (B,) unchanged, (B,ah) consistent  OK")

# ---- 2. AdaRMSNorm: per-sample vs per-token cond equivalence ----
norm = AdaRMSNorm(DIM, cond_dim=DIM)
torch.nn.init.normal_(norm.dense.weight, std=0.02)  # zero-init would make everything trivially equal
torch.nn.init.normal_(norm.dense.bias, std=0.02)
x = torch.randn(B, AH, DIM)
cond_sample = torch.randn(B, DIM)
out_a, gate_a = norm(x, cond_sample)                      # gate_a: (B, 1, DIM), broadcasts over seq
out_b, gate_b = norm(x, cond_sample[:, None, :].expand(B, AH, DIM))  # gate_b: (B, AH, DIM)
gate_a_full = gate_a.expand(B, AH, DIM)
assert torch.allclose(out_a, out_b, atol=1e-6) and torch.allclose(gate_a_full, gate_b, atol=1e-6)
# per-token cond that differs ONLY at token 0 must change only token 0
cond_tok = cond_sample[:, None, :].expand(B, AH, DIM).clone()
cond_tok[:, 0] += 1.0
out_c, gate_c = norm(x, cond_tok)
assert not torch.allclose(out_c[:, 0], out_a[:, 0], atol=1e-6)
assert torch.allclose(out_c[:, 1:], out_a[:, 1:], atol=1e-6) and torch.allclose(gate_c[:, 1:], gate_a_full[:, 1:], atol=1e-6)
print("2. AdaRMSNorm: per-sample == expanded per-token; per-token is truly per-token  OK")

# ---- 3. prefix mask + clamp semantics ----
lengths = torch.tensor([0, 3, AH - 1])
mask = _rtc_prefix_mask(lengths, AH)
assert mask.shape == (3, AH)
assert mask.sum(-1).tolist() == [0, 3, AH - 1]
assert mask[1, :3].all() and not mask[1, 3:].any()
x_t = torch.randn(3, AH, AD)
prefix = torch.randn(3, AH, AD)
clamped = _clamp_action_prefix(x_t, prefix, mask)
assert torch.equal(clamped[0], x_t[0])                       # d=0: untouched
assert torch.equal(clamped[1, :3], prefix[1, :3]) and torch.equal(clamped[1, 3:], x_t[1, 3:])
print(f"3. prefix mask/clamp semantics (CLEAN_TIMESTEP={_CLEAN_TIMESTEP})  OK")

# ---- 4. postfix loss reweight: mean equals per-sample postfix mean, averaged ----
u = torch.randn(3, AH, AD)
v = torch.randn(3, AH, AD)
per_token = ((u - v) ** 2).mean(-1)
postfix_mask = ~mask
postfix_count = postfix_mask.sum(-1, keepdim=True).clamp(min=1)
rw = torch.where(postfix_mask, per_token * (AH / postfix_count), torch.zeros_like(per_token))
expected = torch.stack([per_token[i][postfix_mask[i]].mean() for i in range(3)]).mean()
assert torch.allclose(rw.mean(), expected, atol=1e-6)
# d=0 sample contributes exactly its vanilla mean
assert torch.allclose(rw[0].mean(), per_token[0].mean(), atol=1e-6)
print("4. postfix loss reweight: scale-preserving, equal per-sample weighting  OK")

# ---- 5. absolute-prefix round trip through delta + quantile normalization ----
delta_mask = make_delta_mask("6,-1")
stats = {"q01": (np.random.randn(AD) - 3).tolist(), "q99": (np.random.randn(AD) + 3).tolist()}
state = np.random.randn(AD).astype(np.float32)
prefix_abs = np.random.randn(10, AD).astype(np.float32)
norm_p = quantile_normalize(apply_delta_actions(prefix_abs, state, delta_mask), stats)
back = undo_delta_actions(quantile_unnormalize(norm_p, stats), state, delta_mask)
err = np.max(np.abs(back - prefix_abs))
assert err < 1e-5, f"round-trip error {err}"
# and it round-trips for ANY pose (the delta gotcha: model space depends on the state)
state2 = state + 0.5
norm_p2 = quantile_normalize(apply_delta_actions(prefix_abs, state2, delta_mask), stats)
back2 = undo_delta_actions(quantile_unnormalize(norm_p2, stats), state2, delta_mask)
assert np.max(np.abs(back2 - prefix_abs)) < 1e-5
assert not np.allclose(norm_p, norm_p2)  # model space really is pose-dependent
print(f"5. absolute prefix round-trip (max err {err:.1e}); model space pose-dependent  OK")

print("\nALL CPU UNIT TESTS PASS")
