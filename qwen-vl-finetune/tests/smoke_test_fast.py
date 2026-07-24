"""Smoke tests for FAST action-token supervision (KI recipe) on the tiny model.

Run from qwen-vl-finetune/ in the qwen3vl env:

    python tests/smoke_test_fast.py

Checks:
  1. forward with FAST tokens yields finite flow_loss, lm_loss (subtask) AND fast_loss;
  2. knowledge insulation still holds: the flow-matching loss alone never touches the
     VLM; the VLM gets gradients only through the CE (subtask + FAST) terms;
  3. anti-shortcut: the continuous expert's prediction v_t is INVARIANT to the FAST
     tokens (their presence and their content), proving it cannot read them.
"""

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

from tests.smoke_test_action_expert import build_tiny_vlm, check_insulation
from qwenvl.action_expert import ActionExpertConfig, Qwen3VLWithActionExpert


def make_batch(vocab, vlm, bsz=2, seq=20, horizon=8, adim=14, n_fast=6, device="cpu"):
    """A batch whose sequence is [obs/text ... | <start> FAST... <end>], with labels on
    the last few text tokens (subtask) and on the FAST region, plus fast_token_mask."""
    fast_base = vocab  # pretend the new FAST tokens start right after the base vocab
    input_ids = torch.randint(0, vocab, (bsz, seq), device=device)
    # append: <action_start> + n_fast fast tokens + <action_end>
    start_id, end_id = fast_base, fast_base + 1
    fast_ids = torch.randint(fast_base + 2, fast_base + 2 + 64, (bsz, n_fast), device=device)
    postfix = torch.cat([torch.full((bsz, 1), start_id, device=device), fast_ids,
                         torch.full((bsz, 1), end_id, device=device)], dim=1)
    input_ids = torch.cat([input_ids, postfix], dim=1)
    full = input_ids.shape[1]

    attention_mask = torch.ones(bsz, full, dtype=torch.bool, device=device)
    fast_token_mask = torch.zeros(bsz, full, dtype=torch.bool, device=device)
    fast_token_mask[:, -(n_fast + 2):] = True

    labels = torch.full((bsz, full), -100, device=device)
    labels[:, seq - 3:seq] = input_ids[:, seq - 3:seq]        # a few "subtask" text tokens
    labels[:, -(n_fast + 1):] = input_ids[:, -(n_fast + 1):]  # FAST tokens + end marker

    actions = torch.randn(bsz, horizon, adim, device=device)
    # resize embeddings so the pretend FAST ids are valid
    vlm.resize_token_embeddings(fast_base + 2 + 64)
    return dict(input_ids=input_ids, attention_mask=attention_mask, actions=actions,
                labels=labels, fast_token_mask=fast_token_mask)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    torch.manual_seed(0)

    vlm = build_tiny_vlm().to(device)
    vocab = vlm.config.text_config.vocab_size
    cfg = ActionExpertConfig(num_hidden_layers=2, num_key_value_heads=2, head_dim=16,
                             hidden_size=32, intermediate_size=64, num_attention_heads=4,
                             action_dim=14, action_horizon=8)
    model = Qwen3VLWithActionExpert(vlm, cfg, train_vlm=True).to(device)
    model.train()

    batch = make_batch(vocab, vlm, device=device)

    # (1) all three losses finite
    out = model(**batch)
    assert out.flow_loss is not None and torch.isfinite(out.flow_loss), "flow loss bad"
    assert out.lm_loss is not None and torch.isfinite(out.lm_loss), "subtask lm loss bad"
    assert out.fast_loss is not None and torch.isfinite(out.fast_loss), "fast loss bad"
    print(f"[losses] flow={out.flow_loss.item():.3f}  lm={out.lm_loss.item():.3f}  "
          f"fast={out.fast_loss.item():.3f}  total={out.loss.item():.3f}")

    # (2a) flow-only backward must not touch the VLM (insulation)
    model.zero_grad(set_to_none=True)
    flow_only = model(**{**batch, "labels": None})
    assert flow_only.fast_loss is None and flow_only.lm_loss is None
    flow_only.loss.backward()
    check_insulation(model, "fast/flow-only", expect_vlm_grads=False)

    # (2b) full backward: VLM gets grads (from CE), expert gets grads
    model.zero_grad(set_to_none=True)
    model(**batch).loss.backward()
    check_insulation(model, "fast/full", expect_vlm_grads=True)

    # (3) anti-shortcut: expert v_t is invariant to FAST tokens' presence & content.
    model.eval()
    fixed_noise = torch.randn(2, 8, 14, device=device)
    fixed_time = torch.full((2,), 0.5, device=device)

    def get_v(b):
        with torch.no_grad():
            am = b["attention_mask"]
            pos, _ = model.vlm.model.get_rope_index(b["input_ids"], attention_mask=am)
            pkv, _, _ = model._prefix_forward(
                b["input_ids"], am, pos, None, None, None, None,
                labels=None, fast_token_mask=b.get("fast_token_mask"))
            xt = fixed_time[:, None, None] * fixed_noise + (1 - fixed_time[:, None, None]) * b["actions"]
            return model._expert_forward(xt, fixed_time, pkv, am, pos, b.get("fast_token_mask"))

    # obs-only batch = drop the FAST postfix entirely
    seq0 = batch["input_ids"].shape[1] - 8  # n_fast(6)+2 markers
    obs_only = {"input_ids": batch["input_ids"][:, :seq0],
                "attention_mask": batch["attention_mask"][:, :seq0],
                "actions": batch["actions"], "fast_token_mask": None}
    v_obs = get_v(obs_only)
    v_with_fast = get_v(batch)

    # scramble FAST token CONTENT, keep mask
    scrambled = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    fm = batch["fast_token_mask"]
    scrambled["input_ids"] = batch["input_ids"].clone()
    scrambled["input_ids"][fm] = torch.randint(vocab, vocab + 64, (int(fm.sum()),), device=device)
    v_scrambled = get_v(scrambled)

    d_presence = (v_obs - v_with_fast).abs().max().item()
    d_content = (v_with_fast - v_scrambled).abs().max().item()
    print(f"[anti-shortcut] max |v_obs - v_with_fast| = {d_presence:.2e}  "
          f"max |v_with_fast - v_scrambled| = {d_content:.2e}")
    assert d_presence < 1e-4, "expert output changed when FAST tokens were added — leak!"
    assert d_content < 1e-4, "expert output changed with FAST content — shortcut leak!"
    print("[anti-shortcut] OK: expert cannot see FAST tokens")
    print("=== FAST smoke test PASSED ===")


if __name__ == "__main__":
    main()
