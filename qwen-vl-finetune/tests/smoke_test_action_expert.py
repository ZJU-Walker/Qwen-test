"""Smoke tests for the Qwen3-VL action expert.

Run from qwen-vl-finetune/ inside the qwen3vl conda env:

    python tests/smoke_test_action_expert.py            # tiny random model (fast)
    python tests/smoke_test_action_expert.py --real     # + real 4B model & dataset (GPU)

Checks:
  1. training forward produces a finite flow-matching loss;
  2. knowledge insulation: after backward, NO VLM parameter has a gradient while
     expert parameters do (frozen mode), and in train_vlm mode VLM grads come only
     from the LM loss (flow-only backward leaves the VLM untouched);
  3. sample_actions integrates the flow ODE and returns the right shape.
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))

from transformers import Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.configuration_qwen3_vl import (
    Qwen3VLConfig,
    Qwen3VLTextConfig,
    Qwen3VLVisionConfig,
)

from qwenvl.action_expert import ActionExpertConfig, Qwen3VLWithActionExpert


def build_tiny_vlm():
    text_config = Qwen3VLTextConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=151936,
        max_position_embeddings=4096,
        rope_theta=5000000,
        rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2], "mrope_interleaved": True},
    )
    vision_config = Qwen3VLVisionConfig(
        depth=2,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=64,
        deepstack_visual_indexes=[0, 1],
        num_position_embeddings=64,
    )
    config = Qwen3VLConfig(text_config=text_config.to_dict(), vision_config=vision_config.to_dict())
    config._attn_implementation = "sdpa"
    return Qwen3VLForConditionalGeneration(config)


def fake_text_batch(vocab_size, bsz=2, seq_len=24, horizon=8, action_dim=14, device="cpu"):
    input_ids = torch.randint(0, vocab_size, (bsz, seq_len), device=device)
    attention_mask = torch.ones(bsz, seq_len, dtype=torch.bool, device=device)
    attention_mask[1, -4:] = False  # exercise right padding
    actions = torch.randn(bsz, horizon, action_dim, device=device)
    return dict(input_ids=input_ids, attention_mask=attention_mask, actions=actions)


def check_insulation(model, tag, expect_vlm_grads=False):
    vlm_grads = [n for n, p in model.vlm.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    expert_params = [
        (n, p) for n, p in model.named_parameters() if not n.startswith("vlm.") and p.requires_grad
    ]
    expert_with_grad = [n for n, p in expert_params if p.grad is not None]
    assert len(expert_with_grad) > 0, f"[{tag}] expert got no gradients"
    if expect_vlm_grads:
        assert len(vlm_grads) > 0, f"[{tag}] expected VLM grads from LM loss, found none"
        print(f"[{tag}] OK: VLM has grads from LM loss only ({len(vlm_grads)} tensors)")
    else:
        assert len(vlm_grads) == 0, f"[{tag}] INSULATION VIOLATED: VLM grads on {vlm_grads[:5]}..."
        print(f"[{tag}] OK: 0 VLM grad tensors, {len(expert_with_grad)} expert grad tensors")


def test_tiny(device):
    print("=== tiny model test ===")
    torch.manual_seed(0)
    vlm = build_tiny_vlm().to(device)
    expert_config = ActionExpertConfig(
        num_hidden_layers=2,
        num_key_value_heads=2,
        head_dim=16,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        action_dim=14,
        action_horizon=8,
    )

    # --- frozen VLM (default): flow loss only, no grads may reach the VLM ---
    model = Qwen3VLWithActionExpert(vlm, expert_config, train_vlm=False).to(device)
    model.train()
    batch = fake_text_batch(vlm.config.text_config.vocab_size, device=device)
    out = model(**batch)
    assert torch.isfinite(out.loss), "loss is not finite"
    print(f"[frozen] flow loss = {out.loss.item():.4f}")
    out.loss.backward()
    check_insulation(model, "frozen", expect_vlm_grads=False)

    # --- train_vlm mode: flow grads still insulated; LM loss reaches the VLM ---
    torch.manual_seed(0)
    vlm2 = build_tiny_vlm().to(device)
    model2 = Qwen3VLWithActionExpert(vlm2, expert_config, train_vlm=True).to(device)
    model2.train()
    batch2 = fake_text_batch(vlm2.config.text_config.vocab_size, device=device)

    # flow loss alone (labels=None) must not touch the VLM even with requires_grad=True
    out2 = model2(**batch2)
    assert out2.lm_loss is None
    out2.loss.backward()
    check_insulation(model2, "train_vlm/flow-only", expect_vlm_grads=False)
    model2.zero_grad(set_to_none=True)

    # with labels, the VLM gets grads via the LM loss (and only via it)
    labels = batch2["input_ids"].clone()
    out3 = model2(**batch2, labels=labels)
    assert out3.lm_loss is not None and torch.isfinite(out3.lm_loss)
    print(f"[train_vlm] flow={out3.flow_loss.item():.4f} lm={out3.lm_loss.item():.4f}")
    out3.loss.backward()
    check_insulation(model2, "train_vlm/with-labels", expect_vlm_grads=True)

    # --- sampling ---
    model.eval()
    actions = model.sample_actions(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], num_steps=4
    )
    assert actions.shape == (2, 8, 14), actions.shape
    assert torch.isfinite(actions).all()
    print(f"[sampling] OK, shape {tuple(actions.shape)}")
    print("=== tiny model test PASSED ===\n")


def test_real(device):
    print("=== real 4B model + dataset test ===")
    import os

    os.environ.setdefault("HF_HOME", "/iris/projects/humanoid/ke/Qwen3-VL/qwen_cache/huggingface")
    from transformers import AutoProcessor

    from qwenvl.data.robot_data import RobotDataArguments, make_robot_data_module

    model_path = "Qwen/Qwen3-VL-4B-Instruct"
    cache_dir = os.environ["HF_HOME"]

    data_args = RobotDataArguments()
    data_args.model_type = "qwen3vl"
    processor = AutoProcessor.from_pretrained(model_path, cache_dir=cache_dir)
    data_module = make_robot_data_module(processor, data_args)
    dataset, collator = data_module["train_dataset"], data_module["data_collator"]
    batch = collator([dataset[0], dataset[1]])
    print(
        f"batch: input_ids {tuple(batch['input_ids'].shape)}, "
        f"video {tuple(batch['pixel_values_videos'].shape)}, actions {tuple(batch['actions'].shape)}"
    )

    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, cache_dir=cache_dir, attn_implementation="flash_attention_2", dtype=torch.bfloat16
    )
    text_config = vlm.config.text_config
    expert_config = ActionExpertConfig(
        num_hidden_layers=text_config.num_hidden_layers,
        num_key_value_heads=text_config.num_key_value_heads,
        head_dim=text_config.head_dim,
        action_dim=14,
        action_horizon=data_args.action_horizon,
    )
    model = Qwen3VLWithActionExpert(vlm, expert_config).to(device)
    model.action_expert.to(torch.bfloat16)
    print(f"expert params: {model.num_expert_parameters() / 1e6:.1f}M")
    model.train()

    batch = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in batch.items()
        if v is not None
    }
    out = model(**batch)
    assert torch.isfinite(out.loss)
    print(f"flow loss = {out.loss.item():.4f}")
    out.loss.backward()
    check_insulation(model, "real/frozen", expect_vlm_grads=False)
    model.zero_grad(set_to_none=True)

    model.eval()
    infer = {k: v for k, v in batch.items() if k not in ("actions", "labels")}
    actions = model.sample_actions(**infer, num_steps=10)
    assert actions.shape == (2, data_args.action_horizon, 14)
    assert torch.isfinite(actions).all()
    print(f"[sampling] OK, shape {tuple(actions.shape)}, mean {actions.mean().item():.3f}")
    print("=== real model test PASSED ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="also run the real 4B model + dataset test")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    test_tiny(device)
    if args.real:
        test_real(device)
