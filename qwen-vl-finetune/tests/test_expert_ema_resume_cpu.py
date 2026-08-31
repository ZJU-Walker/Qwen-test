#!/usr/bin/env python3
"""Focused CPU tests for exact action-expert checkpoint/EMA resume.

Run from qwen-vl-finetune:
    python tests/test_expert_ema_resume_cpu.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

FINETUNE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FINETUNE))

from qwenvl.train.expert_ema import (
    CheckpointValidationError,
    ExpertEMACallback,
    load_and_validate_ema,
    select_latest_complete_checkpoint,
    validate_action_expert_checkpoint,
)


DECAY = 0.999
WORLD_SIZE = 2


def _save(path: Path, value=None) -> None:
    torch.save({"value": torch.arange(3)} if value is None else value, path)


def _make_checkpoint(
    root: Path,
    step: int,
    *,
    decay: float = DECAY,
    params: dict[str, torch.Tensor] | None = None,
) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": step}))
    for filename in ("pytorch_model.bin", "training_args.bin", "scheduler.pt"):
        _save(checkpoint / filename)
    for rank in range(WORLD_SIZE):
        _save(checkpoint / f"rng_state_{rank}.pth")

    tag = f"global_step{max(step - 6, 0)}"  # DS tag need not equal Trainer's outer step.
    (checkpoint / "latest").write_text(tag)
    ds_state = checkpoint / tag
    ds_state.mkdir()
    _save(ds_state / "mp_rank_00_model_states.pt")
    for rank in range(WORLD_SIZE):
        _save(ds_state / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt")

    if params is None:
        params = {"action_expert.weight": torch.arange(6, dtype=torch.float32).view(2, 3)}
    _save(
        checkpoint / "ema_expert.pt",
        {"decay": decay, "step": step, "params": params},
    )
    return checkpoint


def _expect_invalid(call, contains: str) -> None:
    try:
        call()
    except CheckpointValidationError as exc:
        assert contains in str(exc), (contains, str(exc))
    else:
        raise AssertionError(f"expected CheckpointValidationError containing {contains!r}")


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vlm = nn.Linear(3, 2, bias=False)
        self.action_expert = nn.Linear(3, 2, bias=False)
        self.flow_head = nn.Parameter(torch.tensor([3.0], dtype=torch.float32))


def test_checkpoint_selection(root: Path) -> None:
    assert select_latest_complete_checkpoint(
        root,
        expected_ema_decay=DECAY,
        expected_world_size=WORLD_SIZE,
        require_deepspeed=True,
    ) is None

    complete = _make_checkpoint(root, 100)
    newest_torn = _make_checkpoint(root, 200)
    (newest_torn / "ema_expert.pt").unlink()  # kill after Trainer/DS save, before callback
    _expect_invalid(
        lambda: select_latest_complete_checkpoint(
            root,
            expected_ema_decay=DECAY,
            expected_world_size=WORLD_SIZE,
            require_deepspeed=True,
        ),
        "preserve it as a quarantine",
    )
    # Once the operator moves the torn directory out of the checkpoint namespace, the
    # previous complete checkpoint becomes the explicit, validated resume target.
    newest_torn.rename(root / "checkpoint-200.incomplete")
    selected = select_latest_complete_checkpoint(
        root,
        expected_ema_decay=DECAY,
        expected_world_size=WORLD_SIZE,
        require_deepspeed=True,
    )
    assert selected == complete

    # A non-empty optimizer file is still incomplete if torch.save never published its
    # ZIP central directory; checking mere existence/size would accept this.
    optimizer = next((complete / "global_step94").glob("*rank_1*optim_states.pt"))
    optimizer.write_bytes(b"PK\x03\x04torn")
    _expect_invalid(
        lambda: select_latest_complete_checkpoint(
            root,
            expected_ema_decay=DECAY,
            expected_world_size=WORLD_SIZE,
            require_deepspeed=True,
        ),
        "newest checkpoint is incomplete",
    )


def test_checkpoint_invariants(root: Path) -> None:
    checkpoint = _make_checkpoint(root, 300)
    validate_action_expert_checkpoint(
        checkpoint,
        expected_ema_decay=DECAY,
        expected_world_size=WORLD_SIZE,
        require_deepspeed=True,
    )

    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 299}))
    _expect_invalid(
        lambda: validate_action_expert_checkpoint(
            checkpoint,
            expected_ema_decay=DECAY,
            expected_world_size=WORLD_SIZE,
            require_deepspeed=True,
        ),
        "Trainer/global directory step mismatch",
    )
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 300}))

    missing_rank = checkpoint / "global_step294" / "bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt"
    missing_rank.unlink()
    _expect_invalid(
        lambda: validate_action_expert_checkpoint(
            checkpoint,
            expected_ema_decay=DECAY,
            expected_world_size=WORLD_SIZE,
            require_deepspeed=True,
        ),
        "optimizer ranks mismatch",
    )

    _save(missing_rank)
    ema_path = checkpoint / "ema_expert.pt"
    _save(
        ema_path,
        {"decay": 0.9, "step": 300,
         "params": {"action_expert.weight": torch.ones(2, 3, dtype=torch.float32)}},
    )
    _expect_invalid(
        lambda: load_and_validate_ema(ema_path, expected_step=300, expected_decay=DECAY),
        "EMA decay mismatch",
    )
    _save(
        ema_path,
        {"decay": DECAY, "step": 299,
         "params": {"action_expert.weight": torch.ones(2, 3, dtype=torch.float32)}},
    )
    _expect_invalid(
        lambda: load_and_validate_ema(ema_path, expected_step=300, expected_decay=DECAY),
        "EMA step mismatch",
    )
    _save(
        ema_path,
        {"decay": DECAY, "step": 300,
         "params": {"action_expert.weight": torch.ones(2, 3, dtype=torch.float16)}},
    )
    _expect_invalid(
        lambda: load_and_validate_ema(ema_path, expected_step=300, expected_decay=DECAY),
        "expected fp32",
    )


def test_callback_restore_update_and_fresh(root: Path) -> None:
    model = _ToyModel()
    saved = {
        "action_expert.weight": torch.full_like(model.action_expert.weight, 2.0, dtype=torch.float32),
        "flow_head": torch.tensor([4.0], dtype=torch.float32),
    }
    checkpoint = _make_checkpoint(root, 7, params=saved)
    with torch.no_grad():
        model.action_expert.weight.fill_(10.0)
        model.flow_head.fill_(10.0)
        model.vlm.weight.fill_(99.0)

    callback = ExpertEMACallback(model, decay=DECAY, resume_checkpoint=checkpoint)
    state = SimpleNamespace(global_step=7)
    callback.on_train_begin(None, state, None)
    assert set(callback.shadow) == set(saved)
    assert torch.equal(callback.shadow["action_expert.weight"], saved["action_expert.weight"])
    assert torch.equal(callback.shadow["flow_head"], saved["flow_head"])
    assert "vlm.weight" not in callback.shadow

    with torch.no_grad():
        model.action_expert.weight.fill_(6.0)
        model.flow_head.fill_(8.0)
    callback.on_step_end(None, SimpleNamespace(global_step=8), None)
    assert torch.allclose(
        callback.shadow["action_expert.weight"],
        saved["action_expert.weight"] * DECAY + 6.0 * (1.0 - DECAY),
    )
    assert torch.allclose(
        callback.shadow["flow_head"], saved["flow_head"] * DECAY + 8.0 * (1.0 - DECAY)
    )

    # Fresh and weights-only initialization both have resume_checkpoint=None: snapshot
    # the already-initialized model rather than looking for or resetting an EMA file.
    fresh = ExpertEMACallback(model, decay=DECAY)
    fresh.on_train_begin(None, SimpleNamespace(global_step=0), None)
    assert torch.equal(fresh.shadow["action_expert.weight"], model.action_expert.weight.float())

    # Exact resume fails closed on a model/EMA name mismatch.
    bad = _make_checkpoint(root, 8, params={"wrong.name": torch.ones(1, dtype=torch.float32)})
    mismatch = ExpertEMACallback(model, decay=DECAY, resume_checkpoint=bad)
    _expect_invalid(
        lambda: mismatch.on_train_begin(None, SimpleNamespace(global_step=8), None),
        "EMA tensor names mismatch",
    )
    wrong_shape = dict(saved)
    wrong_shape["action_expert.weight"] = torch.ones(1, dtype=torch.float32)
    bad_shape = _make_checkpoint(root, 9, params=wrong_shape)
    mismatch = ExpertEMACallback(model, decay=DECAY, resume_checkpoint=bad_shape)
    _expect_invalid(
        lambda: mismatch.on_train_begin(None, SimpleNamespace(global_step=9), None),
        "EMA tensor shape mismatch",
    )


def test_atomic_callback_save(root: Path) -> None:
    model = _ToyModel()
    callback = ExpertEMACallback(model, decay=DECAY)
    state = SimpleNamespace(global_step=9)
    callback.on_train_begin(None, SimpleNamespace(global_step=0), None)
    checkpoint = root / "checkpoint-9"
    checkpoint.mkdir(parents=True)
    callback.on_save(SimpleNamespace(output_dir=str(root)), state, None)
    payload = load_and_validate_ema(
        checkpoint / "ema_expert.pt", expected_step=9, expected_decay=DECAY
    )
    assert set(payload["params"]) == {"action_expert.weight", "flow_head"}
    assert not list(checkpoint.glob(".ema_expert.pt.*.tmp"))


def main() -> None:
    # Make callback rank behavior deterministic even if this test is launched from a
    # shell that happens to retain distributed environment variables.
    old_rank = os.environ.get("RANK")
    os.environ["RANK"] = "0"
    try:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            test_checkpoint_selection(root / "selection")
            test_checkpoint_invariants(root / "invariants")
            test_callback_restore_update_and_fresh(root / "callback")
            test_atomic_callback_save(root / "save")
    finally:
        if old_rank is None:
            os.environ.pop("RANK", None)
        else:
            os.environ["RANK"] = old_rank
    print("ALL expert EMA exact-resume CPU CHECKS PASS")


if __name__ == "__main__":
    main()
