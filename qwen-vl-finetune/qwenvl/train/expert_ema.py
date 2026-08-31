"""Exact-resume support for the action expert's EMA and DeepSpeed checkpoints.

The Hugging Face checkpoint is written in several phases and the EMA callback runs
last.  A Slurm kill can therefore leave a numerically newest ``checkpoint-*``
directory that exists but cannot be resumed exactly.  This module validates that
multi-file commit before selecting a checkpoint and restores the EMA only after the
Trainer has restored the model and DeepSpeed state.
"""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import torch
import transformers


_CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")
_ZERO_RANK_RE = re.compile(r"zero_pp_rank_(\d+)")


class CheckpointValidationError(RuntimeError):
    """A checkpoint is present but is not safe for an exact training resume."""


def _require_file(path: Path, description: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise CheckpointValidationError(f"missing or empty {description}: {path}")


def _require_complete_torch_archive(path: Path, description: str) -> None:
    """Cheaply reject a torch.save interrupted before its ZIP central directory.

    Opening the central directory is O(number of tensors), not O(file size), so this
    also works for the 26-GB ZeRO optimizer shards without reading their tensor data.
    All checkpoints produced by this training environment use PyTorch's ZIP format.
    """
    _require_file(path, description)
    try:
        with zipfile.ZipFile(path) as archive:
            if not archive.infolist():
                raise CheckpointValidationError(f"empty torch archive for {description}: {path}")
    except (OSError, zipfile.BadZipFile) as exc:
        raise CheckpointValidationError(
            f"incomplete/unreadable torch archive for {description}: {path}: {exc}"
        ) from exc


def load_and_validate_ema(
    ema_path: str | os.PathLike[str],
    *,
    expected_step: int,
    expected_decay: float,
) -> dict[str, Any]:
    """Load EMA metadata/tensors and validate its checkpoint-level invariants."""
    path = Path(ema_path)
    _require_complete_torch_archive(path, "EMA state")
    try:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    except Exception as exc:
        raise CheckpointValidationError(f"cannot load EMA state {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise CheckpointValidationError(f"EMA state is not a dictionary: {path}")
    if set(payload) != {"decay", "step", "params"}:
        raise CheckpointValidationError(
            f"EMA state keys mismatch in {path}: expected decay/step/params, "
            f"got {sorted(map(str, payload))}"
        )
    step = payload["step"]
    if isinstance(step, bool) or not isinstance(step, int) or step != expected_step:
        raise CheckpointValidationError(
            f"EMA step mismatch in {path}: saved={step!r}, expected={expected_step}"
        )
    decay = payload["decay"]
    if (isinstance(decay, bool) or not isinstance(decay, (int, float))
            or not math.isfinite(float(decay))
            or not math.isclose(float(decay), float(expected_decay), rel_tol=0.0, abs_tol=1e-12)):
        raise CheckpointValidationError(
            f"EMA decay mismatch in {path}: saved={decay!r}, expected={expected_decay}"
        )
    params = payload["params"]
    if not isinstance(params, dict) or not params:
        raise CheckpointValidationError(f"EMA params are missing/empty in {path}")
    for name, tensor in params.items():
        if not isinstance(name, str) or not torch.is_tensor(tensor):
            raise CheckpointValidationError(
                f"EMA params must map string names to tensors in {path}; bad entry {name!r}"
            )
        if tensor.dtype != torch.float32:
            raise CheckpointValidationError(
                f"EMA tensor {name!r} in {path} has dtype {tensor.dtype}; expected fp32"
            )
    return payload


def validate_action_expert_checkpoint(
    checkpoint: str | os.PathLike[str],
    *,
    expected_ema_decay: float | None,
    expected_world_size: int,
    require_deepspeed: bool,
) -> None:
    """Validate the files required to resume this Trainer run exactly.

    DeepSpeed's internal ``global_step*`` tag is intentionally not compared with the
    Trainer step: after an earlier resume it can lag the outer checkpoint name (for
    example, the validated baseline checkpoint-6000 contains global_step5994).  The
    ``latest`` tag must instead resolve safely to a complete state directory.
    """
    path = Path(checkpoint)
    match = _CHECKPOINT_RE.fullmatch(path.name)
    if not path.is_dir() or match is None:
        raise CheckpointValidationError(f"invalid checkpoint directory name: {path}")
    checkpoint_step = int(match.group(1))
    if expected_world_size < 1:
        raise ValueError(f"expected_world_size must be positive, got {expected_world_size}")

    trainer_state_path = path / "trainer_state.json"
    _require_file(trainer_state_path, "Trainer state")
    try:
        with trainer_state_path.open() as stream:
            trainer_state = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointValidationError(
            f"invalid Trainer state JSON {trainer_state_path}: {exc}"
        ) from exc
    trainer_step = trainer_state.get("global_step")
    if (isinstance(trainer_step, bool) or not isinstance(trainer_step, int)
            or trainer_step != checkpoint_step):
        raise CheckpointValidationError(
            f"Trainer/global directory step mismatch at {path}: "
            f"trainer_state={trainer_step!r}, directory={checkpoint_step}"
        )

    for filename, description in (
        ("pytorch_model.bin", "model weights"),
        ("training_args.bin", "training arguments"),
        ("scheduler.pt", "scheduler state"),
    ):
        _require_complete_torch_archive(path / filename, description)

    if expected_world_size == 1:
        rng_paths = [path / "rng_state.pth"]
        # A one-process torchrun can still use the distributed filename.
        if not rng_paths[0].is_file() and (path / "rng_state_0.pth").is_file():
            rng_paths = [path / "rng_state_0.pth"]
    else:
        rng_paths = [path / f"rng_state_{rank}.pth" for rank in range(expected_world_size)]
    for rank, rng_path in enumerate(rng_paths):
        _require_complete_torch_archive(rng_path, f"RNG state for rank {rank}")

    if require_deepspeed:
        latest_path = path / "latest"
        _require_file(latest_path, "DeepSpeed latest tag")
        try:
            tag = latest_path.read_text().strip()
        except OSError as exc:
            raise CheckpointValidationError(f"cannot read DeepSpeed tag {latest_path}: {exc}") from exc
        # Never follow absolute paths, '..', or nested paths from checkpoint metadata.
        if not tag or Path(tag).name != tag or tag in {".", ".."}:
            raise CheckpointValidationError(f"unsafe/empty DeepSpeed latest tag in {latest_path}: {tag!r}")
        ds_state_dir = path / tag
        if not ds_state_dir.is_dir():
            raise CheckpointValidationError(
                f"DeepSpeed latest tag {tag!r} has no state directory in {path}"
            )
        model_states = sorted(ds_state_dir.glob("*model_states.pt"))
        if not model_states:
            raise CheckpointValidationError(f"no DeepSpeed model state under {ds_state_dir}")
        for model_state in model_states:
            _require_complete_torch_archive(model_state, "DeepSpeed model state")

        optimizer_states = sorted(ds_state_dir.glob("*optim_states.pt"))
        found_ranks: set[int] = set()
        for optimizer_state in optimizer_states:
            rank_match = _ZERO_RANK_RE.search(optimizer_state.name)
            if rank_match is None:
                raise CheckpointValidationError(
                    f"cannot identify ZeRO rank from optimizer shard: {optimizer_state}"
                )
            found_ranks.add(int(rank_match.group(1)))
            _require_complete_torch_archive(optimizer_state, "DeepSpeed optimizer state")
        expected_ranks = set(range(expected_world_size))
        if found_ranks != expected_ranks:
            raise CheckpointValidationError(
                f"DeepSpeed optimizer ranks mismatch under {ds_state_dir}: "
                f"found={sorted(found_ranks)}, expected={sorted(expected_ranks)}"
            )

    if expected_ema_decay is not None:
        load_and_validate_ema(
            path / "ema_expert.pt",
            expected_step=checkpoint_step,
            expected_decay=expected_ema_decay,
        )


def select_latest_complete_checkpoint(
    output_dir: str | os.PathLike[str],
    *,
    expected_ema_decay: float | None,
    expected_world_size: int,
    require_deepspeed: bool,
) -> Path | None:
    """Return the newest checkpoint only when it is complete.

    We deliberately do not silently roll back past a torn newest directory. If training
    later reaches that same step, Hugging Face/DeepSpeed reuse the directory; retaining
    it could mix files from two save attempts. The operator must first move the torn
    directory outside ``output_dir`` (a recoverable quarantine), then relaunch.
    """
    root = Path(output_dir)
    candidates: list[tuple[int, Path]] = []
    if root.is_dir():
        for candidate in root.iterdir():
            match = _CHECKPOINT_RE.fullmatch(candidate.name)
            if candidate.is_dir() and match is not None:
                candidates.append((int(match.group(1)), candidate))
    if not candidates:
        return None

    _, newest = max(candidates, key=lambda item: item[0])
    try:
        validate_action_expert_checkpoint(
            newest,
            expected_ema_decay=expected_ema_decay,
            expected_world_size=expected_world_size,
            require_deepspeed=require_deepspeed,
        )
    except CheckpointValidationError as exc:
        raise CheckpointValidationError(
            f"newest checkpoint is incomplete and cannot be resumed exactly: {newest}: {exc}. "
            f"Move {newest} outside {root} (preserve it as a quarantine for inspection), "
            "then relaunch to validate the next checkpoint. It is not safe to retain and "
            "silently roll back because a later save would reuse this step directory."
        ) from exc
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"[resume] selected complete checkpoint: {newest}", flush=True)
    return newest


def _atomic_torch_save(payload: Any, destination: Path) -> None:
    """Publish a torch archive only after torch.save has completed."""
    fd, tmp_name = tempfile.mkstemp(dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, destination)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


class ExpertEMACallback(transformers.TrainerCallback):
    """Rank-0 fp32 EMA of all trainable non-VLM parameters.

    For a fresh or weights-only run the shadow starts from the model at train begin.
    For an exact Trainer resume, it is restored at train begin: this hook runs after
    model/DeepSpeed, optimizer/scheduler, and Trainer state restoration in Transformers.
    """

    def __init__(
        self,
        model,
        decay: float = 0.99,
        resume_checkpoint: str | os.PathLike[str] | None = None,
    ):
        self.decay = decay
        self.model = model
        self.resume_checkpoint = Path(resume_checkpoint) if resume_checkpoint is not None else None
        self._is_rank0 = int(os.environ.get("RANK", 0)) == 0
        self._tracked = None
        self.shadow = None

    def _tracked_parameters(self):
        return [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad and not name.startswith("vlm.")
        ]

    @torch.no_grad()
    def _snapshot_shadow(self) -> None:
        self._tracked = self._tracked_parameters()
        self.shadow = {name: parameter.detach().float().clone() for name, parameter in self._tracked}
        self._log_summary("initialized from current model")

    def _log_summary(self, source: str) -> None:
        n_params = sum(value.numel() for value in self.shadow.values())
        device = next(iter(self.shadow.values())).device
        print(
            f"[ema] tracking {len(self.shadow)} tensors ({n_params / 1e6:.0f}M params) "
            f"on {device}, decay={self.decay} (~{1 / (1 - self.decay):.0f}-step average), "
            f"fp32 shadow {n_params * 4 / 1e9:.1f} GB; {source}",
            flush=True,
        )

    @torch.no_grad()
    def _restore_shadow(self, state) -> None:
        self._tracked = self._tracked_parameters()
        expected = {name: parameter for name, parameter in self._tracked}
        payload = load_and_validate_ema(
            self.resume_checkpoint / "ema_expert.pt",
            expected_step=state.global_step,
            expected_decay=self.decay,
        )
        saved = payload["params"]
        missing = sorted(set(expected) - set(saved))
        unexpected = sorted(set(saved) - set(expected))
        if missing or unexpected:
            raise CheckpointValidationError(
                f"EMA tensor names mismatch at {self.resume_checkpoint}: "
                f"missing={missing[:8]}{' ...' if len(missing) > 8 else ''}, "
                f"unexpected={unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}"
            )
        for name, parameter in self._tracked:
            tensor = saved[name]
            if tensor.shape != parameter.shape:
                raise CheckpointValidationError(
                    f"EMA tensor shape mismatch for {name!r} at {self.resume_checkpoint}: "
                    f"saved={tuple(tensor.shape)}, model={tuple(parameter.shape)}"
                )
        self.shadow = {
            name: saved[name].to(device=parameter.device, dtype=torch.float32, copy=True)
            for name, parameter in self._tracked
        }
        self._log_summary(f"restored exactly from {self.resume_checkpoint.name} step {state.global_step}")

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._is_rank0 or self.shadow is not None:
            return
        if self.resume_checkpoint is None:
            self._snapshot_shadow()
        else:
            self._restore_shadow(state)

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        if not self._is_rank0:
            return
        if self.shadow is None:
            raise RuntimeError(
                "EMA shadow is uninitialized at on_step_end; on_train_begin did not complete"
            )
        decay = self.decay
        for name, parameter in self._tracked:
            buffer = self.shadow[name]
            if buffer.device != parameter.device:
                buffer = self.shadow[name] = buffer.to(parameter.device)
            buffer.mul_(decay).add_(parameter.detach().float(), alpha=1.0 - decay)

    def on_save(self, args, state, control, **kwargs):
        if not self._is_rank0 or self.shadow is None:
            return
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if checkpoint.is_dir():
            _atomic_torch_save(
                {
                    "decay": self.decay,
                    "step": state.global_step,
                    "params": {name: value.cpu() for name, value in self.shadow.items()},
                },
                checkpoint / "ema_expert.pt",
            )
            print(f"[ema] saved ema_expert.pt @ step {state.global_step}", flush=True)
