"""Train a flow-matching action expert on top of Qwen3-VL (pi0.5-style, with
knowledge insulation: expert gradients never touch the VLM weights).

Launch with scripts/train_action_expert_4b.sh.
"""

import json
import logging
import math
import multiprocessing
import os
import signal
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import transformers

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Trainer

from qwenvl.action_expert import ActionExpertConfig, Qwen3VLWithActionExpert
from qwenvl.data.robot_data import RobotDataArguments, make_robot_data_module
from qwenvl.train.argument import TrainingArguments
from qwenvl.train.expert_ema import ExpertEMACallback, select_latest_complete_checkpoint

# QWEN_STALL_DEBUG=1: SIGUSR1 appends ALL thread stacks of this rank to
# /tmp/qwen_stall_rank<R>.log (in-process faulthandler -- works where ptrace/py-spy is
# forbidden). Fatal-signal tracebacks (including SIGSEGV) go to the same file. Dumps are
# deliberately on demand: dump_traceback_later(repeat=True) itself reproducibly killed
# rank 0 while walking its threads on the 90-second tick (2026-08-07).
# Arm only the two rank processes: spawned dataloader workers import this module too,
# and previously all of them wrote interleaved dumps into their parent's rank log.
if (os.environ.get("QWEN_STALL_DEBUG")
        and multiprocessing.current_process().name == "MainProcess"):
    import faulthandler
    _stall_log = open(f"/tmp/qwen_stall_rank{os.environ.get('RANK', '0')}.log", "a")
    _stall_log.write(f"\n===== new run pid={os.getpid()} =====\n")
    _stall_log.write("Send SIGUSR1 to this pid for an on-demand all-thread dump.\n")
    _stall_log.flush()
    faulthandler.enable(file=_stall_log, all_threads=True)
    faulthandler.register(signal.SIGUSR1, file=_stall_log, all_threads=True, chain=False)


@dataclass
class ActionExpertModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3-VL-4B-Instruct")

    # Expert size (kv heads / head_dim are copied from the VLM).
    expert_hidden_size: int = field(default=1024)
    expert_intermediate_size: int = field(default=4096)
    expert_num_attention_heads: int = field(default=16)
    # SmolVLA-style layer skipping (arXiv:2506.01844): give the expert only N layers,
    # each attending to VLM layer i (i=0..N-1). 0 = use all VLM layers (default, current
    # behavior). SmolVLA's verdict was N = L/2 (half): near-parity task performance at
    # ~half the VLM/expert cost. For Qwen3-VL-4B (L=36) that is 18. At serve time (no
    # subtask decode) the VLM can early-exit at layer N, so this compounds with implicit
    # HL (--expert_attends_subtask False). Must match between train and serve.
    expert_num_layers: int = field(default=0)

    # (action_dim / action_horizon live in RobotDataArguments — they describe the data.)

    # Knowledge-insulation co-training: also train the VLM with next-token prediction
    # on the subtask labels. Expert gradients are detached from the VLM either way.
    train_vlm: bool = field(default=False)
    lm_loss_weight: float = field(default=1.0)
    # FAST action-token cross-entropy weight (KI uses standard weight 1). FAST is
    # enabled from the data side via --use_fast_tokens / --fast_tokenizer_path.
    fast_loss_weight: float = field(default=1.0)
    # LR for the pretrained VLM when train_vlm=True (the main --learning_rate applies
    # to the from-scratch expert; a pretrained 4B needs a far smaller finetuning LR).
    vlm_learning_rate: float = field(default=1e-5)
    tune_mm_vision: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_llm: bool = field(default=True)
    # Training-time RTC (arXiv:2512.05964): sample an action-prefix length (simulated
    # inference delay, in action steps) d ~ Uniform[min, max] per example; the first d
    # ground-truth actions become a clean conditioning prefix and only the postfix is
    # supervised. RTC is active iff rtc_prefix_max_length > 0. Must be < action_horizon.
    # Orthogonal to image_history / predict_subtask -- works with every variant.
    rtc_prefix_min_length: int = field(default=0)
    rtc_prefix_max_length: int = field(default=0)
    # Subtask insulation: False -> the expert's attention EXCLUDES the assistant subtask
    # turn (conditions on images+state only; subtask stays a pure VLM co-training signal).
    # Lets the server run the expert without waiting for subtask generation. Must match at
    # serve time (--insulated_subtask on the server). Only meaningful with predict_subtask.
    expert_attends_subtask: bool = field(default=True)
    # Per-sample (mean-of-per-token-means) averaging for the subtask-text CE, so each
    # question counts equally regardless of answer length. Matters once one batch mixes
    # answer formats of very different lengths (e.g. a full task sentence vs a bin word);
    # pooled per-token CE would weight a sample by its answer length.
    lm_loss_per_sample: bool = field(default=False)
    # Warm-start: load model WEIGHTS from this checkpoint dir's pytorch_model.bin, but start
    # a FRESH optimizer/scheduler/step counter (unlike auto-resume, which restores all of
    # those and requires the same GPU count for ZeRO-2). Use this to continue a run under a
    # new regime -- e.g. more GPUs / a larger batch -- from an existing checkpoint. Point
    # --output_dir at a NEW empty dir so auto-resume does not also kick in.
    init_from: str = field(default="")
    # EMA of the expert (non-vlm) weights, deployed at inference like openpi/Diffusion
    # Policy do (see ExpertEMACallback). 0.99 ~= 100-step average. 0 = off.
    ema_decay: float = field(default=0.99)


def rank0_print(*args):
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args)


class ActionExpertTrainer(Trainer):
    """Trainer with two LR groups: expert params at --learning_rate, VLM params at
    --vlm_learning_rate (only relevant when train_vlm=True; frozen VLM params are
    excluded from the optimizer entirely)."""

    def __init__(self, *args, vlm_learning_rate: float = 1e-5, **kwargs):
        super().__init__(*args, **kwargs)
        self.vlm_learning_rate = vlm_learning_rate

    def create_optimizer(self):
        if self.optimizer is None:
            expert_params = [
                p for n, p in self.model.named_parameters()
                if p.requires_grad and not n.startswith("vlm.")
            ]
            vlm_params = [
                p for n, p in self.model.named_parameters()
                if p.requires_grad and n.startswith("vlm.")
            ]
            # weight_decay MUST be set explicitly on each group: torch fills any missing
            # param-group option from the optimizer's constructor defaults, and AdamW's
            # default is 0.01 -- which silently overrode --weight_decay 0 here for months
            # (found in the 2026-07-28 optimizer audit). openpi uses ~0 (1e-10); Qwen's own
            # SFT uses 0 via the stock Trainer, which wires this field correctly.
            wd = self.args.weight_decay
            groups = [{"params": expert_params, "lr": self.args.learning_rate, "weight_decay": wd}]
            if vlm_params:
                groups.append({"params": vlm_params, "lr": self.vlm_learning_rate, "weight_decay": wd})
            optim_cls, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            optim_kwargs.pop("lr", None)
            optim_kwargs.pop("weight_decay", None)  # per-group above; don't let kwargs fight it
            self.optimizer = optim_cls(groups, **optim_kwargs)
            rank0_print(f"[optimizer] {optim_cls.__name__}  betas={optim_kwargs.get('betas')}  "
                        f"weight_decay={wd} (both groups)  "
                        f"expert lr {self.args.learning_rate} ({len(expert_params)} tensors), "
                        f"vlm lr {self.vlm_learning_rate} ({len(vlm_params)} tensors)")
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if os.environ.get("QWEN_NAN_DEBUG"):
            # Pre-forward scan of the embedding head rows: catches corruption written by
            # the PREVIOUS micro-batch's backward (compute_loss runs forward only; the
            # backward happens in the trainer after we return). Microsecond cost.
            self._micro_idx = getattr(self, "_micro_idx", 0) + 1
            m = model
            while hasattr(m, "module"):
                m = m.module
            w = m.vlm.model.language_model.embed_tokens.weight
            if not torch.isfinite(w[:16]).all():
                bad = (~torch.isfinite(w[:16]).all(dim=1)).nonzero().flatten().tolist()
                raise RuntimeError(
                    f"[QWEN_NAN_DEBUG] embed rows {bad} went non-finite BETWEEN forwards: "
                    f"written by the BACKWARD of micro-batch {self._micro_idx - 1} "
                    f"(global_step {self.state.global_step}) -- an out-of-place write "
                    f"during autograd, not the optimizer.")
        outputs = model(**inputs)
        # Stash the loss components (last micro-batch of the logging interval) so
        # flow / subtask-CE / FAST-CE contributions show up separately in the logs.
        self._last_flow_loss = outputs.flow_loss.item() if outputs.flow_loss is not None else None
        self._last_lm_loss = outputs.lm_loss.item() if outputs.lm_loss is not None else None
        self._last_fast_loss = outputs.fast_loss.item() if outputs.fast_loss is not None else None
        if os.environ.get("QWEN_NAN_DEBUG"):
            comps = {"flow_loss": self._last_flow_loss, "lm_loss": self._last_lm_loss,
                     "fast_loss": self._last_fast_loss}
            if any(v is not None and not math.isfinite(v) for v in comps.values()):
                # Discriminate "weights already corrupt" (a previous optimizer step did
                # it) from "this forward created the nan" (a data-content trigger).
                bad_params = [n for n, p in model.named_parameters()
                              if not torch.isfinite(p).all()]
                prov = list(getattr(self.train_dataset, "_recent_items", []))
                dump_path = os.path.join(self.args.output_dir, "nan_batch.pt")
                torch.save({"inputs": {k: (v.detach().cpu() if torch.is_tensor(v) else v)
                                       for k, v in inputs.items()},
                            "components": comps, "bad_params": bad_params,
                            "provenance": prov, "global_step": self.state.global_step},
                           dump_path)
                raise RuntimeError(
                    f"[QWEN_NAN_DEBUG] non-finite loss {comps} at global_step "
                    f"{self.state.global_step}.\n"
                    f"non-finite PARAMETERS ({len(bad_params)}): {bad_params[:8]}"
                    f"{' ...' if len(bad_params) > 8 else ''}\n"
                    f"(empty list => weights healthy => THIS batch's content triggered it)\n"
                    f"last samples built (newest last): {prov[-4:] if prov else 'none - run with NUM_WORKERS=0'}\n"
                    f"batch saved to {dump_path} for tests/replay_nan_batch.py")
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def log(self, logs, start_time=None):
        if "loss" in logs:
            for name, attr in [("flow_loss", "_last_flow_loss"), ("lm_loss", "_last_lm_loss"),
                               ("fast_loss", "_last_fast_loss")]:
                val = getattr(self, attr, None)
                if val is not None:
                    logs[name] = round(val, 4)
        return super().log(logs, start_time)


class EmbedNanWatchCallback(transformers.TrainerCallback):
    """QWEN_NAN_DEBUG aid: scan the (tied) token-embedding table once at train begin and
    after EVERY optimizer step; crash the moment corruption appears. Pinpoints WHICH
    update wrote non-finite values, into WHICH rows (new FAST/marker tokens start at the
    old vocab size), and whether they are inf (overflow) or nan (e.g. inf*0 clip scale).
    Cost: one isfinite reduction over the embedding per step (~ms)."""

    def __init__(self, model):
        m = model
        while hasattr(m, "module"):
            m = m.module
        self._emb = m.vlm.model.language_model.embed_tokens
        self._head = m.vlm.lm_head

    def _check(self, state, where):
        w = self._emb.weight
        bad_rows = ~torch.isfinite(w).all(dim=1)
        if bad_rows.any():
            ids = bad_rows.nonzero().flatten()
            first_bad_vals = w[ids[0]][~torch.isfinite(w[ids[0]])][:6]
            raise RuntimeError(
                f"[EmbedNanWatch] {int(bad_rows.sum())} non-finite embedding rows {where} "
                f"(global_step {state.global_step}): row ids {ids[:16].tolist()}"
                f"{' ...' if len(ids) > 16 else ''} | sample bad values "
                f"{first_bad_vals.float().tolist()} | "
                f"tied_head={self._head.weight is w}")

    def on_train_begin(self, args, state, control, **kwargs):
        self._check(state, "AT INIT, before any update")

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        # After ALL the cycle's backwards, before clip+step: weights should still be
        # clean here (backward writes grads, not params) and the GRAD tells us whether
        # the optimizer is about to be fed non-finite values for the head rows.
        self._check(state, "after backwards, BEFORE optimizer step")
        # DeepSpeed's ZeRO-2 safe_get_full_grad() is a full-parameter all-reduce. Its
        # availability can differ by rank at this hook (one rank may raise while another
        # enters the collective), which deadlocks the process group. The replicated
        # weight checks above and in on_step_end remain safe and are the load-bearing
        # diagnostics; keep this optional gradient probe single-rank only.
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            return
        g = None
        try:
            # Best-effort: some DeepSpeed versions refuse grad access at this hook point.
            from deepspeed.utils import safe_get_full_grad
            g = safe_get_full_grad(self._emb.weight)
        except Exception:
            pass
        if g is not None and not torch.isfinite(g[:16]).all():
            bad = (~torch.isfinite(g[:16]).all(dim=1)).nonzero().flatten().tolist()
            raise RuntimeError(
                f"[EmbedNanWatch] embed GRAD rows {bad} non-finite before optimizer "
                f"step (global_step {state.global_step}): the backward produced them; "
                f"the optimizer step would smear them into the weights.")

    def on_step_end(self, args, state, control, **kwargs):
        self._check(state, "after optimizer step")


class FirstStepCacheFlushCallback(transformers.TrainerCallback):
    """One-shot cache release after the FIRST optimizer step of a (re)started run.

    Why (2026-08-20, iris-hgx-1): the trainer's very first logging gather
    (_maybe_log_save_evaluate -> all_gather of tr_loss, logging_steps=1) is the
    default process group's FIRST NCCL collective, and NCCL cudaMallocs its
    communicator buffers OUTSIDE PyTorch's caching allocator -- at the exact moment
    PyTorch's reserved pool sits at the resume peak. With another process holding
    ~1 GiB (the allocation-keeper) three resume attempts OOMed deterministically at
    step N+1 in that allgather, while the same code ran on an empty GPU. Releasing
    the cache once, between the first optimizer step and the first gather
    (on_step_end fires before _maybe_log_save_evaluate), hands the unused reserved
    memory back to the driver so NCCL's raw allocation fits. One-time ~100 ms cost;
    steady-state caching behavior is untouched afterwards."""

    def __init__(self):
        self._done = False

    def on_step_end(self, args, state, control, **kwargs):
        if not self._done:
            self._done = True
            import gc

            gc.collect()
            torch.cuda.empty_cache()


def set_vlm_trainable(model_args, vlm):
    vlm.visual.requires_grad_(model_args.tune_mm_vision)
    vlm.visual.merger.requires_grad_(model_args.tune_mm_mlp)
    vlm.model.language_model.requires_grad_(model_args.tune_mm_llm)
    vlm.lm_head.requires_grad_(model_args.tune_mm_llm)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa: SLF001


def atomic_json_dump(obj, path: str):
    """Serving may read these stamps while a preempted run relaunches -- a bare
    open(w)+dump exposes a truncated file for the write duration (and forever, after a
    kill mid-write). mkstemp + os.replace makes the swap atomic."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def save_expert_only(model: Qwen3VLWithActionExpert, output_dir: str):
    """Small standalone checkpoint of just the action expert + flow heads."""
    expert_state = {
        name: param.detach().cpu()
        for name, param in model.state_dict().items()
        if not name.startswith("vlm.")
    }
    torch.save(
        {"expert_config": model.expert_config.to_dict(), "state_dict": expert_state},
        os.path.join(output_dir, "action_expert.pt"),
    )


def train(attn_implementation="flash_attention_2"):
    parser = transformers.HfArgumentParser(
        (ActionExpertModelArguments, RobotDataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)
    if not 0.0 <= model_args.ema_decay < 1.0:
        raise ValueError(f"--ema_decay must be in [0, 1), got {model_args.ema_decay}")
    if (not math.isfinite(data_args.current_state_mask_prob)
            or not 0.0 <= data_args.current_state_mask_prob <= 1.0):
        raise ValueError(
            "--current_state_mask_prob must be finite and in [0, 1], got "
            f"{data_args.current_state_mask_prob}"
        )
    if data_args.current_state_mask_prob > 0.0 and data_args.state_history:
        raise ValueError(
            "--current_state_mask_prob > 0 requires --state_history False; "
            "otherwise Past states would defeat the intended proprioception mask"
        )

    # Resolve an explicit, validated resume target before loading the model or touching
    # warm-start weights. Transformers' implicit True path picks the numerically highest
    # directory even when a preemption interrupted its multi-file save. A torn newest
    # checkpoint is skipped only when an older exact-resume checkpoint is complete; if
    # none is complete, fail closed instead of silently starting over.
    resume_checkpoint = select_latest_complete_checkpoint(
        training_args.output_dir,
        expected_ema_decay=(model_args.ema_decay if model_args.ema_decay > 0 else None),
        expected_world_size=int(os.environ.get("WORLD_SIZE", training_args.world_size)),
        require_deepspeed=bool(training_args.deepspeed),
    )

    # Record the visual budget alongside the checkpoint. Serving MUST feed the model the same
    # resolution it trained on -- these budgets are not recoverable from the weights, and a
    # mismatch degrades the policy silently (no shape error: fewer/more visual tokens is
    # perfectly legal). qwenvl/action_expert/inference.py reads this file back at serve time.
    if int(os.environ.get("RANK", 0)) == 0:
        atomic_json_dump({**{k: getattr(data_args, k) for k in (
            "video_max_pixels", "video_min_pixels", "max_pixels", "min_pixels",
            "wrist_max_pixels", "num_frames", "frame_stride", "image_history",
            # V3 history-video input contract: per-frame GATE-1 pre-resize budget for the
            # cam_high history clip. Meaningless (and unread) when image_history=false.
            "history_max_pixels",
            # Which wrist cameras the model conditions on -- serving a 2-wrist checkpoint
            # with 1 wrist (or vice versa) would degrade silently otherwise. The serve-side
            # budget overlay (inference.build_data_args) applies this automatically; old
            # checkpoints without the key keep the legacy right-wrist default.
            "wrist_cameras",
            # Whether the prompt carries the frame-aligned past states ("Past states:") --
            # an input-shape contract like the above: serving must build the same prompt.
            "state_history",
            # Training-time current-state dropout provenance. The raw state still defines
            # delta-action targets; masked samples render the literal State: [MASKED].
            "current_state_mask_prob",
            # Human-video-prompt conditioning: serving must prepend the demo clip with
            # the same structure and sampling (dirs non-empty <=> mode on; the dirs
            # themselves are irrelevant at serve time -- the demo arrives per request).
            "human_prompt_dirs", "human_prompt_stride", "human_prompt_max_frames",
            "human_prompt_source_fps",
            "explicit_video_timestamps",
            # Segment-level prompts + bins QA (bins_task_plan.md). Serving needs
            # human_prompt_segments only as provenance (uploads are already sub-clips
            # or full clips; the sampling rule above applies unchanged); the mix and
            # full-episode prob are training-time provenance.
            "human_prompt_segments", "human_prompt_full_episode_prob",
            "subtask_format_mix",
            # Which QA module the mix, the answers and the human-pool keys came from
            # ("bins" / "sort"). Serving must decode with the same module: the answer
            # vocabularies and the pool keys differ per task. Old checkpoints without
            # the key keep the "bins" default.
            "subtask_task",
            # Training-time provenance for the sorting task's 'where' format.
            "qa_where_absent_prob", "human_prompt_holdout_per_key",
            # Which robot labels file the run trained on (subtask_labels_4phase.json =
            # pick/place split). The serve-side overlay applies it so the server scans
            # the same labels; old checkpoints without the key keep the default.
            "robot_subtask_labels_file",
            # Training-time provenance only (Track-A language-only order samples).
            "order_sample_prob",
            # Training-time provenance for exact-root standalone pick skill records.
            # Serving does not rescan these roots; inference.build_data_args clears the
            # path allowlist while retaining the robot-QA sampling contract in the
            # checkpoint metadata for reproducibility.
            "unprompted_pick_only_dirs", "standalone_robot_qa_enabled",
            "robot_qa_stride", "robot_qa_max_frames",
            # Exact text is part of the token-level input contract. The first
            # human-prompt checkpoint predates this key (serving has a compatibility
            # fallback), but all new checkpoints must carry it explicitly.
            "subtask_question",
        )},
            # Explicit serving-mode stamp. Keep human_prompt_dirs above for old-server
            # compatibility/provenance, but serving should never need to mount or scan it.
            "human_prompt_enabled": bool(data_args.human_prompt_dirs),
            # Whether the expert attended the subtask turn in training. Attention-mask-only
            # (identical state_dict either way), so a mismatched serve loads cleanly and
            # degrades silently; the server reads this stamp and auto-applies it.
            "expert_attends_subtask": model_args.expert_attends_subtask,
            # Provenance only (training-time loss averaging; nothing at serve reads it).
            "lm_loss_per_sample": model_args.lm_loss_per_sample,
            # Max frozen-prefix length RTC training clamped to: operators must keep the
            # serve-time DELAY_STEPS at or below this (the expert never trained on longer
            # prefixes; exceeding it degrades silently).
            "rtc_prefix_max_length": model_args.rtc_prefix_max_length,
        }, os.path.join(training_args.output_dir, "visual_budget.json"))

    if "qwen3" not in model_args.model_name_or_path.lower():
        raise ValueError("The action expert currently supports Qwen3-VL models only")
    data_args.model_type = "qwen3vl"

    if data_args.use_fast_tokens and not model_args.train_vlm:
        raise ValueError(
            "--use_fast_tokens True requires --train_vlm True: FAST action tokens are a "
            "cross-entropy signal on the VLM backbone (the whole point of the KI recipe). "
            "With a frozen VLM there is nothing for them to train."
        )
    if model_args.train_vlm and not (data_args.predict_subtask or data_args.use_fast_tokens):
        raise ValueError(
            "--train_vlm True needs a VLM loss: enable --predict_subtask and/or "
            "--use_fast_tokens. Otherwise the flow-matching loss is insulated from the VLM "
            "(by design), so its trainable params get no gradient — wasting memory and "
            "potentially hanging DeepSpeed. Or set --train_vlm False."
        )
    if (getattr(data_args, "unprompted_pick_only_dirs", "").strip()
            and not model_args.expert_attends_subtask):
        raise ValueError(
            "--unprompted_pick_only_dirs requires --expert_attends_subtask True "
            "because standalone action records place the oracle 'pick <object>' in "
            "the assistant turn."
        )

    vlm = Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        dtype=(torch.bfloat16 if training_args.bf16 else None),
    )

    text_config = vlm.config.text_config
    n_expert_layers = model_args.expert_num_layers or text_config.num_hidden_layers
    if not (1 <= n_expert_layers <= text_config.num_hidden_layers):
        raise ValueError(
            f"--expert_num_layers {model_args.expert_num_layers} must be in "
            f"[1, {text_config.num_hidden_layers}] (0 = use all)"
        )
    rank0_print(f"[action-expert] expert layers: {n_expert_layers} / {text_config.num_hidden_layers} VLM layers"
                + (" (SmolVLA layer skipping)" if n_expert_layers < text_config.num_hidden_layers else ""))
    expert_config = ActionExpertConfig(
        num_hidden_layers=n_expert_layers,
        num_key_value_heads=text_config.num_key_value_heads,
        head_dim=text_config.head_dim,
        hidden_size=model_args.expert_hidden_size,
        intermediate_size=model_args.expert_intermediate_size,
        num_attention_heads=model_args.expert_num_attention_heads,
        action_dim=data_args.action_dim,
        action_horizon=data_args.action_horizon,
    )

    model = Qwen3VLWithActionExpert(
        vlm,
        expert_config,
        train_vlm=model_args.train_vlm,
        lm_loss_weight=model_args.lm_loss_weight,
        fast_loss_weight=model_args.fast_loss_weight,
        rtc_prefix_min_length=model_args.rtc_prefix_min_length,
        rtc_prefix_max_length=model_args.rtc_prefix_max_length,
        expert_attends_subtask=model_args.expert_attends_subtask,
        lm_loss_per_sample=model_args.lm_loss_per_sample,
    )
    if training_args.bf16:
        # Expert transformer in bf16 like the VLM; flow-matching heads (action/time
        # projections) stay float32, as in openpi.
        model.action_expert.to(torch.bfloat16)

    if model_args.train_vlm:
        set_vlm_trainable(model_args, vlm)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    rank0_print(
        f"Action expert params: {model.num_expert_parameters() / 1e6:.1f}M | "
        f"trainable: {trainable / 1e6:.1f}M | frozen: {frozen / 1e6:.1f}M | "
        f"train_vlm={model_args.train_vlm}"
    )
    if model_args.rtc_prefix_max_length > 0:
        rank0_print(
            f"Training-time RTC ON: prefix length d ~ Uniform[{model_args.rtc_prefix_min_length}, "
            f"{model_args.rtc_prefix_max_length}] action steps (clean prefix, postfix-only flow loss)"
        )
    if not model_args.expert_attends_subtask:
        if not data_args.predict_subtask:
            raise ValueError(
                "--expert_attends_subtask False only makes sense with --predict_subtask True "
                "(in subtask-input mode the subtask is part of the user prompt, not an "
                "assistant turn the expert could be insulated from)."
            )
        rank0_print(
            "Subtask insulation ON: the expert attends to images+state only; the subtask "
            "is supervised on the VLM but excluded from the expert's attention."
        )

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path, cache_dir=training_args.cache_dir
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    data_module = make_robot_data_module(processor, data_args)

    # Mixed subtask supervision guards: unlabeled episodes build no assistant turn, so two
    # otherwise-legal configs become hazardous with them in the mix (2026-08-02 audit).
    n_unlabeled = sum(1 for ep in data_module["train_dataset"].episodes if not ep["subtasks"])
    if n_unlabeled and data_args.predict_subtask:
        rank0_print(f"Mixed subtask supervision: {n_unlabeled}/{len(data_module['train_dataset'].episodes)} "
                    "episodes unlabeled (no language loss; FAST-only VLM signal)")
        if model_args.train_vlm and not data_args.use_fast_tokens:
            raise ValueError(
                "Mixed data with --train_vlm True needs --use_fast_tokens True: an "
                "all-unlabeled batch would otherwise give the VLM zero gradient "
                "(DeepSpeed/DDP hazard, and those samples teach the VLM nothing)."
            )
        if model_args.expert_attends_subtask:
            raise ValueError(
                "Mixed data requires --expert_attends_subtask False: unlabeled samples "
                "end in a bare generation header, and a non-insulated expert would train "
                "attending a dangling header that non-insulated serving (which always "
                "decodes a subtask) never presents. Insulate, or label the data."
            )

    # Stamp the resolved norm stats into the output dir (like visual_budget.json above):
    # serving auto-loads <ckpt>/norm_stats.json so the deployed policy always normalizes
    # with exactly the stats it trained with, independent of dataset-dir state.
    if int(os.environ.get("RANK", 0)) == 0:
        atomic_json_dump(data_module["train_dataset"].norm_stats,
                         os.path.join(training_args.output_dir, "norm_stats.json"))

    if data_args.use_fast_tokens:
        # The dataset registered FAST tokens on processor.tokenizer; grow the VLM's
        # embedding + tied lm_head to match. New rows train from scratch (train_vlm=True).
        train_dataset = data_module["train_dataset"]
        new_vocab = train_dataset.vlm_vocab_size
        model.vlm.resize_token_embeddings(new_vocab)
        tokenizer.add_tokens(
            ["<|action_start|>", "<|action_end|>"]
            + [f"<|action_{i}|>" for i in range(train_dataset.fast_vocab_size)]
        )
        if training_args.bf16:
            model.vlm.get_input_embeddings().to(torch.bfloat16)
            model.vlm.lm_head.to(torch.bfloat16)
        rank0_print(f"Resized VLM embeddings to {new_vocab} for FAST action tokens")

    # Warm-start weights (fresh optimizer/scheduler). Runs on every rank BEFORE DeepSpeed
    # wraps the model; ZeRO-2 replicates params so each rank needs the full state dict. The
    # embeddings were already resized above, so shapes match the FAST checkpoint.
    if model_args.init_from and resume_checkpoint is None:
        ckpt_bin = os.path.join(model_args.init_from, "pytorch_model.bin")
        if not os.path.isfile(ckpt_bin):
            raise FileNotFoundError(f"--init_from lacks pytorch_model.bin: {ckpt_bin}")
        rank0_print(f"Warm-starting weights from {ckpt_bin} (fresh optimizer/scheduler)")
        state_dict = torch.load(ckpt_bin, map_location="cpu", mmap=True, weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        del state_dict
        if missing or unexpected:
            rank0_print(
                f"  load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected "
                f"(missing[:3]={missing[:3]}, unexpected[:3]={unexpected[:3]})"
            )
    elif model_args.init_from:
        rank0_print(
            f"Exact-resuming {resume_checkpoint}; ignoring --init_from {model_args.init_from}"
        )

    trainer = ActionExpertTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        vlm_learning_rate=model_args.vlm_learning_rate,
        **data_module,
    )
    if model_args.ema_decay > 0:
        # Fresh/weights-only runs snapshot at train begin. Exact resumes load the
        # checkpoint EMA at that same hook, after Trainer/DeepSpeed and TrainerState.
        trainer.add_callback(
            ExpertEMACallback(
                model,
                decay=model_args.ema_decay,
                resume_checkpoint=resume_checkpoint,
            )
        )
    trainer.add_callback(FirstStepCacheFlushCallback())
    if os.environ.get("QWEN_NAN_DEBUG"):
        trainer.add_callback(EmbedNanWatchCallback(model))

    if resume_checkpoint is not None:
        logging.info("resuming exact checkpoint %s", resume_checkpoint)
        trainer.train(resume_from_checkpoint=str(resume_checkpoint))
    else:
        trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    if trainer.args.should_save:
        save_expert_only(model, training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    # QWEN_ATTN_IMPL=sdpa swaps the attention kernel (nan-debug: discriminates a
    # flash-attn varlen kernel bug from everything else). Default unchanged.
    train(attn_implementation=os.environ.get("QWEN_ATTN_IMPL", "flash_attention_2"))
