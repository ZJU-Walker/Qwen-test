"""Train a flow-matching action expert on top of Qwen3-VL (pi0.5-style, with
knowledge insulation: expert gradients never touch the VLM weights).

Launch with scripts/train_action_expert_4b.sh.
"""

import logging
import os
import pathlib
import sys
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
    # Warm-start: load model WEIGHTS from this checkpoint dir's pytorch_model.bin, but start
    # a FRESH optimizer/scheduler/step counter (unlike auto-resume, which restores all of
    # those and requires the same GPU count for ZeRO-2). Use this to continue a run under a
    # new regime -- e.g. more GPUs / a larger batch -- from an existing checkpoint. Point
    # --output_dir at a NEW empty dir so auto-resume does not also kick in.
    init_from: str = field(default="")


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
            groups = [{"params": expert_params, "lr": self.args.learning_rate}]
            if vlm_params:
                groups.append({"params": vlm_params, "lr": self.vlm_learning_rate})
            optim_cls, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            optim_kwargs.pop("lr", None)
            self.optimizer = optim_cls(groups, **optim_kwargs)
        return self.optimizer

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        # Stash the loss components (last micro-batch of the logging interval) so
        # flow / subtask-CE / FAST-CE contributions show up separately in the logs.
        self._last_flow_loss = outputs.flow_loss.item() if outputs.flow_loss is not None else None
        self._last_lm_loss = outputs.lm_loss.item() if outputs.lm_loss is not None else None
        self._last_fast_loss = outputs.fast_loss.item() if outputs.fast_loss is not None else None
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def log(self, logs, start_time=None):
        if "loss" in logs:
            for name, attr in [("flow_loss", "_last_flow_loss"), ("lm_loss", "_last_lm_loss"),
                               ("fast_loss", "_last_fast_loss")]:
                val = getattr(self, attr, None)
                if val is not None:
                    logs[name] = round(val, 4)
        return super().log(logs, start_time)


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
    if model_args.init_from:
        ckpt_bin = os.path.join(model_args.init_from, "pytorch_model.bin")
        rank0_print(f"Warm-starting weights from {ckpt_bin} (fresh optimizer/scheduler)")
        state_dict = torch.load(ckpt_bin, map_location="cpu", mmap=True, weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        del state_dict
        if missing or unexpected:
            rank0_print(
                f"  load_state_dict: {len(missing)} missing, {len(unexpected)} unexpected "
                f"(missing[:3]={missing[:3]}, unexpected[:3]={unexpected[:3]})"
            )

    trainer = ActionExpertTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        vlm_learning_rate=model_args.vlm_learning_rate,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    if trainer.args.should_save:
        save_expert_only(model, training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
