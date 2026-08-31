"""Qwen3-VL + flow-matching action expert, following pi0.5 (arXiv:2504.16054) with
knowledge insulation (arXiv:2505.23705).

Training forward pass:
  1. The Qwen3-VL runs over the multimodal prefix (images/video + text prompt with the
     discretized robot state) exactly as it does for SFT, producing a per-layer KV cache.
     If the VLM is frozen (default) this runs under `torch.no_grad()`; if `train_vlm`
     is set, an auxiliary next-token-prediction loss is computed on `labels` (e.g.
     subtask labels), which is the knowledge-insulation co-training objective.
  2. The KV cache is **detached** before the expert consumes it, so gradients from the
     flow-matching loss can never reach the VLM weights (knowledge insulation).
  3. Noisy actions x_t = t * noise + (1 - t) * actions are projected into the expert,
     which attends to the detached prefix KV at every layer and predicts the flow
     target u_t = noise - actions. Timestep conditioning uses adaRMS (pi0.5).

Inference (`sample_actions`) runs the prefix once, then integrates the flow ODE with
`num_steps` Euler steps, re-running only the small expert each step.
"""

import contextlib
import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from qwenvl.action_expert.modeling_action_expert import ActionExpertConfig, ActionExpertModel


@dataclass
class ActionExpertOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    flow_loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    fast_loss: Optional[torch.FloatTensor] = None


# Flow-matching timestep convention in this file (note: this is the OPPOSITE of the RTC paper
# arXiv:2512.05964, which uses tau=1 for clean data). Here x_t = t*noise + (1-t)*actions, so:
_CLEAN_TIMESTEP = 0.0  # t=0 -> ground-truth actions
_NOISE_TIMESTEP = 1.0  # t=1 -> pure noise


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float
) -> torch.Tensor:
    """Sine-cosine embedding of scalar timesteps, as in pi0.

    `time` may carry arbitrary leading dims: (B,) for one timestep per sample, or (B, ah)
    for a per-action-token timestep (training-time RTC). Returns (*time.shape, dimension).
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=time.device)
    period = min_period * (max_period / min_period) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor * time[..., None].float()
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=-1)


def _rtc_prefix_mask(prefix_lengths: torch.Tensor, action_horizon: int) -> torch.Tensor:
    """(B, ah) bool, True for the first `prefix_lengths[i]` action tokens of each sample
    (the clean, already-committed action prefix)."""
    ar = torch.arange(action_horizon, device=prefix_lengths.device)
    return ar[None, :] < prefix_lengths[:, None]


def _clamp_action_prefix(x_t: torch.Tensor, action_prefix: torch.Tensor, prefix_mask: torch.Tensor) -> torch.Tensor:
    """Overwrite the masked leading action slots of `x_t` with the known action prefix."""
    return torch.where(prefix_mask[..., None], action_prefix, x_t)


class Qwen3VLWithActionExpert(nn.Module):
    # Our forward returns a MEAN-reduced loss and ignores num_items_in_batch. Without
    # this flag, HF Trainer sees **kwargs in forward, assumes a sum-reduced loss, and
    # skips dividing by gradient_accumulation_steps — inflating loss and gradients by
    # the accumulation factor (same guard as Qwen3VLForConditionalGeneration, #37208).
    accepts_loss_kwargs = False

    def __init__(
        self,
        vlm,
        expert_config: ActionExpertConfig,
        train_vlm: bool = False,
        lm_loss_weight: float = 1.0,
        fast_loss_weight: float = 1.0,
        # Training-time RTC (arXiv:2512.05964): range of action-prefix lengths (simulated
        # inference delays, in action steps) to sample per training example. RTC is active
        # iff rtc_prefix_max_length > 0. Requires 0 <= min <= max < action_horizon.
        rtc_prefix_min_length: int = 0,
        rtc_prefix_max_length: int = 0,
        # Subtask insulation: when False, the expert's attention EXCLUDES the assistant
        # subtask turn (marked by subtask_token_mask) -- it conditions on images+state only,
        # and the subtask remains a pure VLM co-training signal. Lets inference run the
        # expert without waiting for subtask generation. Must match between train and serve.
        expert_attends_subtask: bool = True,
        # Average the subtask-text CE per SAMPLE (mean of per-token means) instead of
        # pooling all text tokens. With answer formats of very different lengths in one
        # batch (a 3-token bin word vs a full task sentence) the pooled mean weights a
        # sample by its answer length; per-sample weights each question equally.
        # Training-time only (no labels at serve).
        lm_loss_per_sample: bool = False,
    ):
        super().__init__()
        text_config = vlm.config.text_config
        mismatches = []
        # Expert layer i attends to VLM layer i's KV, so the expert may have FEWER layers
        # than the VLM (SmolVLA-style layer skipping: N <= L, attends to VLM layers 0..N-1);
        # it must never have MORE (there would be no VLM KV for the extra layers).
        if expert_config.num_hidden_layers > text_config.num_hidden_layers:
            mismatches.append(
                f"num_hidden_layers: expert {expert_config.num_hidden_layers} exceeds vlm "
                f"{text_config.num_hidden_layers} (expert attends VLM layers 0..N-1, so N <= L)"
            )
        if expert_config.num_key_value_heads != text_config.num_key_value_heads:
            mismatches.append(
                f"num_key_value_heads: expert {expert_config.num_key_value_heads} vs vlm {text_config.num_key_value_heads}"
            )
        if expert_config.head_dim != text_config.head_dim:
            mismatches.append(f"head_dim: expert {expert_config.head_dim} vs vlm {text_config.head_dim}")
        if mismatches:
            raise ValueError(
                "Action expert geometry must match the VLM language model for joint attention: "
                + "; ".join(mismatches)
            )

        if not (0 <= rtc_prefix_min_length <= rtc_prefix_max_length < expert_config.action_horizon):
            raise ValueError(
                f"RTC prefix lengths must satisfy 0 <= min ({rtc_prefix_min_length}) <= "
                f"max ({rtc_prefix_max_length}) < action_horizon ({expert_config.action_horizon})"
            )

        self.vlm = vlm
        self.expert_config = expert_config
        self.action_expert = ActionExpertModel(expert_config)
        self.train_vlm = train_vlm
        self.lm_loss_weight = lm_loss_weight
        self.fast_loss_weight = fast_loss_weight
        self.rtc_prefix_min_length = rtc_prefix_min_length
        self.rtc_prefix_max_length = rtc_prefix_max_length
        self.expert_attends_subtask = expert_attends_subtask
        self.lm_loss_per_sample = lm_loss_per_sample

        width = expert_config.hidden_size
        # Flow-matching interface, kept in float32 like openpi.
        self.action_in_proj = nn.Linear(expert_config.action_dim, width)
        self.action_out_proj = nn.Linear(width, expert_config.action_dim)
        self.time_mlp_in = nn.Linear(width, width)
        self.time_mlp_out = nn.Linear(width, width)

        if not train_vlm:
            self.vlm.requires_grad_(False)

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------
    def train(self, mode: bool = True):
        super().train(mode)
        if not self.train_vlm:
            # Keep the frozen VLM in eval mode so its behavior matches inference.
            self.vlm.eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        # Only the expert is checkpointed. The VLM must NOT use HF gradient
        # checkpointing here because GradientCheckpointingLayer drops the KV cache
        # (use_cache is forced off), which the expert needs. The frozen VLM runs
        # under no_grad, so it does not store activations anyway.
        self.action_expert.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        self.action_expert.gradient_checkpointing = False

    @property
    def language_model(self):
        return self.vlm.model.language_model

    def num_expert_parameters(self) -> int:
        heads = [self.action_in_proj, self.action_out_proj, self.time_mlp_in, self.time_mlp_out]
        return sum(p.numel() for p in self.action_expert.parameters()) + sum(
            p.numel() for m in heads for p in m.parameters()
        )

    # ------------------------------------------------------------------
    # Flow matching pieces
    # ------------------------------------------------------------------
    def sample_noise(self, shape, device):
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device)

    def sample_time(self, bsize, device):
        # Beta(1.5, 1) emphasis on high noise levels, as in pi0.
        beta = torch.distributions.Beta(1.5, 1.0).sample((bsize,)).to(device)
        return beta * 0.999 + 0.001

    def embed_suffix(self, noisy_actions: torch.Tensor, timestep: torch.Tensor):
        """Project noisy actions to expert tokens and timestep to the adaRMS condition.

        `timestep` is (B,) for vanilla flow matching or (B, ah) per-action-token for RTC;
        the adaRMS condition comes out (B, width) or (B, ah, width) accordingly (AdaRMSNorm
        handles both)."""
        action_emb = self.action_in_proj(noisy_actions.to(self.action_in_proj.weight.dtype))

        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.expert_config.hidden_size,
            min_period=self.expert_config.time_min_period,
            max_period=self.expert_config.time_max_period,
        ).to(self.time_mlp_in.weight.dtype)
        x = F.silu(self.time_mlp_in(time_emb))
        adarms_cond = F.silu(self.time_mlp_out(x))
        return action_emb, adarms_cond

    # ------------------------------------------------------------------
    # Prefix (VLM) forward
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _vlm_truncated_to_expert_depth(self):
        """Temporarily run the VLM language model for only its first N layers, where N is
        the expert depth (SmolVLA layer skipping). Valid ONLY when no VLM LM output is
        needed: the expert reads VLM KV layers 0..N-1, so layers N..L-1 cannot affect it
        (the final norm / lm_head on the truncated stack are meaningless and unused). This
        is the serve-time early-exit -- it halves the LM prefill when N = L/2. No-op when
        the expert already uses all layers. DeepStack visual injection only touches the
        first few layers (<< N), so it is unaffected.
        """
        lm = self.vlm.model.language_model
        n = self.expert_config.num_hidden_layers
        if n >= len(lm.layers):
            yield
            return
        full = lm.layers
        try:
            lm.layers = full[:n]  # nn.ModuleList slice -> same layer objects, shorter loop
            yield
        finally:
            lm.layers = full

    def _detach_prefix_kv(self, cache) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Pull per-layer (keys, values) out of a DynamicCache and stop gradients,
        sliced to the expert's layer count.

        With SmolVLA-style layer skipping (arXiv:2506.01844) the expert has N <= L
        layers and expert layer i attends to VLM layer i, so only the VLM's first N
        layers matter for the action prediction. The VLM still runs all L layers here
        (its upper layers are trained by the LM/FAST co-training losses and produce the
        subtask logits); at serve time, when no subtask is decoded, the server can
        early-exit the VLM at layer N for the latency win. Slicing to N is the single
        chokepoint every prefix->expert path passes through.

        This detach is the knowledge-insulation boundary: everything the action expert
        sees from the VLM passes through here.
        """
        n = self.expert_config.num_hidden_layers
        if hasattr(cache, "layers"):  # transformers >= 4.56
            return [(layer.keys.detach(), layer.values.detach()) for layer in cache.layers[:n]]
        return [(k.detach(), v.detach()) for k, v in zip(cache.key_cache[:n], cache.value_cache[:n])]

    def _prefix_forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        pixel_values,
        image_grid_thw,
        pixel_values_videos,
        video_grid_thw,
        labels=None,
        fast_token_mask=None,
    ):
        vlm_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            use_cache=True,
        )

        lm_loss = None
        fast_loss = None
        # We need the VLM's LM head (all L layers) only to compute the co-training losses
        # during training. Every OTHER call -- all of inference -- uses the prefix purely to
        # feed the expert (subtask decoding runs its own generate_subtask* path, never this
        # one), so we early-exit the VLM at the expert depth N. No-op when N == L.
        need_lm_head = self.train_vlm and labels is not None and (labels != -100).any()
        trunc = contextlib.nullcontext() if need_lm_head else self._vlm_truncated_to_expert_depth()
        if self.train_vlm:
            with trunc:
                outputs = self.vlm.model(**vlm_kwargs)
            if need_lm_head:
                lm_loss, fast_loss = self._language_losses(
                    self.vlm.lm_head(outputs.last_hidden_state), labels, fast_token_mask,
                    lm_per_sample=self.lm_loss_per_sample,
                )
        else:
            with torch.no_grad(), trunc:
                outputs = self.vlm.model(**vlm_kwargs)

        prefix_key_values = self._detach_prefix_kv(outputs.past_key_values)
        return prefix_key_values, lm_loss, fast_loss

    @staticmethod
    def _language_losses(logits, labels, fast_token_mask, lm_per_sample=False):
        """Next-token CE, split into the text/subtask term and the FAST action-token term
        (both are the KI cross-entropy objective; splitting is only for separate logging).
        Standard shift: logits at position i predict the token at i+1.

        lm_per_sample=True averages the text term per sample first (mean over each
        sample's text tokens, then mean over samples that have any) so each QUESTION
        contributes equally regardless of its answer length. FAST stays pooled: chunk
        lengths are comparable across samples, so pooling is already balanced there."""
        batch = labels.shape[0]
        shift_logits = logits[:, :-1].float().reshape(-1, logits.shape[-1])
        shift_labels = labels[:, 1:].reshape(-1)
        per_tok = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="none")
        valid = shift_labels != -100
        if fast_token_mask is not None:
            is_fast = fast_token_mask[:, 1:].reshape(-1)
        else:
            is_fast = torch.zeros_like(valid)
        text_sel = valid & ~is_fast
        fast_sel = valid & is_fast
        if not text_sel.any():
            lm_loss = None
        elif lm_per_sample:
            tok = per_tok.view(batch, -1)
            sel = text_sel.view(batch, -1).float()
            counts = sel.sum(dim=1)
            rows = counts > 0
            lm_loss = ((tok * sel).sum(dim=1)[rows] / counts[rows]).mean()
        else:
            lm_loss = per_tok[text_sel].mean()
        fast_loss = per_tok[fast_sel].mean() if fast_sel.any() else None
        return lm_loss, fast_loss

    # ------------------------------------------------------------------
    # torch.compile of the expert Euler step (Phase 4 latency work)
    # ------------------------------------------------------------------
    def enable_expert_compile(self, bucket: int = 64, mode: str = "reduce-overhead"):
        """Compile `_expert_forward` -- the function called once per Euler step, whose
        ~1000 tiny kernel launches per call dominate the flow loop (profiled ~27 ms/step
        with the GPU largely idle awaiting CPU dispatch). `reduce-overhead` records the
        kernels into a CUDA graph and replays them with a single dispatch.

        Compiled graphs are shape-specialized (`dynamic=False`), and the prompt length
        jitters a few tokens between requests (the discretized state string changes
        width) -- so `sample_actions` pads the prefix KV/masks up to a multiple of
        `bucket` tokens (see _pad_prefix_to_bucket). Every request then presents
        identical shapes and there are NO mid-episode recompile stalls. Callers should
        warm up (2+ calls) before serving and verify against eager output."""
        self._compile_bucket = bucket
        # mode=None -> bucket padding WITHOUT compilation (an eager-but-padded reference:
        # lets callers separate padding numerics -- bf16 SDPA tiling changes with length --
        # from compile numerics, and gate strictly on the latter).
        self._compiled_expert = self._expert_forward if mode is None else torch.compile(
            self._expert_forward, mode=mode, fullgraph=False, dynamic=False
        )

    def disable_expert_compile(self):
        self._compiled_expert = None

    def _pad_prefix_to_bucket(self, prefix_key_values, attention_mask, position_ids,
                              subtask_token_mask):
        """Right-pad the prefix (KV + masks + positions) to a multiple of the compile
        bucket. Padded slots carry attention_mask=0, so `_expert_forward`'s obs_mask
        excludes them from the softmax -- numerically this only perturbs bf16 SDPA tiling
        (~5e-3, same class as the validated single-prefill/insulation deltas)."""
        S = prefix_key_values[0][0].shape[2]
        bucket = self._compile_bucket
        pad = (bucket - S % bucket) % bucket
        if pad == 0:
            return prefix_key_values, attention_mask, position_ids, subtask_token_mask
        prefix_key_values = [
            (F.pad(k, (0, 0, 0, pad)), F.pad(v, (0, 0, 0, pad))) for k, v in prefix_key_values
        ]
        attention_mask = F.pad(attention_mask, (0, pad), value=0)
        position_ids = F.pad(position_ids, (0, pad), value=0)
        if subtask_token_mask is not None:
            subtask_token_mask = F.pad(subtask_token_mask, (0, pad), value=False)
        return prefix_key_values, attention_mask, position_ids, subtask_token_mask

    # ------------------------------------------------------------------
    # Suffix (expert) helpers
    # ------------------------------------------------------------------
    def _suffix_position_embeddings(self, obs_positions, suffix_len, batch_size, device, dtype):
        """Rope cos/sin for action tokens at positions continuing after the OBSERVATION
        prefix (excluding any FAST tokens, so obs->expert relative positions match at
        train/inference where FAST is absent).

        Text tokens in Qwen3-VL have t == h == w positions, so giving action tokens
        identical coordinates on all three M-RoPE axes reduces to standard 1D rope,
        consistent with how the prefix keys were rotated.
        """
        next_pos = obs_positions.amax(dim=1) + 1  # (B,) max obs position per sample
        suffix_pos = next_pos[:, None] + torch.arange(suffix_len, device=device)[None, :]
        suffix_pos = suffix_pos[None].expand(3, batch_size, suffix_len)
        dummy = torch.zeros(1, dtype=dtype, device=device)
        return self.language_model.rotary_emb(dummy, suffix_pos)

    @staticmethod
    def _suffix_attention_mask(obs_mask, suffix_len):
        """Bool mask (B, 1, suffix, prefix + suffix): the action chunk attends to every
        valid OBSERVATION prefix token (images/state/text/subtask) and bidirectionally to
        itself (pi0 block attention). FAST action-token positions are excluded from
        obs_mask so the expert cannot read them (anti-shortcut, per KI eq. attention rule)."""
        bsz, prefix_len = obs_mask.shape
        prefix_part = obs_mask[:, None, None, :].bool().expand(bsz, 1, suffix_len, prefix_len)
        suffix_part = torch.ones(
            bsz, 1, suffix_len, suffix_len, dtype=torch.bool, device=obs_mask.device
        )
        return torch.cat([prefix_part, suffix_part], dim=-1)

    def _expert_forward(self, x_t, timestep, prefix_key_values, attention_mask, position_ids,
                        fast_token_mask=None, subtask_token_mask=None):
        """One expert pass: noisy actions + timestep -> predicted flow v_t. The expert
        attends to observation tokens only (valid & not FAST; with subtask insulation, also
        not the assistant subtask turn -- images+state alone)."""
        bsz, suffix_len = x_t.shape[:2]
        action_emb, adarms_cond = self.embed_suffix(x_t, timestep)

        # Observation tokens = valid (non-pad) AND not FAST action tokens.
        obs_mask = attention_mask.bool()
        if fast_token_mask is not None:
            obs_mask = obs_mask & ~fast_token_mask.bool()
        if not self.expert_attends_subtask:
            if subtask_token_mask is None:
                raise ValueError(
                    "expert_attends_subtask=False requires subtask_token_mask (pass an "
                    "all-False mask if the sequence has no assistant turn)"
                )
            obs_mask = obs_mask & ~subtask_token_mask.bool()

        # Positions: action tokens continue after the max OBSERVATION position (row 0 of
        # the 3-axis M-RoPE ids; text/obs tokens share t==h==w).
        if position_ids.ndim == 3:  # (3, B, S)
            text_positions = position_ids[0]
        else:  # (B, S)
            text_positions = position_ids
        obs_positions = text_positions.masked_fill(~obs_mask, -1)  # ignore non-obs in the max

        expert_dtype = self.action_expert.layers[0].self_attn.q_proj.weight.dtype
        position_embeddings = self._suffix_position_embeddings(
            obs_positions, suffix_len, bsz, x_t.device, expert_dtype
        )
        attn_mask = self._suffix_attention_mask(obs_mask, suffix_len)

        suffix_out = self.action_expert(
            action_emb.to(expert_dtype),
            position_embeddings,
            prefix_key_values,
            attn_mask,
            adarms_cond,
        )
        return self.action_out_proj(suffix_out.to(self.action_out_proj.weight.dtype)).float()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.LongTensor,
        actions: torch.FloatTensor,  # (B, action_horizon, action_dim), normalized
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        fast_token_mask: Optional[torch.Tensor] = None,
        subtask_token_mask: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        time: Optional[torch.Tensor] = None,
        action_loss_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> ActionExpertOutput:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if position_ids is None:
            position_ids, _ = self.vlm.model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask=attention_mask
            )

        prefix_key_values, lm_loss, fast_loss = self._prefix_forward(
            input_ids,
            attention_mask,
            position_ids,
            pixel_values,
            image_grid_thw,
            pixel_values_videos,
            video_grid_thw,
            labels=labels,
            fast_token_mask=fast_token_mask,
        )

        actions = actions.float()
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Training-time RTC (paper change #2): treat the first `d` action tokens as an
        # already-known, clean action prefix. Overwrite their noisy x_t with the ground-truth
        # actions and pin their flow-matching timestep to CLEAN (t=0 in this convention), so
        # the model learns to denoise the postfix conditioned on the prefix. `d` is sampled
        # per-sample to simulate a range of inference delays.
        suffix_timestep = time
        rtc_mask = None
        if self.rtc_prefix_max_length > 0:
            prefix_lengths = torch.randint(
                self.rtc_prefix_min_length, self.rtc_prefix_max_length + 1,
                (actions.shape[0],), device=actions.device,
            )
            rtc_mask = _rtc_prefix_mask(prefix_lengths, actions.shape[1])  # (B, ah)
            x_t = torch.where(rtc_mask[..., None], actions, x_t)
            suffix_timestep = torch.where(rtc_mask, _CLEAN_TIMESTEP, time[:, None])  # (B, ah)

        v_t = self._expert_forward(
            x_t, suffix_timestep, prefix_key_values, attention_mask, position_ids,
            fast_token_mask, subtask_token_mask
        )

        if rtc_mask is None:
            per_sample = ((u_t - v_t) ** 2).mean(dim=(1, 2))  # (B,)
        else:
            # Paper change #3: only supervise the postfix. Zero the prefix tokens' loss and
            # reweight each sample's postfix by ah/postfix_count, so the overall mean equals
            # the mean over each sample's postfix averaged over the batch -- same scale as
            # vanilla training, every sample weighted equally regardless of its sampled d.
            per_token = ((u_t - v_t) ** 2).mean(-1)  # (B, ah)
            postfix_mask = ~rtc_mask
            postfix_count = postfix_mask.sum(-1, keepdim=True).clamp(min=1)
            per_token = torch.where(
                postfix_mask, per_token * (actions.shape[1] / postfix_count), torch.zeros_like(per_token)
            )
            per_sample = per_token.mean(-1)  # (B,)
        if action_loss_mask is None:
            flow_loss = per_sample.mean()
        else:
            # Language-only samples (order samples, robot_data.order_sample_prob): their
            # action chunk belongs to a DIFFERENT conditioning (the natural pairing), so
            # the flow loss is masked per sample; their FAST labels arrive as -100 (no
            # dataset change needed here) and only the subtask CE supervises them.
            m = action_loss_mask.float()
            flow_loss = (per_sample * m).sum() / m.sum().clamp(min=1.0)
        # KI objective: flow matching (alpha=1) + subtask CE + FAST action CE. The expert's
        # flow gradient is insulated from the VLM (detached KV); the VLM learns actions
        # only through the FAST/subtask cross-entropy terms.
        loss = flow_loss
        if lm_loss is not None:
            loss = loss + self.lm_loss_weight * lm_loss
        if fast_loss is not None:
            loss = loss + self.fast_loss_weight * fast_loss
        return ActionExpertOutput(
            loss=loss,
            flow_loss=flow_loss.detach(),
            lm_loss=lm_loss.detach() if lm_loss is not None else None,
            fast_loss=fast_loss.detach() if fast_loss is not None else None,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        pixel_values_videos: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        noise: Optional[torch.Tensor] = None,
        num_steps: int = 10,
        # RTC inference: a known action prefix (the previously-committed actions overlapping
        # this chunk, in NORMALIZED model action space) and how many of its leading entries
        # are valid. When provided, those action slots are hard-clamped to the prefix
        # throughout denoising and their timestep is pinned to CLEAN. `action_prefix` may be
        # (B, p, ad) with p <= action_horizon (padded internally); `prefix_length` defaults
        # to the prefix's length. Backward compatible: absent -> identical to before.
        action_prefix: Optional[torch.Tensor] = None,
        prefix_length: Optional[int] = None,
        # Single-prefill fast path: a KV cache (DynamicCache) already covering input_ids --
        # e.g. kept from subtask generation -- so the VLM prefix pass is skipped entirely.
        # input_ids/position_ids must correspond to the SAME token sequence the cache covers.
        prefix_key_values=None,
        # Subtask insulation (expert_attends_subtask=False checkpoints): bool mask over
        # input_ids marking the assistant subtask turn the expert must NOT attend to.
        subtask_token_mask: Optional[torch.Tensor] = None,
        # Profiling: pass a dict to receive {"expert_prefill_ms", "flow_loop_ms",
        # "flow_steps"}. Adds cuda synchronizations -- leave None in production.
        timings: Optional[dict] = None,
    ) -> torch.Tensor:
        """Integrate the flow ODE from noise (t=1) to actions (t=0) with Euler steps.

        Returns normalized actions of shape (B, action_horizon, action_dim); callers
        must unnormalize (and convert deltas to absolute joints) with the dataset stats.
        """
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if position_ids is None:
            position_ids, _ = self.vlm.model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask=attention_mask
            )

        import time as _time

        if timings is not None and input_ids.is_cuda:
            torch.cuda.synchronize(input_ids.device)
        _t0 = _time.perf_counter()

        if prefix_key_values is None:
            prefix_key_values, _, _ = self._prefix_forward(
                input_ids,
                attention_mask,
                position_ids,
                pixel_values,
                image_grid_thw,
                pixel_values_videos,
                video_grid_thw,
            )
        else:
            # reuse a cache kept from subtask generation (single-prefill fast path); the
            # detach is a no-op under inference_mode but keeps the KI boundary explicit
            prefix_key_values = self._detach_prefix_kv(prefix_key_values)
            if prefix_key_values[0][0].shape[2] != input_ids.shape[1]:
                raise ValueError(
                    f"prefix_key_values covers {prefix_key_values[0][0].shape[2]} tokens but "
                    f"input_ids has {input_ids.shape[1]} -- they must describe the same sequence"
                )

        if timings is not None:
            if input_ids.is_cuda:
                torch.cuda.synchronize(input_ids.device)
            timings["expert_prefill_ms"] = round((_time.perf_counter() - _t0) * 1e3, 1)
            _t0 = _time.perf_counter()

        bsz = input_ids.shape[0]
        device = input_ids.device
        horizon = self.expert_config.action_horizon
        if noise is None:
            noise = self.sample_noise((bsz, horizon, self.expert_config.action_dim), device)

        # RTC setup: pad the prefix to the full horizon so it lines up slot-for-slot with the
        # action chunk, build the clamp mask, and seed x_t's prefix slots with the clean prefix.
        rtc_mask = None
        if action_prefix is not None:
            action_prefix = action_prefix.to(device=device, dtype=noise.dtype)
            if action_prefix.ndim == 2:
                action_prefix = action_prefix[None]
            if prefix_length is None:
                prefix_length = action_prefix.shape[-2]
            prefix_length = int(min(max(prefix_length, 0), horizon))
            action_prefix = action_prefix[:, :horizon]
            if action_prefix.shape[-2] < horizon:
                pad = noise.new_zeros(bsz, horizon - action_prefix.shape[-2], action_prefix.shape[-1])
                action_prefix = torch.cat([action_prefix, pad], dim=-2)
            lengths = torch.full((bsz,), prefix_length, dtype=torch.long, device=device)
            rtc_mask = _rtc_prefix_mask(lengths, horizon)  # (B, ah)
            noise = _clamp_action_prefix(noise, action_prefix, rtc_mask)

        # No FAST tokens at inference; the expert attends to all valid prefix tokens.
        # Compiled path: pad the prefix to the compile bucket so every request presents
        # the same shapes (no recompiles), then run the compiled per-step forward.
        expert_fn = getattr(self, "_compiled_expert", None)
        if expert_fn is not None:
            prefix_key_values, attention_mask, position_ids, subtask_token_mask = \
                self._pad_prefix_to_bucket(prefix_key_values, attention_mask,
                                           position_ids, subtask_token_mask)
        else:
            expert_fn = self._expert_forward

        dt = -1.0 / num_steps
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            if rtc_mask is None:
                timestep = time.expand(bsz)
            else:
                # pin prefix tokens to CLEAN, postfix tokens to the current denoising time
                timestep = torch.where(rtc_mask, _CLEAN_TIMESTEP, time)  # (B, ah)
            v_t = expert_fn(
                x_t, timestep, prefix_key_values, attention_mask, position_ids,
                None, subtask_token_mask
            )
            x_t = x_t + dt * v_t
            if rtc_mask is not None:
                # re-clamp the prefix slots after the Euler step (hard inpainting, no VJP)
                x_t = _clamp_action_prefix(x_t, action_prefix, rtc_mask)
            time = time + dt

        if timings is not None:
            if input_ids.is_cuda:
                torch.cuda.synchronize(input_ids.device)
            timings["flow_loop_ms"] = round((_time.perf_counter() - _t0) * 1e3, 1)
            timings["flow_steps"] = num_steps
        return x_t
