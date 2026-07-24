"""Action expert transformer for Qwen3-VL, following the pi0.5 recipe (arXiv:2504.16054).

The expert is a small Qwen3-style decoder that runs *next to* the Qwen3-VL language
model: at every layer, the expert's action tokens attend over the concatenation of

    [ VLM key/value cache at that layer  |  the expert's own keys/values ]

which is the same joint prefix-suffix attention used by openpi's
`PaliGemmaWithExpertModel`, expressed in the KV-cache formulation. For this to be
possible the expert must share three geometry parameters with the VLM's language
model: `num_hidden_layers`, `num_key_value_heads`, and `head_dim`. Everything else
(hidden width, MLP width, number of query heads) is free.

The flow-matching timestep is injected with adaptive RMSNorm (adaRMS), as in pi0.5:
each norm layer maps the timestep embedding to (scale, shift, gate); scale/shift
modulate the normalized activations and gate modulates the residual branch. The
modulation projection is zero-initialized so the expert starts as a plain
pre-norm transformer.
"""

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    apply_rotary_pos_emb,
    repeat_kv,
)


@dataclass
class ActionExpertConfig:
    # Geometry that MUST match the Qwen3-VL text config (checked at model build time).
    num_hidden_layers: int = 36
    num_key_value_heads: int = 8
    head_dim: int = 128

    # Free parameters of the expert.
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_attention_heads: int = 16
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False

    # Action interface.
    action_dim: int = 14
    action_horizon: int = 50

    # Sinusoidal timestep embedding range, as in pi0.
    time_min_period: float = 4e-3
    time_max_period: float = 4.0

    def __post_init__(self):
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")

    def to_dict(self):
        return dict(self.__dict__)


def gated_residual(x: torch.Tensor, y: torch.Tensor, gate: Optional[torch.Tensor]) -> torch.Tensor:
    if gate is None:
        return x + y
    return x + y * gate


class AdaRMSNorm(nn.Module):
    """RMSNorm with optional adaptive (scale, shift, gate) conditioning, as in pi0.5.

    Without a conditioning vector this is a standard Qwen-style RMSNorm. With one, the
    normalized activations become `norm(x) * (1 + scale) + shift` and the returned gate
    is applied to the residual branch by the caller. The conditioning projection is
    zero-initialized so that at init scale = shift = 0 and gate = 0.

    `cond` may be per-sample (B, cond_dim) -- broadcast over the sequence -- or per-token
    (B, seq, cond_dim), as used by training-time RTC where each action token carries its
    own flow-matching timestep. Zero new parameters either way.
    """

    def __init__(self, dim: int, eps: float = 1e-6, cond_dim: Optional[int] = None):
        super().__init__()
        self.eps = eps
        self.cond_dim = cond_dim
        if cond_dim is not None:
            self.dense = nn.Linear(cond_dim, dim * 3, bias=True)
            nn.init.zeros_(self.dense.weight)
            nn.init.zeros_(self.dense.bias)
            self.weight = None
        else:
            self.dense = None
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor, cond: Optional[torch.Tensor] = None):
        dtype = x.dtype
        xf = x.float()
        var = xf.pow(2).mean(-1, keepdim=True)
        normed = xf * torch.rsqrt(var + self.eps)

        if cond is None or self.dense is None:
            normed = normed * self.weight.float()
            return normed.to(dtype), None

        modulation = self.dense(cond.to(self.dense.weight.dtype))
        if x.ndim == 3 and modulation.ndim == 2:  # per-sample cond -> broadcast over seq
            modulation = modulation.unsqueeze(1)
        # per-token cond (B, seq, 3*dim) already lines up with x (B, seq, dim)
        scale, shift, gate = torch.chunk(modulation.float(), 3, dim=-1)
        normed = normed * (1.0 + scale) + shift
        return normed.to(dtype), gate.to(dtype)


class ActionExpertAttention(nn.Module):
    """Qwen3-style attention (QK-norm + M-RoPE) that also attends to a prefix KV cache.

    The prefix keys/values come straight out of the VLM's `DynamicCache`, i.e. they are
    already k_norm-ed and rotated by the VLM. The expert applies its own q/k norms and
    rotary embedding (with positions continuing after the prefix) to its own tokens, so
    relative positions between action tokens and prefix tokens are consistent.
    """

    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = AdaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = AdaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        prefix_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        hidden_shape = (bsz, seq_len, -1, self.head_dim)

        query_states, _ = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        key_states, _ = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if prefix_key_value is not None:
            prefix_keys, prefix_values = prefix_key_value
            key_states = torch.cat([prefix_keys.to(key_states.dtype), key_states], dim=2)
            value_states = torch.cat([prefix_values.to(value_states.dtype), value_states], dim=2)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,  # bool mask, True = attend
            scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.o_proj(attn_output)


class ActionExpertMLP(nn.Module):
    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ActionExpertDecoderLayer(nn.Module):
    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.self_attn = ActionExpertAttention(config)
        self.mlp = ActionExpertMLP(config)
        self.input_layernorm = AdaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=config.hidden_size)
        self.post_attention_layernorm = AdaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, cond_dim=config.hidden_size
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        prefix_key_value: Optional[tuple[torch.Tensor, torch.Tensor]],
        attention_mask: Optional[torch.Tensor],
        adarms_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states, gate = self.input_layernorm(hidden_states, adarms_cond)
        hidden_states = self.self_attn(hidden_states, position_embeddings, prefix_key_value, attention_mask)
        hidden_states = gated_residual(residual, hidden_states, gate)

        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, adarms_cond)
        hidden_states = self.mlp(hidden_states)
        return gated_residual(residual, hidden_states, gate)


class ActionExpertModel(nn.Module):
    """Stack of decoder layers; layer i attends to the VLM's layer-i KV cache."""

    def __init__(self, config: ActionExpertConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(ActionExpertDecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = AdaRMSNorm(config.hidden_size, eps=config.rms_norm_eps, cond_dim=config.hidden_size)
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        prefix_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]],
        attention_mask: Optional[torch.Tensor],
        adarms_cond: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if prefix_key_values is not None and len(prefix_key_values) != len(self.layers):
            raise ValueError(
                f"Got {len(prefix_key_values)} prefix KV layers for {len(self.layers)} expert layers; "
                "the expert must have the same number of layers as the VLM language model."
            )

        hidden_states = inputs_embeds
        for i, layer in enumerate(self.layers):
            prefix_kv = prefix_key_values[i] if prefix_key_values is not None else None
            if self.gradient_checkpointing and self.training:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    position_embeddings,
                    prefix_kv,
                    attention_mask,
                    adarms_cond,
                    use_reentrant=False,
                    preserve_rng_state=False,
                )
            else:
                hidden_states = layer(hidden_states, position_embeddings, prefix_kv, attention_mask, adarms_cond)

        hidden_states, _ = self.norm(hidden_states, adarms_cond)
        return hidden_states
