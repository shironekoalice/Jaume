"""Causal backbone for the Jaume-Causal twin.

Physically separate from `backbone.JaumeBackbone`. `RMSNorm` is imported from
the bidirectional module on purpose: the norm implementation must be
byte-identical between the twins, otherwise the attention form would not be
the only controlled variable.

Layer state-dict keys are identical to the bidirectional backbone
(`layers.<i>.self_attn.*`, `layers.<i>.mlp.*`, ...), so the two checkpoints
differ only in values and in the recorded `causal` / `model_type` config.
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .attention_causal import CausalGQA
from .backbone import RMSNorm
from .moe import MoEFeedForward, FeedForward


class CausalEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = CausalGQA(config)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        self.mlp = MoEFeedForward(config) if config.use_moe else FeedForward(config)
        self.use_moe = config.use_moe

    def forward(self, x, attention_mask=None, kv_cache=None,
                position_offset=None):
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, attention_mask, kv_cache=kv_cache,
                           position_offset=position_offset)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        if self.use_moe:
            x, aux_loss = self.mlp(x)
        else:
            x, aux_loss = self.mlp(x), None
        x = residual + x
        return x, aux_loss


class CausalBackbone(nn.Module):
    """Bidirectional-layout Transformer Encoder with a causal attention mask.

    Returns final hidden states plus the summed MoE load-balance aux loss.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([CausalEncoderLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size)
        self._gradient_checkpointing = False

    def gradient_checkpointing_enable(self):
        self._gradient_checkpointing = True

    def _run_layer(self, layer, x, attention_mask):
        if self._gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                layer, x, attention_mask, use_reentrant=False
            )
        return layer(x, attention_mask)

    def forward(self, input_ids, attention_mask=None, kv_caches=None,
                position_offset=None):
        """Forward with optional incremental KV caching.

        kv_caches: list of KVCache, one per layer (see kv_cache.make_caches).
        When given, position_offset defaults to the cache length so decoding
        continues at the correct absolute position.
        """
        hidden_states = self.embed_tokens(input_ids)

        if kv_caches is not None and position_offset is None:
            position_offset = kv_caches[0].length

        aux_total = torch.tensor(0.0, device=input_ids.device)
        for i, layer in enumerate(self.layers):
            cache = kv_caches[i] if kv_caches is not None else None
            if self._gradient_checkpointing and self.training:
                # checkpointing cannot carry a mutable cache; only used in
                # training, where incremental decoding never happens
                hidden_states, aux = self._run_layer(
                    layer, hidden_states, attention_mask)
            else:
                hidden_states, aux = layer(
                    hidden_states, attention_mask, kv_cache=cache,
                    position_offset=position_offset)
            if aux is not None:
                aux_total = aux_total + aux

        return self.norm(hidden_states), aux_total
