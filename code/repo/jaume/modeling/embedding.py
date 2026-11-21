import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import JaumeBackbone
from .backbone_causal import CausalBackbone
from .pooling import build_pooling
from .memory import build_memory_pooling

# poolers that go through the memory module (JM-S / JM-R / parameter-matched
# controls). The classic ones keep using jaume.modeling.pooling so the released
# v0.1 checkpoints load byte-identically.
MEMORY_POOLINGS = (
    "attention_mlp",
    "latent_memory",
    "recurrent_memory",
    "recurrent_memory_gate",
)


class _ClassicPooler(nn.Module):
    """Classic pooler kept as its own module so state-dict keys stay
    `pooling.query` / `pooling.*` and the released v0.1 checkpoints load
    unchanged, while still exposing the (pooled, state) signature."""

    def __init__(self, name, hidden_size):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, hidden_size)) if name == "attention" else None
        self._name = name

    def forward(self, hidden_states, attention_mask=None, memory_state=None):
        if self._name == "mean":
            if attention_mask is None:
                return hidden_states.mean(dim=1), None
            mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9), None
        if self._name == "cls":
            return hidden_states[:, 0, :], None
        # attention
        scores = (hidden_states @ self.query.transpose(-2, -1)).squeeze(-1) * (hidden_states.size(-1) ** -0.5)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1).unsqueeze(-1)
        return (hidden_states * weights).sum(dim=1), None


def _build_pooler(config):
    """Builds the arm's pooler. Every arm is wrapped so the (pooled, state)
    signature is uniform and the rest of the pipeline stays arm-agnostic."""
    name = getattr(config, "pooling", "mean")
    if name in MEMORY_POOLINGS:
        return build_memory_pooling(config)
    if name in ("mean", "cls", "attention"):
        return _ClassicPooler(name, config.hidden_size)
    from .memory import _ClassicWrapper

    return _ClassicWrapper(build_pooling(config.pooling, config.hidden_size))


def _pooler_call(pooler, hidden_states, attention_mask, memory_state=None):
    """Calls a pooler; returns (pooled [B, D], next slot state or None)."""
    out = pooler(hidden_states, attention_mask, memory_state)
    if isinstance(out, tuple):
        return out[0], out[1] if len(out) > 1 else None
    return out, None


class JaumeEmbeddingModel(nn.Module):
    """Backbone + Pooling + Projection + L2 normalize.

    forward() returns (embeddings [B, embedding_dim], aux_loss).
    Embeddings are unit-normalized so cosine similarity == dot product.

    `memory_state` (JM-R) carries the slot bank across chunks: pass the slot
    state returned by the previous chunk to continue a document, or None to
    start/reset. Single-input arms ignore it.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # The attention form is decided by the config, which is stored inside
        # every checkpoint. A causal checkpoint therefore always rebuilds a
        # causal backbone and a bidirectional one always rebuilds a
        # bidirectional backbone -- the two can never be mixed by accident.
        self.backbone = (
            CausalBackbone(config)
            if getattr(config, "causal", False)
            else JaumeBackbone(config)
        )
        self.pooling = _build_pooler(config)
        self.projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.embedding_dim, bias=False),
            nn.LayerNorm(config.embedding_dim),
        )

    def encode(self, input_ids, attention_mask=None, memory_state=None):
        """hidden states -> (pooled vector before projection, next slot state)."""
        hidden_states, aux_loss = self.backbone(input_ids, attention_mask)
        pooled, next_state = _pooler_call(self.pooling, hidden_states, attention_mask, memory_state)
        return pooled, aux_loss, next_state

    def forward(self, input_ids, attention_mask=None, memory_state=None,
                return_state=False):
        pooled, aux_loss, next_state = self.encode(input_ids, attention_mask, memory_state)
        embeddings = self.projection(pooled)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        if return_state:
            # used by the slot-diversity regulariser (train_memory.py)
            return embeddings, aux_loss, next_state
        return embeddings, aux_loss

    def forward_matryoshka(self, input_ids, attention_mask=None, memory_state=None):
        """Returns {dim: embeddings} for every configured Matryoshka dim."""
        embeddings, aux_loss = self.forward(input_ids, attention_mask, memory_state)
        out = {self.config.embedding_dim: embeddings}
        for d in self.config.matryoshka_dims:
            if d < self.config.embedding_dim:
                out[d] = F.normalize(embeddings[..., :d], p=2, dim=-1)
        return out, aux_loss
