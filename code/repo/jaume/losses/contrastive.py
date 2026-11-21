import torch
import torch.nn as nn
import torch.nn.functional as F


class MultipleNegativesRankingLoss(nn.Module):
    """InfoNCE over in-batch negatives, with optional per-query hard negatives.

    For each query Qi, all passages Pj (j != i) act as negatives. When
    `hard_negative_embeddings` [B, H, D] is given, each query additionally
    sees its H mined hard negatives as candidate columns.
    Temperature and scale follow sentence-transformers defaults.
    """

    def __init__(self, scale: float = 20.0, similarity_fct=torch.nn.functional.cosine_similarity):
        super().__init__()
        self.scale = scale
        self.similarity_fct = similarity_fct

    def forward(self, query_embeddings, passage_embeddings, hard_negative_embeddings=None):
        scores = self.similarity_fct(
            query_embeddings.unsqueeze(1),
            passage_embeddings.unsqueeze(0),
            dim=-1,
        ) * self.scale
        if hard_negative_embeddings is not None:
            hard_scores = self.similarity_fct(
                query_embeddings.unsqueeze(1),   # [B, 1, D]
                hard_negative_embeddings,        # [B, H, D]
                dim=-1,
            ) * self.scale                       # [B, H]
            scores = torch.cat([scores, hard_scores], dim=1)
        labels = torch.arange(scores.size(0), device=scores.device)
        return F.cross_entropy(scores, labels)


class MatryoshkaLoss(nn.Module):
    """Wraps any contrastive loss with Matryoshka truncated-dim training."""

    def __init__(self, base_loss, dims):
        super().__init__()
        self.base_loss = base_loss
        self.dims = list(dims)

    def forward(self, query_embeddings, passage_embeddings, hard_negative_embeddings=None):
        total = self.base_loss(query_embeddings, passage_embeddings, hard_negative_embeddings)
        for d in self.dims[1:-1]:
            total = total + self.base_loss(
                query_embeddings[..., :d],
                passage_embeddings[..., :d],
                None if hard_negative_embeddings is None else hard_negative_embeddings[..., :d],
            )
        return total / len(self.dims)
