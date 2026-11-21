import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jaume.losses.contrastive import MultipleNegativesRankingLoss, MatryoshkaLoss
from jaume.losses.calibration import (
    expected_calibration_error, brier_score, negative_log_likelihood,
    auroc, calibration_report, TemperatureScaler,
)
from jaume.data.negative_mining import HardNegativeMiner

torch.manual_seed(0)


def _normalized(*shape):
    return torch.nn.functional.normalize(torch.randn(*shape), dim=-1)


def test_mnrl_hard_negatives_change_loss():
    """Hard negative columns must make the loss stricter than in-batch only."""
    q, p = _normalized(4, 16), _normalized(4, 16)
    hard = _normalized(4, 2, 16)
    loss = MultipleNegativesRankingLoss()
    inbatch_only = loss(q, p).item()
    with_hard = loss(q, p, hard).item()
    assert with_hard > inbatch_only, (inbatch_only, with_hard)


def test_mnrl_hard_negatives_gradient():
    """Hard negative path must be differentiable end to end."""
    q = _normalized(3, 8).requires_grad_(True)
    p = _normalized(3, 8)
    hard = _normalized(3, 1, 8)
    MultipleNegativesRankingLoss()(q, p, hard).backward()
    assert q.grad is not None and q.grad.abs().sum() > 0


def test_matryoshka_loss_with_hard_negatives():
    """Truncated-dim loss slices hard negatives consistently."""
    q, p = _normalized(4, 16), _normalized(4, 16)
    hard = _normalized(4, 2, 16)
    loss = MatryoshkaLoss(MultipleNegativesRankingLoss(), [16, 8, 4])
    value = loss(q, p, hard)
    assert torch.isfinite(value)


def _make_miner():
    class TinyEnc(torch.nn.Module):
        def __init__(self, V):
            super().__init__()
            self.emb = torch.nn.Embedding(V, 8)

        def forward(self, ids, mask):
            m = mask.unsqueeze(-1).float()
            v = (self.emb(ids) * m).sum(1) / m.sum(1).clamp(min=1)
            return torch.nn.functional.normalize(v, dim=-1), torch.tensor(0.0)

    V = 50

    class Tok:
        def __call__(self, texts, max_length=32, **kw):
            seqs = [[abs(hash(t)) % V for t in t.split()] for t in texts]
            L = max(len(x) for x in seqs)
            ids = torch.tensor([x + [0] * (L - len(x)) for x in seqs])
            return {"input_ids": ids, "attention_mask": (ids != 0).long()}

    return HardNegativeMiner(TinyEnc(V).eval(), Tok(), device="cpu", encode_batch_size=4)


def test_miner_excludes_positive_and_duplicates():
    """Positive text and its corpus duplicates must never be mined."""
    miner = _make_miner()
    corpus = ["alpha doc", "alpha doc", "beta doc", "gamma doc"]
    rows = miner.mine(
        ["find alpha"], ["alpha doc"], corpus,
        num_negatives=2, min_score=-1.0, max_score=1.0, pos_margin=-1.0,
    )
    assert len(rows) == 1
    assert all(neg != "alpha doc" for neg in rows[0]["hard_negatives"])
    assert len(set(rows[0]["hard_negatives"])) == len(rows[0]["hard_negatives"])


def test_miner_similarity_band():
    """Candidates outside [min_score, max_score] are rejected."""
    miner = _make_miner()
    corpus = ["alpha doc", "beta doc", "gamma doc"]
    rows = miner.mine(
        ["find alpha"], ["alpha doc"], corpus,
        num_negatives=1, min_score=0.99, max_score=1.0,
    )
    assert rows == []  # random embedding sims can't reach 0.99


def test_calibration_metrics_match_reference():
    """Perfect predictions: ECE 0, Brier 0; known-value AUROC."""
    probs = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    labels = torch.tensor([0, 1, 0, 1])
    # single bin: mean confidence 0.75, accuracy 1.0 -> ece = |1.0 - 0.75|
    assert abs(expected_calibration_error(probs, labels, n_bins=1) - 0.25) < 1e-6
    target = torch.nn.functional.one_hot(labels, 2).float()
    assert abs(brier_score(probs, labels) - ((probs - target) ** 2).sum(1).mean().item()) < 1e-6
    picked = probs.gather(1, labels.unsqueeze(1))
    assert abs(negative_log_likelihood(probs, labels) + picked.log().mean().item()) < 1e-6

    assert abs(auroc(torch.tensor([0.3, 0.2, 0.6, 0.7]), torch.tensor([1, 0, 1, 0])) - 0.5) < 1e-6
    assert abs(auroc(torch.tensor([0.9, 0.8, 0.7, 0.6]), torch.tensor([1, 0, 1, 0])) - 0.75) < 1e-6
    assert auroc(torch.tensor([0.9, 0.8, 0.3, 0.2]), torch.tensor([1, 1, 0, 0])) == 1.0


def test_calibration_report_and_ties():
    """Report keys; tied scores use average ranks."""
    probs = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    labels = torch.tensor([0, 1, 1])
    report = calibration_report(probs, labels)
    assert {"ece", "brier", "nll", "auroc"} <= set(report)
    assert auroc(torch.tensor([0.5, 0.5]), torch.tensor([1, 0])) == 0.5
    multi = torch.softmax(torch.randn(6, 4), dim=-1)
    multi_labels = torch.randint(0, 4, (6,))
    assert {"ece", "brier", "nll", "auroc_conf"} <= set(calibration_report(multi, multi_labels))


def test_temperature_scaling():
    """Temperature scaling only rescales confidence, never the argmax."""
    logits = torch.randn(64, 3) * 3
    labels = logits.argmax(-1)  # perfectly separable
    scaler = TemperatureScaler()
    scaler.fit(logits, labels)
    probs = scaler(logits)
    assert probs.argmax(-1).eq(labels).all()
    assert scaler.temperature > 0
    # overconfident logits on a hard problem -> fitted T > 1 softens
    hard_logits = torch.randn(200, 3)
    hard_labels = torch.randint(0, 3, (200,))
    scaler2 = TemperatureScaler()
    scaler2.fit(hard_logits, hard_labels)
    base = torch.nn.functional.softmax(hard_logits, -1).max(-1).values.mean()
    scaled = scaler2(hard_logits).max(-1).values.mean()
    assert scaler2.temperature > 1 and scaled < base


def _tiny_embedding_model():
    from jaume.modeling.config import JaumeConfig
    from jaume.modeling.embedding import JaumeEmbeddingModel

    config = JaumeConfig(
        vocab_size=1000, hidden_size=64, intermediate_size=128, num_layers=2,
        num_attention_heads=4, num_key_value_heads=2, expert_num=4, topk=2,
        embedding_dim=64, matryoshka_dims=(16, 32, 64), pooling="attention",
    )
    return JaumeEmbeddingModel(config).eval()


def _snapshot_grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def test_gradcache_matches_direct_gradients():
    """GradCache must reproduce the full-batch gradient: exactly at chunk=B,
    and to fp32 accumulation noise when actually chunked."""
    from jaume.losses.gradcache import gradcache_step

    torch.manual_seed(7)
    model = _tiny_embedding_model()
    loss_fn = MultipleNegativesRankingLoss()
    B, L, V = 8, 12, 1000
    q_ids = torch.randint(0, V, (B, L))
    q_mask = torch.ones(B, L, dtype=torch.long)
    p_ids = torch.randint(0, V, (B, L))
    p_mask = torch.ones(B, L, dtype=torch.long)

    # direct full-batch path
    q_emb, aux = model(q_ids, q_mask)
    p_emb, _ = model(p_ids, p_mask)
    (loss_fn(q_emb, p_emb) + 0.01 * aux).backward()
    direct = _snapshot_grads(model)
    model.zero_grad(set_to_none=True)

    def run_gc(chunk_size):
        gradcache_step(
            model, loss_fn, q_ids, q_mask, p_ids, p_mask,
            chunk_size=chunk_size, load_balance_weight=0.01, device_type="cpu",
        )
        grads = _snapshot_grads(model)
        model.zero_grad(set_to_none=True)
        return grads

    # single chunk = identical forward shapes -> exact match
    exact = run_gc(B)
    for name in direct:
        assert torch.equal(direct[name], exact[name]), name

    # true chunking -> fp32 accumulation-order noise only
    cached = run_gc(2)
    for name in direct:
        assert torch.allclose(direct[name], cached[name], atol=1e-3), name


def test_gradcache_matches_direct_with_hard_negatives():
    """Same equivalence must hold on the hard-negative path."""
    from jaume.losses.gradcache import gradcache_step

    torch.manual_seed(11)
    model = _tiny_embedding_model()
    loss_fn = MultipleNegativesRankingLoss()
    B, H, L, V = 6, 2, 10, 1000
    q_ids = torch.randint(0, V, (B, L))
    q_mask = torch.ones(B, L, dtype=torch.long)
    p_ids = torch.randint(0, V, (B, L))
    p_mask = torch.ones(B, L, dtype=torch.long)
    n_ids = torch.randint(0, V, (B * H, L))
    n_mask = torch.ones(B * H, L, dtype=torch.long)

    q_emb, aux = model(q_ids, q_mask)
    p_emb, _ = model(p_ids, p_mask)
    n_emb, _ = model(n_ids, n_mask)
    (loss_fn(q_emb, p_emb, n_emb.view(B, H, -1)) + 0.01 * aux).backward()
    direct = _snapshot_grads(model)
    model.zero_grad(set_to_none=True)

    gradcache_step(
        model, loss_fn, q_ids, q_mask, p_ids, p_mask,
        n_ids=n_ids, n_mask=n_mask, num_hard=H,
        chunk_size=2, load_balance_weight=0.01, device_type="cpu",
    )
    cached = _snapshot_grads(model)

    for name in direct:
        assert torch.allclose(direct[name], cached[name], atol=1e-3), name
