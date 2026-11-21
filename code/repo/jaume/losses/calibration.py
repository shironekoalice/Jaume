"""Calibration metrics and temperature scaling (SPEC §24, Stage 6).

Decision confidences are only useful when they carry statistical meaning:
a model claiming 0.90 should be right ~90% of the time. This module
implements the evaluation side (ECE / Brier / NLL / AUROC) and the cheapest
post-hoc fix (temperature scaling on a held-out split).
"""
import math

import torch
import torch.nn.functional as F


def _onehot(labels, n_classes):
    return F.one_hot(labels, n_classes).float()


def expected_calibration_error(probs, labels, n_bins=15):
    """ECE: sum over bins of |acc - conf| weighted by bin share.

    probs: [N, C] predicted probabilities (rows sum to 1).
    labels: [N] int class indices.
    """
    probs, labels = probs.detach().float().cpu(), labels.detach().long().cpu()
    conf, pred = probs.max(dim=-1)
    correct = pred.eq(labels).float()
    ece = torch.zeros(())
    edges = torch.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.float().mean() * (correct[in_bin].mean() - conf[in_bin].mean()).abs()
    return ece.item()


def brier_score(probs, labels):
    """Multiclass Brier: mean over samples of sum_c (p_c - onehot_c)^2."""
    probs, labels = probs.detach().float().cpu(), labels.detach().long().cpu()
    target = _onehot(labels, probs.size(-1))
    return ((probs - target) ** 2).sum(dim=-1).mean().item()


def negative_log_likelihood(probs, labels, eps=1e-12):
    probs, labels = probs.detach().float().cpu(), labels.detach().long().cpu()
    picked = probs.gather(1, labels.unsqueeze(1)).clamp(min=eps)
    return -picked.log().mean().item()


def auroc(scores, labels):
    """Binary AUROC by rank statistics (Mann-Whitney U).

    scores: [N] continuous scores; labels: [N] 0/1.
    Ties handled with average ranks.
    """
    scores = scores.detach().float().cpu()
    labels = labels.detach().long().cpu()
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = torch.argsort(scores)
    ranks = torch.empty(len(scores), dtype=torch.float32)
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1  # average rank for ties
        i = j + 1
    rank_sum_pos = ranks[labels == 1].sum()
    return ((rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)).item()


def reliability_diagram(probs, labels, n_bins=15):
    """Returns per-bin (confidence, accuracy, share) lists for plotting."""
    probs, labels = probs.detach().float().cpu(), labels.detach().long().cpu()
    conf, pred = probs.max(dim=-1)
    correct = pred.eq(labels).float()
    edges = torch.linspace(0, 1, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        share = in_bin.float().mean().item()
        if in_bin.any():
            bins.append({
                "bin": [lo.item(), hi.item()],
                "confidence": conf[in_bin].mean().item(),
                "accuracy": correct[in_bin].mean().item(),
                "share": share,
            })
        else:
            bins.append({"bin": [lo.item(), hi.item()], "confidence": None,
                         "accuracy": None, "share": share})
    return bins


def calibration_report(probs, labels, n_bins=15):
    """One-call report for a decision head's probabilities.

    For binary problems AUROC is computed against the labels; otherwise it
    measures how well confidence discriminates correct from incorrect
    predictions (auroc_conf).
    """
    probs, labels = probs.detach().float().cpu(), labels.detach().long().cpu()
    conf = probs.max(dim=-1).values
    report = {
        "ece": expected_calibration_error(probs, labels, n_bins),
        "brier": brier_score(probs, labels),
        "nll": negative_log_likelihood(probs, labels),
    }
    if probs.size(1) == 2:
        report["auroc"] = auroc(probs[:, 1], labels)
    else:
        correct = probs.argmax(dim=-1).eq(labels)
        report["auroc_conf"] = auroc(conf, correct.long())
    return report


class TemperatureScaler:
    """Post-hoc calibration: fit a single temperature on validation logits.

    Fitting minimizes NLL of softmax(logits / T) against labels. Lowering
    the loss sharpens/softens all classes uniformly, leaving accuracy and
    ranking unchanged — only the confidence changes.
    """

    def __init__(self):
        self.log_temperature = torch.nn.Parameter(torch.zeros(()))

    @property
    def temperature(self):
        return self.log_temperature.exp().item()

    def fit(self, logits, labels, max_iter=200, lr=0.05):
        logits = logits.detach().float().cpu()
        labels = labels.detach().long().cpu()
        self.log_temperature = torch.nn.Parameter(torch.zeros(()))
        opt = torch.optim.LBFGS([self.log_temperature], lr=lr, max_iter=max_iter)

        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(logits / self.log_temperature.exp(), labels)
            loss.backward()
            return loss

        opt.step(closure)
        return self.temperature

    def __call__(self, logits):
        return F.softmax(logits.detach().float().cpu() / self.temperature, dim=-1)
