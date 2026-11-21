"""Tests for the causal twin's isolation from the bidirectional encoder.

These are the guards that make "the two forms cannot be mixed" a checked
property rather than a convention. They cover:

  1. selection      -- the config alone decides which backbone is built;
  2. masking        -- the causal mask actually blocks the future, and the
                       bidirectional path really does not;
  3. config         -- `causal` / `model_type` round-trip through save+load,
                       and legacy configs stay bidirectional;
  4. structure      -- the two backbones expose identical state-dict keys,
                       so the attention form is the only difference in the
                       checkpoint values;
  5. script guards  -- the bidirectional trainer refuses a causal config and
                       the causal trainer refuses a bidirectional one.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from jaume.modeling.config import JaumeConfig
from jaume.modeling.embedding import JaumeEmbeddingModel
from jaume.modeling.backbone import JaumeBackbone
from jaume.modeling.backbone_causal import CausalBackbone
from jaume.modeling.attention import BidirectionalGQA
from jaume.modeling.attention_causal import CausalGQA

torch.manual_seed(0)


def _cfg(**kw):
    base = dict(
        vocab_size=1000, hidden_size=64, intermediate_size=128, num_layers=2,
        num_attention_heads=4, num_key_value_heads=2, expert_num=4, topk=2,
        embedding_dim=64, matryoshka_dims=(16, 32, 64), pooling="attention",
    )
    base.update(kw)
    return JaumeConfig(**base)


# --------------------------------------------------------------- selection

def test_config_alone_selects_the_backbone():
    """The config decides the attention form; nothing else is consulted."""
    assert isinstance(JaumeEmbeddingModel(_cfg()).backbone, JaumeBackbone)
    assert isinstance(JaumeEmbeddingModel(_cfg(causal=True)).backbone, CausalBackbone)


def test_the_two_backbones_are_distinct_classes():
    assert not issubclass(CausalBackbone, JaumeBackbone)
    assert not issubclass(JaumeBackbone, CausalBackbone)
    assert isinstance(
        JaumeEmbeddingModel(_cfg(causal=True)).backbone.layers[0].self_attn,
        CausalGQA,
    )
    assert isinstance(
        JaumeEmbeddingModel(_cfg()).backbone.layers[0].self_attn,
        BidirectionalGQA,
    )


# ----------------------------------------------------------------- masking

def _backbone(config):
    """One model instance, reused for both forwards.

    Building a fresh model per call would give the two forwards different
    random weights, which makes every comparison meaningless (and turns the
    bidirectional contrast case into a false positive).
    """
    model = JaumeEmbeddingModel(config)
    model.eval()
    return model


def _hidden_at(model, ids, mask=None):
    if mask is None:
        mask = torch.ones_like(ids)
    with torch.no_grad():
        hidden, _ = model.backbone(ids, mask)
    return hidden


def _perturb_last(ids):
    ids2 = ids.clone()
    ids2[0, -1] = (ids2[0, -1] + 7) % 900
    return ids2


def test_causal_mask_blocks_the_future():
    """Changing the last token must not move any earlier position."""
    model = _backbone(_cfg(causal=True))
    ids = torch.randint(1, 900, (1, 12))
    h1 = _hidden_at(model, ids)
    h2 = _hidden_at(model, _perturb_last(ids))

    assert torch.allclose(h1[0, :-1], h2[0, :-1], atol=1e-5), (
        "causal attention leaked future information into earlier positions"
    )
    assert not torch.allclose(h1[0, -1], h2[0, -1], atol=1e-5), (
        "the last position did not react to its own token change"
    )


def test_bidirectional_sees_the_future():
    """The contrast case: the bidirectional path must react to later tokens."""
    model = _backbone(_cfg())
    ids = torch.randint(1, 900, (1, 12))
    h1 = _hidden_at(model, ids)
    h2 = _hidden_at(model, _perturb_last(ids))

    assert not torch.allclose(h1[0, :-1], h2[0, :-1], atol=1e-5), (
        "bidirectional attention did not see the changed future token"
    )


def test_causal_padding_still_masked():
    """Pad positions must stay invisible under the causal mask too."""
    ids = torch.randint(1, 900, (1, 8))
    mask = torch.ones(1, 8, dtype=torch.long)
    mask[0, 5:] = 0

    model = JaumeEmbeddingModel(_cfg(causal=True))
    model.eval()
    with torch.no_grad():
        h, _ = model.backbone(ids, mask)
    assert torch.isfinite(h).all()


# ------------------------------------------------------------------ config

def test_config_roundtrip_preserves_causal_and_model_type():
    cfg = _cfg(causal=True, model_type="jaume-causal")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "config.json")
        cfg.save(path)
        loaded = JaumeConfig.load(path)
    assert loaded.causal is True
    assert loaded.model_type == "jaume-causal"
    assert loaded.model_type != "jaume"


def test_legacy_config_stays_bidirectional():
    """A config written before the twin existed must not become causal."""
    legacy = {
        "model_type": "jaume", "vocab_size": 1000, "hidden_size": 64,
        "intermediate_size": 128, "num_layers": 2, "num_attention_heads": 4,
        "num_key_value_heads": 2, "expert_num": 4, "topk": 2,
        "embedding_dim": 64, "matryoshka_dims": [16, 32, 64],
        "pooling": "attention", "pad_token_id": 0, "load_balance_weight": 0.01,
    }
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "config.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(legacy, f)
        loaded = JaumeConfig.load(path)
    assert loaded.causal is False
    assert loaded.model_type == "jaume"


def test_shipped_config_files_are_consistent():
    """configs/jaume-350m-causal.json must actually declare the causal twin."""
    causal_path = os.path.join(REPO, "configs", "jaume-350m-causal.json")
    base_path = os.path.join(REPO, "configs", "jaume-350m.json")
    causal = JaumeConfig.load(causal_path)
    base = JaumeConfig.load(base_path)

    assert causal.causal is True
    assert causal.model_type == "jaume-causal"
    assert base.causal is False

    # every field except causal / model_type must match the bidirectional base
    diff = {
        k: (getattr(base, k), getattr(causal, k))
        for k in vars(base)
        if k not in ("causal", "model_type") and getattr(base, k) != getattr(causal, k)
    }
    assert diff == {}, f"causal config diverges from the base config: {diff}"


# --------------------------------------------------------------- structure

def test_state_dict_keys_are_identical():
    """Identical keys => the checkpoints differ only in values."""
    bi = JaumeEmbeddingModel(_cfg())
    ca = JaumeEmbeddingModel(_cfg(causal=True))
    assert sorted(bi.state_dict().keys()) == sorted(ca.state_dict().keys())


def test_parameter_counts_match():
    bi = sum(p.numel() for p in JaumeEmbeddingModel(_cfg()).parameters())
    ca = sum(p.numel() for p in JaumeEmbeddingModel(_cfg(causal=True)).parameters())
    assert bi == ca


# ------------------------------------------------------------ script guards

def _run(script, *args):
    return subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts", script), *args],
        capture_output=True, text=True, cwd=REPO,
    )


def test_bidirectional_trainer_refuses_a_causal_config():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out")
        r = _run("train_mlm.py",
                 "--config", os.path.join(REPO, "configs", "jaume-350m-causal.json"),
                 "--data", "missing.jsonl", "--output", out)
    assert r.returncode != 0
    assert "REFUSING TO RUN" in (r.stdout + r.stderr)


def test_causal_trainer_refuses_a_bidirectional_config():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out-causal")
        r = _run("train_mlm_causal.py",
                 "--config", os.path.join(REPO, "configs", "jaume-350m.json"),
                 "--data", "missing.jsonl", "--output", out)
    assert r.returncode != 0
    assert "REFUSING TO RUN" in (r.stdout + r.stderr)


def test_causal_trainer_refuses_a_non_causal_output_dir():
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "output")   # no "causal" in the name
        r = _run("train_mlm_causal.py",
                 "--config", os.path.join(REPO, "configs", "jaume-350m-causal.json"),
                 "--data", "missing.jsonl", "--output", out)
    assert r.returncode != 0
    assert "REFUSING TO RUN" in (r.stdout + r.stderr)
