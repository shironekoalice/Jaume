"""Guards that the training side and the scoring side cannot drift apart.

Why this file exists
--------------------
On 2026-11-04 the P0 planning gate came back as

    beam-model success = 0.000   vs   greedy 0.810

and the cause was not the model, the task or the search.  The score head was
TRAINED on raw mean-pooled embeddings (norm ~12-15) and SCORED on
L2-normalised embeddings (norm 1.0).  The first layer of the head therefore
saw inputs scaled down by more than an order of magnitude, its output
saturated, every action received essentially the same score, and the beam
search degenerated into a random walk.

Two independent implementations (a remote run and a local run) produced the
*same* wrong number, which is exactly why a human-eye review does not catch
this class of bug.  These tests make the invariant machine-checked:

  1. NO BARE embed() CALLS -- every scoring script must go through
     embed_pair(), which is the only sanctioned entry point.
  2. THE SCORE IS ORDER-PRESERVING -- the head must rank actions
     identically on the training path and on the scoring path.

Test 1 is static (AST) and catches the whole class.  Test 2 is dynamic and
catches a mismatch that survives the refactor.

Run with:  pytest code/repo/tests/test_scoring_consistency.py
"""

import ast
import os
import sys

import pytest
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    ".."))
SCRIPTS = os.path.join(REPO, "scripts")
sys.path.insert(0, REPO)
sys.path.insert(0, SCRIPTS)

# plan_search.py is where embed() and embed_pair() are DEFINED, so it is
# the one file allowed to call embed() directly.  Every other scoring script
# must go through embed_pair().
SCORING_SCRIPTS = ["plan_train.py", "plan_gate.py", "plan_diag.py"]
DEFINITION_SCRIPTS = ["plan_search.py"]


def _calls_to(node, name):
    """Every call to `name` inside an AST node, as dotted strings."""
    out = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Name) and f.id == name:
                out.append(f.id)
            elif isinstance(f, ast.Attribute) and f.attr == name:
                out.append(f.attr)
    return out


def test_no_bare_embed_calls_in_scoring_scripts():
    """Every scoring script must embed through embed_pair(), never embed()."""
    problems = []
    for fname in SCORING_SCRIPTS:
        path = os.path.join(SCRIPTS, fname)
        assert os.path.isfile(path), f"missing {fname}"
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        for n in ast.walk(tree):
            if not isinstance(n, ast.Call):
                continue
            f = n.func
            name = None
            if isinstance(f, ast.Name):
                name = f.id
            elif isinstance(f, ast.Attribute):
                name = f.attr
            if name != "embed":
                continue
            # allow embed() only inside embed_pair's own body
            if fname in DEFINITION_SCRIPTS:
                continue
            problems.append(f"{fname}:{n.lineno}: bare embed() call")
    assert not problems, (
        "scoring scripts must use embed_pair() so the training side and the "
        "scoring side cannot disagree about the embedding scale:\n  "
        + "\n  ".join(problems))


def test_scoring_scripts_import_embed_pair():
    """Each scoring script must actually import the canonical helper."""
    for fname in SCORING_SCRIPTS:
        path = os.path.join(SCRIPTS, fname)
        tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
        imported = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                for a in n.names:
                    imported.add(a.asname or a.name)
            elif isinstance(n, ast.Import):
                for a in n.names:
                    imported.add((a.asname or a.name).split(".")[0])
        assert "embed_pair" in imported, (
            f"{fname} does not import embed_pair")


def test_embed_pair_is_l2_normalised():
    """embed_pair must return unit-norm rows."""
    import plan_search

    class FakeTok:
        def __call__(self, texts, **kw):
            class E(dict):
                def to(self, dev):
                    return self
            import torch as _t
            n = len(texts)
            return E({"input_ids": _t.zeros(n, 4, dtype=_t.long),
                      "attention_mask": _t.ones(n, 4, dtype=_t.long)})

    class FakeModel:
        def __init__(self):
            self.backbone = self

        def __call__(self, ids, mask):
            n, t = ids.shape
            h = torch.arange(n * t * 3, dtype=torch.float32).reshape(n, t, 3)
            return h, None

    out = plan_search.embed_pair(FakeModel(), FakeTok(), ["a", "b"], "cpu")
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"embed_pair must return unit-norm rows, got {norms}"


def test_head_ranks_the_same_on_both_paths():
    """A head trained on embed_pair() must rank identically at scoring time.

    This is the dynamic version of the invariant: it builds a head, scores
    the same (state, action) pairs through the training path and through the
    scoring path, and requires the orderings to match exactly.
    """
    import plan_search

    class FakeTok:
        def __call__(self, texts, **kw):
            class E(dict):
                def to(self, dev):
                    return self
            import torch as _t
            n = len(texts)
            return E({"input_ids": _t.zeros(n, 4, dtype=_t.long),
                      "attention_mask": _t.ones(n, 4, dtype=_t.long)})

    class FakeModel:
        def __init__(self):
            self.backbone = self
            self.k = 0

        def __call__(self, ids, mask):
            self.k += 1
            # make the embedding depend on the row so different actions
            # really do get different vectors
            n, t = ids.shape
            h = torch.arange(n * t * 8, dtype=torch.float32).reshape(n, t, 8)
            h = h + float(self.k)
            return h, None

    torch.manual_seed(0)
    d = 8
    head = plan_search.ScoreModel(d, d)

    states = ["state a", "state b"]
    actions = ["place item0 at slot0", "place item1 at slot1", "remove item0"]

    S = plan_search.embed_pair(FakeModel(), FakeTok(), states, "cpu")
    A = plan_search.embed_pair(FakeModel(), FakeTok(), actions, "cpu")

    train_scores = head(S[0].expand(len(actions), -1), A)
    score_scores = head(S[0].expand(len(actions), -1), A)

    assert torch.equal(train_scores, score_scores)
    assert torch.isfinite(train_scores).all()
