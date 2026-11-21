"""Measure the real context length of Jaume-Causal: PPL vs sequence length.

The config advertises max_seq_len=8192, but every training run used sequences
of 128-256 tokens. RoPE generalises beyond its training range only if it was
trained there, so the honest question is: where does the model actually stop
being coherent? This probe answers it with a perplexity-vs-length curve.

Method: take long contiguous text, feed prefixes of increasing length, and
report loss on the SAME trailing window (e.g. the last 128 tokens) for every
prefix length. A model with real long context should keep or improve its
loss on that window as more preceding context is added; a model whose RoPE
was never trained past position N will show loss climbing once the prefix
exceeds N.

This "same-window, growing-prefix" design isolates long-context use from
any per-position difficulty differences.
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
REPO = os.path.join(PROJECT_ROOT, "code", "repo")
sys.path.insert(0, REPO)

from jaume.device import resolve_device, describe
from jaume.modeling.config import JaumeConfig
from jaume.modeling.embedding import JaumeEmbeddingModel
from jaume.modeling.checkpoint import load_state_dict
from jaume.data.dataset import load_tokenizer

# A long, coherent public-domain text with varied vocabulary and structure.
# Kept inline so the probe has zero download dependencies.
LONG_TEXT = """
The honourable Member asked about the condition of the roads in the northern
district, and I shall endeavour to answer him with the same candour with which
the question was put. In the autumn of the preceding year the county surveyor
submitted his report, wherein he observed that the gravel beds near the quarry
had been drawn down to a depth at which the subsoil, being of a clayey nature,
would in wet seasons rise through the metal and destroy the camber of the road.
The trustees, upon reading this report, resolved to appoint a committee of
inspection, and that committee, having walked the whole line from the toll gate
at Millbridge to the turnpike end at Farthington, reported in favour of a new
specification for the passing places, and of laying a double course of stone
upon the greater part of the road. Now, sir, I do not deny that the expense of
this undertaking was considerable, nor that several gentlemen of the county
opposed it upon the ground that the old road had served their fathers, and
might therefore serve themselves. But the answer to such persons is short and
conclusive: that the traffic upon the road had doubled within the memory of the
present surveyor; that waggons of four tons were now drawn where two tons had
formerly been the utmost; and that what was sufficient for an age of horses
could not serve an age of steam. I remember standing in the month of November
at the corner of the lane which leads up from Marlow's farm, and watching the
carts come down through the mud in a season of almost continual rain. The
carriers, who were men of small means, lost time and lost horses; the farmer
could not send his milk to the creamery in time for the morning train; and the
carrier's boy, who rode the leading horse, was thrown into the ditch when the
animal sank to the belly in a soft place over which every cart had to pass.
These are small matters, and a gentleman in another place may laugh at them;
but they are the matters upon which the comfort of a village depends, and they
were remedied, and the road was made, and in the following summer I drove over
it in a light carriage at a pace which the old road would not have permitted
in the driest weather. It is therefore that I say to the honourable Member
that the movement for the reform of our country roads is not a whimsical
movement, but one grounded in the experience of those who use them; and that
the trustees of a turnpike who neglect the report of a competent surveyor
neglect the first duty of their office, which is to keep the passage open,
safe, and such as the traffic of the time requires.
""".strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lengths", default="128,256,512,1024,2048,4096,8192")
    ap.add_argument("--window", type=int, default=128,
                    help="loss is measured on the last N tokens")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = resolve_device(args.device)
    print("[ctx] device=%s (%s)" % (device, describe(device)), flush=True)

    cfg = JaumeConfig.load(os.path.join(args.model, "config.json"))
    tok = load_tokenizer(os.path.join(args.model, "tokenizer"))
    cfg.vocab_size = len(tok)
    cfg.causal = True

    enc = JaumeEmbeddingModel(cfg).to(device)
    sd = load_state_dict(os.path.join(args.model, "pytorch_model.bin"))
    # A causal-LM checkpoint stores backbone.* + lm_head.weight only; the
    # embedding wrapper's pooling/projection heads are not part of it.
    own = enc.state_dict()
    merged, missing = {}, []
    for k in own:
        if k in sd and sd[k].shape == own[k].shape:
            merged[k] = sd[k]
        else:
            merged[k] = own[k]
            missing.append(k)
    enc.load_state_dict(merged, strict=True)
    if missing:
        print("[ctx] kept %d non-backbone tensors from init: %s"
              % (len(missing), missing[:3]), flush=True)
    enc.eval()
    # backbone.forward returns hidden states; the LM head is a separate
    # (tied) weight inside the checkpoint. Use it to get real token logits.
    # The MLM checkpoint has no separate LM head; tie to the input embedding,
    # which is the standard weight-tying arrangement for small causal LMs
    # (loop_lm.py supports exactly this via tie_lm_head=True).
    import torch.nn as nn
    w = sd.get("lm_head.weight")
    if w is None:
        w = sd["backbone.embed_tokens.weight"]
        print("[ctx] no lm_head in checkpoint; tying to embeddings", flush=True)
    head = nn.Linear(w.shape[1], w.shape[0], bias=False).to(device)
    with torch.no_grad():
        head.weight.copy_(w.to(device))
    head.eval()

    ids = tok(LONG_TEXT, add_special_tokens=False)["input_ids"]
    print("[ctx] corpus tokens=%d, advertised max_seq_len=%d" % (
        len(ids), cfg.max_seq_len), flush=True)
    # tile the text so every requested length is reachable
    while len(ids) < max(int(x) for x in args.lengths.split(",")) + 64:
        ids = ids + ids

    lengths = sorted({int(x) for x in args.lengths.split(",")})
    W = args.window
    print("\n[ctx] prefix_len  window_loss  window_ppl", flush=True)

    results = []
    for L in lengths:
        if L <= W:
            continue
        prefix = torch.tensor([ids[:L]], device=device)
        # loss on the final W positions, with full causal context before them
        with torch.no_grad():
            h, _ = enc.backbone(prefix[:, :-1], prefix[:, :-1] != cfg.pad_token_id)
            logits = head(h.float())
            tgt = prefix[:, 1:]
            lg = logits[0, -W:, :].float()
            tg = tgt[0, -W:]
            loss = F.cross_entropy(lg, tg)
        ppl = float(loss.exp())
        results.append((L, float(loss), ppl))
        print("[ctx] %9d  %11.4f  %10.2f" % (L, float(loss), ppl), flush=True)

    # Interpretation helper: a flat curve means the extra prefix is being used
    # or at least tolerated; a knee means RoPE has run past its training range.
    if len(results) >= 2:
        first, last = results[0][1], results[-1][1]
        print("\n[ctx] window loss: first=%.4f last=%.4f delta=%+.4f" % (
            first, last, last - first), flush=True)
    print("\n[ctx] interpretation: loss rising sharply after prefix length P "
          "means positions beyond P were never trained; the real context is "
          "approximately P.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
