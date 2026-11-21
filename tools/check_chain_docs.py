import sys
sys.path.insert(0, r"code\repo\scripts")
from derive_env import make_chain_puzzle, render
from train_lm_derive import build_docs, encode
from jaume.data.dataset import load_tokenizer
tok = load_tokenizer(r"code\repo\jaume_tokenizer")
tok.add_tokens(["[MASK]"])
for k in (2, 4):
    docs = build_docs(3, 8, 4, 0, 0, 0, 1, chain_len=k)
    enc = encode(docs, tok, 192)
    print("chain", k, "docs", len(docs), "enc", len(enc))
    for d, e in zip(docs, enc):
        print("  prefix:", d["prefix"][:100])
        print("  t1:", d["target1"], "| t2:", d["target2"],
              "| span", e["start1"], e["start2"], e["len"])
