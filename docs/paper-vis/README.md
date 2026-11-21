# What a 350M Vision-Language Model Can and Cannot Do

Working draft. Three controlled negative results plus a diagnostic protocol,
all from building Jaume-Vis on a single Ascend 910B3.

## Files

- `paper.html` - open in a browser (MathJax renders the formulas)
- `paper.tex`  - LaTeX source for submission
- `refs.bib`   - bibliography

## One-line claim

A representation either contains the information a task needs or it does not.
A three-minute linear probe on the frozen encoder answers that before any
training run, and it predicted all three of our results.

## The three results

1. **Looping cannot help when depth does not.** Eight task structures, four
   modalities, 8L always matches 16L. The decisive case is 8-digit addition:
   learnable (0.89), strictly scored, genuinely serial, still saturated.
2. **A semantic vision tower cannot read.** SigLIP2 and InternViT-448 both
   score 0.21 (chance 0.25) on four-way text location; TrOCR reaches 0.75.
   Resolution is not the cause - InternViT at 448px fails too.
3. **Exact match hides capability.** Point localisation reads 0.11 exact vs
   0.50 with a tolerance; detection reads 0.058 exact vs 0.158 IoU.

## Evidence

Each number in the paper traces to a document in `docs/`:
`loop-spec.md`, `vlm-dual-tower-spec.md`, `ocr-diagnosis.md`,
`context-length.md`, `kv-cache.md`.