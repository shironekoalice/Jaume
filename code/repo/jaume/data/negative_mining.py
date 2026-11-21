"""Hard negative mining (SPEC §16).

Pipeline: initial encoder → retrieve top-K from corpus → exclude the true
positive and its near-duplicates → similarity-band filtering → optional
teacher/human filtering → {query, positive, hard_negative} training rows.

The miner is encoder-agnostic: any model exposing an
``encode(texts) -> Tensor[N, D]`` (L2-normalized rows) works.
"""
import torch


class HardNegativeMiner:
    """Dense-retrieval hard negative miner over an in-memory corpus."""

    def __init__(
        self,
        encoder,
        tokenizer,
        device=None,
        batch_size=128,
        max_length=128,
        encode_batch_size=256,
    ):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.device = torch.device(device) if isinstance(device, str) else (
            device or next(encoder.parameters()).device
        )
        self.batch_size = batch_size
        self.max_length = max_length
        self.encode_batch_size = encode_batch_size

    @torch.no_grad()
    def _encode(self, texts):
        """Returns [N, D] L2-normalized embeddings on self.device."""
        use_amp = self.device.type in ("cuda", "xpu")
        chunks = []
        for i in range(0, len(texts), self.encode_batch_size):
            batch = texts[i : i + self.encode_batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=use_amp):
                emb, _ = self.encoder(
                    enc["input_ids"].to(self.device),
                    enc["attention_mask"].to(self.device),
                )
            chunks.append(emb.float())
        emb = torch.cat(chunks, dim=0)
        return torch.nn.functional.normalize(emb, dim=-1)

    def build_index(self, corpus_texts):
        """Encodes the corpus once; returns (texts, embedding matrix)."""
        self.encoder.eval()
        embeddings = self._encode(corpus_texts)
        return corpus_texts, embeddings

    @torch.no_grad()
    def mine(
        self,
        queries,
        positives,
        corpus_texts,
        corpus_embeddings=None,
        num_negatives=1,
        top_k=64,
        min_score=0.30,
        max_score=0.90,
        pos_margin=0.02,
        dup_threshold=0.95,
        filter_fn=None,
    ):
        """Mines up to `num_negatives` hard negatives per (query, positive).

        A candidate is accepted when all hold:
        - not the positive itself, nor a near-duplicate of it
          (cosine to positive > `dup_threshold`)
        - not a near-duplicate of the query (false-negative guard)
        - similarity to the query in [min_score, max_score] and at least
          `pos_margin` below the query-positive similarity
        - `filter_fn(query, positive, negative, score)` returns True,
          when provided (teacher / human filtering hook)

        Candidates are scanned hardest-first (highest similarity).
        Returns a list of dicts with query/positive/hard_negative fields;
        rows with too few accepted candidates are dropped.
        """
        if corpus_embeddings is None:
            corpus_texts, corpus_embeddings = self.build_index(corpus_texts)
        self.encoder.eval()

        # group corpus rows by text so every duplicate of the positive is excluded
        text_to_idx = {}
        for i, t in enumerate(corpus_texts):
            text_to_idx.setdefault(t, []).append(i)

        rows = []
        for i in range(0, len(queries), self.batch_size):
            q_texts = queries[i : i + self.batch_size]
            p_texts = positives[i : i + self.batch_size]
            q_emb = self._encode(q_texts)  # [B, D]
            sims = q_emb @ corpus_embeddings.T  # [B, N]

            pos_emb = self._encode(p_texts)  # [B, D]
            pos_sims = (q_emb * pos_emb).sum(-1)  # [B, N=1]
            dup_to_pos = pos_emb @ corpus_embeddings.T > dup_threshold  # [B, N]

            k = min(top_k, sims.size(1))
            top_scores, top_idx = sims.topk(k, dim=1)  # [B, k]

            for b in range(len(q_texts)):
                query, positive = q_texts[b], p_texts[b]
                excluded = set(text_to_idx.get(positive, []))
                p_sim = pos_sims[b].item()
                mined = []
                for score, idx in zip(top_scores[b].tolist(), top_idx[b].tolist()):
                    if len(mined) >= num_negatives:
                        break
                    if idx in excluded:
                        continue
                    if dup_to_pos[b, idx]:
                        continue
                    cand = corpus_texts[idx]
                    if cand == query or cand == positive:
                        continue
                    if not (min_score <= score <= max_score):
                        continue
                    if score >= p_sim - pos_margin:
                        continue
                    if filter_fn is not None and not filter_fn(query, positive, cand, score):
                        continue
                    mined.append({"text": cand, "score": round(score, 4)})
                if not mined:
                    continue
                rows.append({
                    "query": query,
                    "positive": positive,
                    "hard_negatives": [m["text"] for m in mined],
                    "hard_negative_scores": [m["score"] for m in mined],
                })
        return rows

    @staticmethod
    def save(rows, path):
        import json

        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
