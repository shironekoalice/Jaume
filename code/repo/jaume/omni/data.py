"""Dataset contract for Jaume Omni precomputed features.

Storage is one JSONL manifest plus one .pt file per sample:

    {"id": "...", "audio": "features/000001.pt",
     "audio_frames": 98, "code_frames": 13, "text": "..."}

The .pt file is a dict:
    audio_features: float32 [T_in, 512]  SenseVoice encoder output
    audio_codes:    int64   [T_out, 8]   Mimi codes at 12.5 Hz
    text_ids:       int64   [L]          optional pre-tokenised text

Input and output frame rates differ (SenseVoice ~16.7 Hz vs Mimi 12.5 Hz),
so the two lengths are stored separately. Never infer one from the other.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class OmniFeatureDataset(Dataset):
    def __init__(self, manifest: str, text_tokenizer=None, max_text_length: int = 128,
                 audio_dim: int = 512, num_codebooks: int = 8):
        self.root = os.path.dirname(os.path.abspath(manifest))
        self.records: List[dict] = []
        with open(manifest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.tokenizer = text_tokenizer
        self.max_text_length = max_text_length
        self.audio_dim = audio_dim
        self.num_codebooks = num_codebooks

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rec = self.records[idx]
        blob = torch.load(os.path.join(self.root, rec["audio"]), map_location="cpu")
        text = rec.get("text") or ""
        if self.tokenizer is not None and text:
            text_ids = self.tokenizer(
                text, truncation=True, max_length=self.max_text_length,
                return_tensors="pt",
            )["input_ids"][0].long()
        elif "text_ids" in blob:
            text_ids = blob["text_ids"].long()
        else:
            text_ids = torch.empty(0, dtype=torch.long)
        return {
            "audio_features": blob["audio_features"].float(),
            "audio_codes": blob["audio_codes"].long(),
            "text_ids": text_ids,
        }


def collate_omni(batch: List[Dict[str, torch.Tensor]], pad_token_id: int = 0,
                 audio_dim: int = 512):
    """Right-pad text and audio time axes; masks follow the true lengths."""
    bsz = len(batch)
    max_text = max((b["text_ids"].numel() for b in batch), default=1)
    max_ain = max((b["audio_features"].shape[0] for b in batch), default=1)
    max_aout = max((b["audio_codes"].shape[0] for b in batch), default=1)
    n_cb = max((b["audio_codes"].shape[1] for b in batch), default=8)

    text_ids = torch.full((bsz, max_text), pad_token_id, dtype=torch.long)
    text_mask = torch.zeros(bsz, max_text, dtype=torch.bool)
    audio_features = torch.zeros(bsz, max_ain, audio_dim, dtype=torch.float32)
    audio_input_mask = torch.zeros(bsz, max_ain, dtype=torch.bool)
    audio_codes = torch.zeros(bsz, max_aout, n_cb, dtype=torch.long)
    audio_codes_mask = torch.zeros(bsz, max_aout, dtype=torch.bool)

    for i, b in enumerate(batch):
        nt = b["text_ids"].numel()
        ni = b["audio_features"].shape[0]
        no = b["audio_codes"].shape[0]
        text_ids[i, :nt] = b["text_ids"]
        text_mask[i, :nt] = True
        audio_features[i, :ni] = b["audio_features"]
        audio_input_mask[i, :ni] = True
        audio_codes[i, :no, :b["audio_codes"].shape[1]] = b["audio_codes"]
        audio_codes_mask[i, :no] = True

    return {
        "input_ids": text_ids,
        "text_mask": text_mask,
        "audio_features": audio_features,
        "audio_input_mask": audio_input_mask,
        "audio_codes": audio_codes,
        "audio_codes_mask": audio_codes_mask,
    }
