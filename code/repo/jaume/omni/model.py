"""Jaume Omni: frozen encoders, trainable audio projection and Talker.

Training consumes *precomputed* SenseVoice features and Mimi codes so the NPU
process never imports funasr/torchaudio. That boundary is deliberate: it made
the audio stack reproducible on the Ascend image, where the system torchaudio
ABI is broken.

Sequence layout is [text][audio]. Text loss covers the text span; audio
loss covers only the codec frames. Input feature frames and output codec
frames have different rates, so the model resamples Talker states explicitly
instead of assuming equal lengths.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen3VLForConditionalGeneration

from .config import OmniConfig


class AudioProjector(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HiddenBridge(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CodebookHead(nn.Module):
    """Output [B, T, K, V]; codebooks stay a separate axis."""

    def __init__(self, hidden_size: int, num_codebooks: int, vocab_size: int):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.proj = nn.Linear(hidden_size, num_codebooks * vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        return self.proj(x).view(b, t, self.num_codebooks, self.vocab_size)


class JaumeOmniModel(nn.Module):
    def __init__(self, config: OmniConfig, load_thinker: bool = True):
        super().__init__()
        self.config = config
        self.thinker = None
        self.audio_projector = AudioProjector(
            config.audio_hidden_size,
            config.audio_projector_hidden,
            config.thinker_hidden_size,
        )
        self.bridge = HiddenBridge(config.thinker_hidden_size, config.bridge_hidden_size)
        talker_layer = nn.TransformerEncoderLayer(
            d_model=config.talker_hidden_size,
            nhead=config.talker_num_heads,
            dim_feedforward=config.talker_hidden_size * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.talker = nn.TransformerEncoder(
            talker_layer, num_layers=config.talker_num_layers
        )
        # Learned first frame for the codec stream. It makes the Talker an
        # actual causal next-frame predictor rather than a parallel classifier.
        self.codec_start = nn.Parameter(torch.zeros(1, 1, config.bridge_hidden_size))
        self.codebook_head = CodebookHead(
            config.talker_hidden_size,
            config.codec_num_codebooks,
            config.codec_codebook_size,
        )
        if load_thinker:
            self.load_thinker()

    def load_thinker(self) -> None:
        cfg = self.config
        if not cfg.thinker_path or not os.path.isdir(cfg.thinker_path):
            raise FileNotFoundError(f"thinker_path not found: {cfg.thinker_path!r}")
        dtype = torch.bfloat16 if cfg.torch_dtype == "bfloat16" else torch.float32
        self.thinker = Qwen3VLForConditionalGeneration.from_pretrained(
            cfg.thinker_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            attn_implementation=cfg.attn_implementation,
        )
        if cfg.freeze_thinker:
            for p in self.thinker.parameters():
                p.requires_grad_(False)
        n = cfg.unfreeze_last_n_layers
        if n > 0:
            layers = self.thinker.model.language_model.layers
            for layer in layers[-n:]:
                for p in layer.parameters():
                    p.requires_grad_(True)
            for p in self.thinker.model.language_model.norm.parameters():
                p.requires_grad_(True)
            for p in self.thinker.lm_head.parameters():
                p.requires_grad_(True)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def _resample_time(self, x: torch.Tensor, target_len: int, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Linear time resample of [B,T,D] to target_len.

        This is only a shape adapter before the Talker. It has no parameters
        and does not pretend to learn alignment; the Talker and bridge learn
        the semantic mapping from the resampled sequence plus codec targets.
        """
        if x.shape[1] == target_len:
            return x
        b, t, d = x.shape
        if mask is not None:
            # Gather valid frames per sample, resample each independently.
            outs = []
            for i in range(b):
                n = int(mask[i].sum().item())
                if n == 0:
                    outs.append(x.new_zeros(target_len, d))
                    continue
                xi = x[i, :n].transpose(0, 1).unsqueeze(0)
                yi = F.interpolate(xi.float(), size=target_len, mode="linear", align_corners=False)
                outs.append(yi.squeeze(0).transpose(0, 1).to(x.dtype))
            return torch.stack(outs, dim=0)
        xi = x.transpose(1, 2)
        yi = F.interpolate(xi.float(), size=target_len, mode="linear", align_corners=False)
        return yi.transpose(1, 2).to(x.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        audio_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        audio_input_mask: Optional[torch.Tensor] = None,
        text_labels: Optional[torch.Tensor] = None,
        text_segment_len: Optional[int] = None,
        audio_codes: Optional[torch.Tensor] = None,
        audio_codes_mask: Optional[torch.Tensor] = None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.shape[0], input_ids.shape[1] + audio_features.shape[1],
                device=input_ids.device, dtype=torch.long,
            )
        text_embeds = self.thinker.model.language_model.embed_tokens(input_ids)
        audio_embeds = self.audio_projector(audio_features.to(text_embeds.dtype))
        inputs_embeds = torch.cat([text_embeds, audio_embeds], dim=1)
        out = self.thinker(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        losses = {}
        total = None

        if text_labels is not None:
            # Causal LM objective: position t predicts token t+1. Predicting
            # text_labels[t] from logits[t] would leak the answer.
            n = text_segment_len if text_segment_len is not None else input_ids.shape[1]
            logits = out.logits[:, : max(n - 1, 1), :]
            targets = text_labels[:, 1:n]
            text_loss = F.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
            losses["text"] = text_loss.detach()
            total = self.config.text_loss_weight * text_loss if total is None else total + self.config.text_loss_weight * text_loss

        if audio_codes is not None:
            n_audio_in = audio_features.shape[1]
            hidden = out.hidden_states[self.config.bridge_layer][:, -n_audio_in:, :]
            talker_in = self.bridge(hidden)
            talker_in = self._resample_time(talker_in, audio_codes.shape[1], audio_input_mask)
            # Teacher-forced autoregressive Talker: [start, s_0, ..., s_{T-2}]
            # predicts codes at frames [0..T-1]. The causal mask prevents any
            # frame from attending to its own future frame.
            talker_in = torch.cat(
                [self.codec_start.expand(talker_in.shape[0], -1, -1), talker_in[:, :-1, :]],
                dim=1,
            )
            t = talker_in.shape[1]
            causal = torch.triu(
                torch.ones(t, t, device=talker_in.device, dtype=torch.bool), diagonal=1
            )
            talker_out = self.talker(talker_in, mask=causal)
            code_logits = self.codebook_head(talker_out)
            if audio_codes_mask is None:
                audio_codes_mask = torch.ones(audio_codes.shape[:2], device=audio_codes.device, dtype=torch.bool)
            mask = audio_codes_mask.bool()
            logits = code_logits[mask]
            targets = audio_codes[mask]
            audio_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
            losses["audio"] = audio_loss.detach()
            total = self.config.audio_loss_weight * audio_loss if total is None else total + self.config.audio_loss_weight * audio_loss

        if total is None:
            raise ValueError("forward needs text_labels and/or audio_codes")
        return total, losses

    @torch.no_grad()
    def generate_codes(self, talker_states: torch.Tensor) -> torch.Tensor:
        return self.codebook_head(talker_states).argmax(dim=-1)
