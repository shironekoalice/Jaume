"""Configuration for the Jaume Omni Thinker-Talker model.

The config deliberately records *both* the frozen component paths and the
trainable dimensions. A checkpoint is therefore not reusable with a different
audio encoder or codec by accident: the loader checks channel/frame contract
before it silently trains a mismatched projector.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List


@dataclass
class OmniConfig:
    # Thinker (Qwen3-VL for the first version)
    thinker_path: str = ""
    thinker_hidden_size: int = 2048
    thinker_num_layers: int = 28
    # Hidden state read by the Talker bridge. The config stores an absolute
    # layer index; -1 means the last transformer layer output.
    bridge_layer: int = 14

    # Audio input: SenseVoice-Small encoder output (verified 512-d, ~16.7 Hz)
    audio_encoder_path: str = ""
    audio_hidden_size: int = 512
    audio_frame_rate: float = 16.7
    audio_projector_hidden: int = 2048

    # Speech output: Mimi, 8 codebooks, 12.5 Hz, 24 kHz
    codec_path: str = ""
    codec_num_codebooks: int = 8
    codec_codebook_size: int = 2048
    codec_frame_rate: float = 12.5
    codec_sample_rate: int = 24000
    # Interleaved (MusicGen-style) decoding: one shared vocabulary predicted
    # in (frame, codebook) order, so residual codebooks see their parents.
    # Parallel heads treat codebooks as independent, which mismatches Mimi's
    # residual-VQ structure.
    codec_interleave: bool = False

    # Talker / bridge
    talker_hidden_size: int = 768
    talker_num_layers: int = 4
    talker_num_heads: int = 12
    bridge_hidden_size: int = 768

    # Optional speaker/timbre conditioning for TTS. 0 disables it.
    speaker_num: int = 0
    speaker_embedding_dim: int = 256
    speaker_condition_weight: float = 1.0

    # Training
    text_loss_weight: float = 1.0
    audio_loss_weight: float = 1.0
    freeze_thinker: bool = True
    unfreeze_last_n_layers: int = 0
    attn_implementation: str = "sdpa"
    torch_dtype: str = "bfloat16"

    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["model_type"] = "jaume-omni"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "OmniConfig":
        known = {f for f in cls.__dataclass_fields__}
        extra = {k: v for k, v in d.items() if k not in known}
        kwargs = {k: v for k, v in d.items() if k in known}
        kwargs["extra"] = extra
        return cls(**kwargs)
