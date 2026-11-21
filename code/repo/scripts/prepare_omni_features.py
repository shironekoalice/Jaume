#!/usr/bin/env python3
"""Precompute SenseVoice features and Mimi codes for Omni training.

Run inside the isolated CPU audio venv (the NPU trainer never imports these):

    LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1     TORCH_DEVICE_BACKEND_AUTOLOAD=0     /mnt/workspace/venvs/jaume-omni/bin/python scripts/prepare_omni_features.py       --input manifest.jsonl --output /mnt/workspace/data/omni_smoke

Input JSONL: {"id": "...", "audio": "path.wav|mp3", "text": "optional transcript"}
Output: features/<id>.pt and train.jsonl. Each .pt holds:
  audio_features float32 [T_in, 512]  SenseVoice speech frames only
  audio_codes    int64   [T_out, 8]   Mimi codes at 12.5 Hz
  text_ids       int64   [L]          only when a transcript is given

SenseVoice's encoder emits four control/query positions before the speech
frames. We drop those four, matching the model's own CTC path
(), so the stored length is actual audio frames.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

import torch
import torchaudio


_TAG = re.compile(r"<\|[^|]*\|>")


def strip_rich_tags(text: str) -> str:
    return _TAG.sub("", text).strip()


def load_wav(path: str, sample_rate: int) -> torch.Tensor:
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


def _pad_waveforms(wavs, lengths):
    """Right-pad mono waveforms and return [B,T] plus int64 lengths."""
    max_len = max(int(x.shape[1]) for x in wavs)
    batch = torch.zeros(len(wavs), max_len, dtype=torch.float32)
    lens = torch.tensor([x.shape[1] for x in wavs], dtype=torch.int64)
    for i, x in enumerate(wavs):
        batch[i, :x.shape[1]] = x[0]
    return batch, lens


def _sensevoice_encode_batch(sv, frontend, wavs, language="zh"):
    """Run the frozen SenseVoice encoder on a true batch.

    `AutoModel.generate()` loops one utterance at a time, so it cannot use the
    CPU's available parallelism. This path reproduces its frontend/query logic
    but calls the encoder once per batch.
    """
    speech, speech_lengths = _pad_waveforms(wavs, None)
    with torch.no_grad():
        speech, speech_lengths = frontend(speech, speech_lengths)
        language_query = sv.embed(
            torch.LongTensor([[sv.lid_dict.get(language, 0)]])
        ).repeat(speech.size(0), 1, 1)
        textnorm_query = sv.embed(
            torch.LongTensor([[sv.textnorm_dict["withitn"]]])
        ).repeat(speech.size(0), 1, 1)
        speech = torch.cat((textnorm_query, speech), dim=1)
        speech_lengths += 1
        event_emo_query = sv.embed(
            torch.LongTensor([[1, 2]])
        ).repeat(speech.size(0), 1, 1)
        input_query = torch.cat((language_query, event_emo_query), dim=1)
        speech = torch.cat((input_query, speech), dim=1)
        speech_lengths += 3
        enc_out, enc_lens = sv.encoder(speech, speech_lengths)
        if isinstance(enc_out, tuple):
            enc_out = enc_out[0]
    return enc_out.cpu(), enc_lens.cpu()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sensevoice", default="/mnt/workspace/models/SenseVoiceSmall")
    ap.add_argument("--mimi", default="/mnt/workspace/models/Mimi")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--num-codebooks", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Encoder batch size for feature extraction.")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="Split input into this many disjoint shards.")
    ap.add_argument("--shard-index", type=int, default=0,
                    help="Which shard this process handles (0-based).")
    args = ap.parse_args()
    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        raise SystemExit("invalid shard configuration")

    feat_dir = os.path.join(args.output, "features")
    os.makedirs(feat_dir, exist_ok=True)

    from funasr import AutoModel
    from transformers import AutoTokenizer, MimiModel

    am = AutoModel(model=args.sensevoice, device="cpu", disable_update=True)
    sv = am.model
    mimi = MimiModel.from_pretrained(args.mimi, torch_dtype=torch.float32).to(args.device).eval()
    tok = AutoTokenizer.from_pretrained(
        "/home/developer/.cache/modelscope/models/Qwen--Qwen3-VL-2B-Instruct/snapshots/master",
        trust_remote_code=True,
    )

    with open(args.input, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        records = records[: args.limit]
    records = records[args.shard_index::args.num_shards]

    shard_tag = f".shard-{args.shard_index:03d}" if args.num_shards > 1 else ""
    manifest_path = os.path.join(args.output, f"train{shard_tag}.jsonl")
    kept = 0
    t0 = time.time()
    batch_size = max(1, int(args.batch_size))
    with open(manifest_path, "w", encoding="utf-8") as out:
        for batch_start in range(0, len(records), batch_size):
            batch_records = records[batch_start:batch_start + batch_size]
            audio_paths = []
            for rec in batch_records:
                audio_path = rec["audio"]
                if not os.path.isabs(audio_path):
                    audio_path = os.path.join(
                        os.path.dirname(os.path.abspath(args.input)), audio_path
                    )
                audio_paths.append(audio_path)

            wavs = [load_wav(p, 16000) for p in audio_paths]
            enc_batch, enc_lens = _sensevoice_encode_batch(
                sv, am.kwargs["frontend"], wavs, args.language
            )

            wavs24 = _pad_waveforms(
                [load_wav(p, 24000) for p in audio_paths], None
            )[0].to(args.device)
            with torch.no_grad():
                code_batch = mimi.encode(
                    wavs24.unsqueeze(1), num_quantizers=args.num_codebooks
                ).audio_codes.cpu()
            # Mimi returns [B,K,T]; store [T,K] per sample.
            code_batch = code_batch.long().transpose(1, 2).contiguous()

            for j, (rec, audio_path) in enumerate(zip(batch_records, audio_paths)):
                n = max(int(enc_lens[j].item()) - 4, 1)
                feats = enc_batch[j, 4:4 + n, :].detach().float().contiguous()
                text = (rec.get("text") or "").strip()
                if not text:
                    result = am.generate(
                        input=audio_path, cache={}, language=args.language,
                        use_itn=True, batch_size_s=60,
                    )
                    text = strip_rich_tags(result[0].get("text", "")).strip()

                # Real length, not padded length. Mimi frame rate is fixed, so
                # sum the valid 24k samples before converting to codec frames.
                n_samples = int(load_wav(audio_path, 24000).shape[1])
                n_codes = min(int(round(n_samples / 1920)), code_batch.shape[1])
                codes = code_batch[j, :max(n_codes, 1)]
                text_ids = tok(
                    text, return_tensors="pt", truncation=True, max_length=128
                )["input_ids"][0]

                i = batch_start + j
                sid = rec.get("id") or f"sample_{i:06d}"
                safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(sid))
                if args.num_shards > 1:
                    safe = f"{safe}{shard_tag}"
                rel = os.path.join("features", f"{safe}.pt")
                torch.save({
                    "audio_features": feats,
                    "audio_codes": codes,
                    "text_ids": text_ids,
                }, os.path.join(args.output, rel))
                out.write(json.dumps({
                    "id": str(sid),
                    "audio": rel,
                    "text": text,
                    "audio_frames": int(feats.shape[0]),
                    "code_frames": int(codes.shape[0]),
                }, ensure_ascii=False) + "\n")
                kept += 1
            out.flush()
            total_done = batch_start + len(batch_records)
            if batch_start == 0 or total_done % 40 == 0:
                print(
                    f"[prepare] {total_done}/{len(records)} "
                    f"kept={kept} elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )

    print(
        f"[prepare] done {kept} samples in {time.time()-t0:.0f}s -> {manifest_path}",
        flush=True,
    )

if __name__ == "__main__":
    main()
