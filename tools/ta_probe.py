import torch, sys
print("torch", torch.__version__)
try:
    import torchaudio
    print("torchaudio", torchaudio.__version__)
    import torchaudio.functional as F
    sr = 16000
    wav = torch.sin(2*3.1415926*440*torch.arange(sr*1)/sr).unsqueeze(0)
    torchaudio.save("/tmp/tone.wav", wav, sr)
    w2, sr2 = torchaudio.load("/tmp/tone.wav")
    print("save/load ok", w2.shape, sr2)
    m = F.mel_spectrogram(wav, sample_rate=sr, n_fft=512, hop_length=160, n_mels=80)
    print("mel ok", m.shape)
except Exception as e:
    print("FAIL", repr(e))