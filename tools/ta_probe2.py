import torch, torchaudio, torchaudio.transforms as T, torchaudio.functional as F
print("torchaudio", torchaudio.__version__)
sr=16000
wav=torch.sin(2*3.1415926*440*torch.arange(sr*1)/sr).unsqueeze(0)
mel=T.MelSpectrogram(sample_rate=sr,n_fft=512,hop_length=160,n_mels=80)(wav)
print("mel", mel.shape)
spec=F.spectrogram(wav, pad=0, window=torch.hann_window(512), n_fft=512, hop_length=160, power=2.0)
print("spec", spec.shape)