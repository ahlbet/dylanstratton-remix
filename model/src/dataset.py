import os
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import Resample
from glob import glob
import soundfile as sf
import librosa
import numpy as np
from config import PROCESSED_DATA_DIR, SAMPLE_RATE


class AudioDataset(Dataset):
    def __init__(self, folder, seq_len, n_mels=80, hop_length=200, n_fft=1024):
        self.seq_len = seq_len
        self.paths = glob(os.path.join(folder, "**/*.wav"), recursive=True)
        self.files = []
        for p in self.paths:
            audio = self._quick_load(p)
            if self._is_valid(audio):
                self.files.append(p)
            else:
                print(f"⚠️  Skipping bad file: {p}")
        self.resample = Resample(orig_freq=44100, new_freq=SAMPLE_RATE)

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        self.hop_length = hop_length

    def _quick_load(self, path):
        # load, resample, mono, pad/trim—same as your process_file but minimal
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
        # pad/trim
        if len(wav) < self.seq_len:
            wav = np.pad(wav, (0, self.seq_len - len(wav)))
        else:
            wav = wav[: self.seq_len]
        return wav.astype(np.float32)

    def _is_valid(self, audio):
        # no NaNs/Infs, not silent, not clipped
        if not np.isfinite(audio).all():
            return False
        peak = np.max(np.abs(audio))
        if peak < 1e-5:  # essentially silent
            return False
        if peak > 1.0 + 1e-3:  # clipped
            return False
        return True

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.files[idx])
        if sr != SAMPLE_RATE:
            wav = self.resample(wav)
        wav = wav.mean(dim=0, keepdim=True)  # mono
        if wav.shape[-1] < self.seq_len:
            pad_len = self.seq_len - wav.shape[-1]
            wav = nn.functional.pad(wav, (0, pad_len))
        else:
            wav = wav[:, : self.seq_len]
        wav = wav / wav.abs().max()  # normalize

        # Generate mel spectrogram
        mel = self.mel_transform(
            wav.squeeze(0)
        )  # Remove channel dim before mel transform
        # Convert to log-scale
        mel = torch.log1p(mel)  # [n_mels, T]

        # Interpolate mel to fixed time dimension
        target_length = self.seq_len // self.hop_length
        mel = torch.nn.functional.interpolate(
            mel.unsqueeze(0),  # [1, n_mels, T]
            size=target_length,
            mode="linear",
            align_corners=False,
        ).squeeze(0)  # [n_mels, target_length]

        return wav, mel  # [1, L], [n_mels, T]
