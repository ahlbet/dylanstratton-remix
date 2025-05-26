import os
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
    def __init__(self, folder, seq_len):
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
        wav = wav / wav.abs().max()
        return wav
