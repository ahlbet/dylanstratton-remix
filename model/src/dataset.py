import os
import numpy as np
import soundfile as sf
import librosa
from torch.utils.data import Dataset
from src.config import PROCESSED_DATA_DIR, SAMPLE_RATE


class AudioDataset(Dataset):
    def __init__(self, folder, seq_len_samples):
        self.paths = [
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".wav")
        ]
        self.seq_len = seq_len_samples
        self.sample_rate = SAMPLE_RATE

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        wav, sr = sf.read(self.paths[idx])
        # mono
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        # resample
        if sr != self.sample_rate:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=self.sample_rate)
        # normalize
        wav = wav / (np.max(np.abs(wav)) + 1e-9)
        # pad/trim
        if len(wav) < self.seq_len:
            wav = np.pad(wav, (0, self.seq_len - len(wav)))
        else:
            wav = wav[: self.seq_len]
        return wav.astype(np.float32)
