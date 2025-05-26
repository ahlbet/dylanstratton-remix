import os
import numpy as np
import torch
import soundfile as sf
from dataset import AudioDataset
from config import PROCESSED_DATA_DIR, SAMPLE_RATE


def test_dataset_shapes(tmp_path, monkeypatch):
    # create a dummy WAV file
    dummy = np.random.randn(16000).astype(np.float32)
    file = tmp_path / "test.wav"
    sf.write(str(file), dummy, SAMPLE_RATE)
    # point dataset to tmp_path
    ds = AudioDataset(str(tmp_path), seq_len=16000)
    sample = ds[0]
    assert isinstance(sample, (np.ndarray, torch.Tensor))
    assert sample.shape == (16000,) or sample.shape == (1, 16000), (
        "Sample shape mismatch"
    )

    # test torch compatibility
    if isinstance(sample, np.ndarray):
        tensor = torch.from_numpy(sample)
    elif isinstance(sample, torch.Tensor):
        tensor = sample
    else:
        raise TypeError(f"Unexpected sample type: {type(sample)}")
    assert tensor.dtype == torch.float32
