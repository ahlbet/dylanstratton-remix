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


def test_empty_directory(tmp_path):
    ds = AudioDataset(str(tmp_path), seq_len=16000)
    assert len(ds) == 0, "Dataset should be empty if no files are present"


def test_multiple_files(tmp_path):
    # Create multiple dummy wav files
    for i in range(3):
        dummy = np.random.randn(16000).astype(np.float32)
        file = tmp_path / f"test_{i}.wav"
        sf.write(str(file), dummy, SAMPLE_RATE)
    ds = AudioDataset(str(tmp_path), seq_len=16000)
    assert len(ds) == 3, "Dataset should find all wav files"
    for i in range(3):
        sample = ds[i]
        assert isinstance(sample, (np.ndarray, torch.Tensor))
        assert sample.shape == (16000,) or sample.shape == (1, 16000)


def test_non_wav_files_ignored(tmp_path):
    # Create a dummy wav and a txt file
    dummy = np.random.randn(16000).astype(np.float32)
    wav_file = tmp_path / "test.wav"
    txt_file = tmp_path / "ignore.txt"
    sf.write(str(wav_file), dummy, SAMPLE_RATE)
    txt_file.write_text("not audio")
    ds = AudioDataset(str(tmp_path), seq_len=16000)
    assert len(ds) == 1, "Dataset should ignore non-wav files"


def test_short_file_is_handled(tmp_path):
    # Create a file shorter than seq_len
    dummy = np.random.randn(1000).astype(np.float32)
    file = tmp_path / "short.wav"
    sf.write(str(file), dummy, SAMPLE_RATE)
    ds = AudioDataset(str(tmp_path), seq_len=16000)
    # Depending on your implementation, this may be 0 or 1
    # Adjust the assertion as needed:
    assert len(ds) in (0, 1)


def test_dtype_is_float32(tmp_path):
    dummy = np.random.randn(16000).astype(np.float32)
    file = tmp_path / "test.wav"
    sf.write(str(file), dummy, SAMPLE_RATE)
    ds = AudioDataset(str(tmp_path), seq_len=16000)
    sample = ds[0]
    if isinstance(sample, np.ndarray):
        assert sample.dtype == np.float32
    elif isinstance(sample, torch.Tensor):
        assert sample.dtype == torch.float32
