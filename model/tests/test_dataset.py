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
    audio, mel = ds[0]  # Dataset now returns (audio, mel) tuple

    # Test audio shape and type
    assert isinstance(audio, torch.Tensor), f"Expected torch.Tensor, got {type(audio)}"
    assert audio.shape == (1, 16000), f"Expected shape (1, 16000), got {audio.shape}"
    assert audio.dtype == torch.float32, f"Expected float32, got {audio.dtype}"

    # Test mel spectrogram shape and type
    assert isinstance(mel, torch.Tensor), f"Expected torch.Tensor, got {type(mel)}"
    assert len(mel.shape) == 2, f"Expected 2D tensor for mel, got shape {mel.shape}"
    assert mel.shape[0] == 80, (
        f"Expected 80 mel bins, got {mel.shape[0]}"
    )  # frequency bins
    assert mel.dtype == torch.float32, f"Expected float32, got {mel.dtype}"

    # Test value ranges
    assert torch.all(audio >= -1) and torch.all(audio <= 1), (
        "Audio values outside [-1, 1] range"
    )
    assert torch.all(mel >= 0), "Negative values in mel spectrogram"


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
        audio, mel = ds[i]
        assert isinstance(audio, torch.Tensor)
        assert audio.shape == (1, 16000)
        assert isinstance(mel, torch.Tensor)
        assert len(mel.shape) == 2
        assert mel.shape[0] == 80  # mel frequency bins


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
    audio, mel = ds[0]
    assert audio.dtype == torch.float32
    assert mel.dtype == torch.float32
