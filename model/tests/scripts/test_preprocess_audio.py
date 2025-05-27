import os
import numpy as np
import torch
import soundfile as sf
from scripts import preprocess_audio


def test_process_file_and_augment(tmp_path, monkeypatch):
    # Create a dummy wav file
    dummy = np.random.randn(16000).astype(np.float32)
    wav_path = tmp_path / "test.wav"
    sf.write(str(wav_path), dummy, 16000)

    # Patch INPUT_DIR and OUTPUT_DIR in the script
    monkeypatch.setattr(preprocess_audio, "INPUT_DIR", tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(preprocess_audio, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(preprocess_audio, "SAMPLE_RATE", 16000)
    monkeypatch.setattr(preprocess_audio, "N_AUG", 2)

    # Patch augment to just return a list of arrays
    monkeypatch.setattr(
        preprocess_audio, "augment", lambda audio, sr: [audio + 1, audio - 1]
    )

    # Patch process_file to just return the audio
    monkeypatch.setattr(
        preprocess_audio, "process_file", lambda path: sf.read(str(path))[0]
    )

    # Run main
    preprocess_audio.main()

    # Check that cleaned and augmented files exist
    files = list(out_dir.glob("*.wav"))
    assert any("clean" in f.name for f in files)
    assert any("aug1" in f.name for f in files)
    assert any("aug2" in f.name for f in files)


def test_main_handles_no_files(tmp_path, monkeypatch):
    # Patch INPUT_DIR and OUTPUT_DIR in the script
    monkeypatch.setattr(preprocess_audio, "INPUT_DIR", tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    monkeypatch.setattr(preprocess_audio, "OUTPUT_DIR", out_dir)
    monkeypatch.setattr(preprocess_audio, "SAMPLE_RATE", 16000)
    monkeypatch.setattr(preprocess_audio, "N_AUG", 2)
    # Patch augment and process_file to avoid errors
    monkeypatch.setattr(preprocess_audio, "augment", lambda audio, sr: [audio])
    monkeypatch.setattr(
        preprocess_audio, "process_file", lambda path: np.zeros(16000, dtype=np.float32)
    )
    # Should not raise
    preprocess_audio.main()
    assert not list(out_dir.glob("*.wav")), (
        "No files should be created if no input files exist"
    )
