import os
import torch
import types
import tempfile
import torchaudio
import shutil
from scripts import generate_audio


def test_main_creates_audio_file(tmp_path, monkeypatch):
    # Patch GENERATED_DIR to tmp_path
    monkeypatch.setattr(generate_audio, "GENERATED_DIR", str(tmp_path))
    # Patch CHECKPOINT_DIR to a temp dir (simulate a checkpoint)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    # Create a dummy generator checkpoint
    dummy_state = torch.nn.Linear(100, 100).state_dict()
    g_ckpt = checkpoint_dir / "G_epoch_1.pt"
    torch.save(dummy_state, g_ckpt)
    d_ckpt = checkpoint_dir / "D_epoch_1.pt"
    torch.save(dummy_state, d_ckpt)
    # Patch find_latest_checkpoint to return our dummy files
    monkeypatch.setattr(
        generate_audio,
        "find_latest_checkpoint",
        lambda _: (str(g_ckpt), str(d_ckpt), 1),
    )

    # Patch build_generator_for to return a dummy model
    class DummyGen(torch.nn.Module):
        def forward(self, z):
            # Return a fixed waveform: [batch, 1, seq_len]
            return torch.zeros(z.shape[0], 1, generate_audio.SEQ_LEN)

        def load_state_dict(self, state_dict, strict=True):
            return self  # Ignore loading

    monkeypatch.setattr(
        generate_audio, "build_generator_for", lambda duration: DummyGen()
    )
    # Run main
    generate_audio.main(duration_s=1)
    # Check that a file was created
    files = list(tmp_path.glob("*.wav"))
    assert files, "No audio file was generated"
    # Optionally, check file duration
    waveform, sr = torchaudio.load(files[0])
    assert sr == generate_audio.SAMPLE_RATE
    assert waveform.shape[1] == generate_audio.SEQ_LEN


def test_main_runs_without_error(tmp_path, monkeypatch):
    # Patch everything as above, but just check for exceptions
    monkeypatch.setattr(generate_audio, "GENERATED_DIR", str(tmp_path))
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    dummy_state = torch.nn.Linear(100, 100).state_dict()
    g_ckpt = checkpoint_dir / "G_epoch_1.pt"
    torch.save(dummy_state, g_ckpt)
    d_ckpt = checkpoint_dir / "D_epoch_1.pt"
    torch.save(dummy_state, d_ckpt)
    monkeypatch.setattr(
        generate_audio,
        "find_latest_checkpoint",
        lambda _: (str(g_ckpt), str(d_ckpt), 1),
    )

    class DummyGen(torch.nn.Module):
        def forward(self, z):
            return torch.zeros(z.shape[0], 1, generate_audio.SEQ_LEN)

        def load_state_dict(self, state_dict, strict=True):
            return self

    monkeypatch.setattr(
        generate_audio, "build_generator_for", lambda duration: DummyGen()
    )
    # Should not raise
    generate_audio.main(duration_s=1)
