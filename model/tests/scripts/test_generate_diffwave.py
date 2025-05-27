import diffwave_model
import torch
import torchaudio
from scripts import generate_diffwave


def test_main_creates_audio_file(tmp_path, monkeypatch):
    # Patch GENERATED_DIR to tmp_path
    monkeypatch.setattr(generate_diffwave, "GENERATED_DIR", str(tmp_path))
    # Patch CHECKPOINT_DIR to a temp dir (simulate a checkpoint)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    # Create a dummy model checkpoint
    dummy_state = torch.nn.Linear(100, 100).state_dict()
    g_ckpt = checkpoint_dir / "diffwave_epoch_1.pt"
    torch.save(dummy_state, g_ckpt)
    # Patch find_latest_checkpoint to return our dummy file
    monkeypatch.setattr(
        generate_diffwave,
        "find_latest_diffwave_checkpoint",
        lambda _: (str(g_ckpt)),
    )

    # Patch build_diffwave_model to return a dummy model
    class DummyDiffWave(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            # Dummy parameters to avoid unused parameter warnings

        def forward(self, z):
            return torch.zeros(z.shape[0], 1, 16000)

        def inference(self, *args, **kwargs):
            # Return a dummy waveform, adjust shape as needed
            return torch.zeros(1, 1, 16000)

        def load_state_dict(self, state_dict, strict=True):
            # Use parameters to avoid unused parameter warnings
            _ = state_dict, strict
            return self

        def eval(self):
            return self

    monkeypatch.setattr(diffwave_model, "DiffWave", DummyDiffWave)
    monkeypatch.setattr(generate_diffwave, "DiffWave", DummyDiffWave)

    # Run main
    generate_diffwave.main(duration_s=1)
    # Check that a file was created
    files = list(tmp_path.glob("*.wav"))
    assert files, "No audio file was generated"
    # Optionally, check file duration
    waveform, sr = torchaudio.load(files[0])
    assert sr == generate_diffwave.SAMPLE_RATE
    assert waveform.shape[1] == 16000  # 1 second at 16kHz


def test_main_runs_without_error(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_diffwave, "GENERATED_DIR", str(tmp_path))
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    g_ckpt = checkpoint_dir / "diffwave_epoch_1.pt"
    torch.save({}, g_ckpt)
    monkeypatch.setattr(
        generate_diffwave,
        "find_latest_diffwave_checkpoint",
        lambda _: (str(g_ckpt)),
    )

    # Patch build_diffwave_model to return a dummy model
    class DummyDiffWave(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            # Dummy parameters to avoid unused parameter warnings

        def forward(self, z):
            return torch.zeros(z.shape[0], 1, 16000)

        def inference(self, *args, **kwargs):
            # Return a dummy waveform, adjust shape as needed
            return torch.zeros(1, 1, 16000)

        def load_state_dict(self, state_dict, strict=True):
            # Use parameters to avoid unused parameter warnings
            _ = state_dict, strict
            return self

        def eval(self):
            return self

    monkeypatch.setattr(diffwave_model, "DiffWave", DummyDiffWave)
    monkeypatch.setattr(generate_diffwave, "DiffWave", DummyDiffWave)

    monkeypatch.setattr(
        generate_diffwave, "torchaudio", type("T", (), {"save": lambda *a, **k: None})
    )
    generate_diffwave.main(duration_s=1)  # Should not raise
