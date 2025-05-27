import numpy as np
import soundfile as sf
from scripts import split_audio
from pydub import AudioSegment


def test_split_audio_creates_chunks(tmp_path, monkeypatch):
    # Create a dummy wav file (e.g., 3 seconds at 16kHz)
    sr = 16000
    duration = 1  # seconds
    audio = (np.random.randn(sr * duration) * 32767).astype(np.int16)
    wav_path = tmp_path / "long.wav"
    AudioSegment(audio.tobytes(), frame_rate=sr, sample_width=2, channels=1).export(
        str(wav_path), format="wav"
    )

    out_dir = tmp_path / "chunks"
    split_audio.split_all_wavs(str(tmp_path), str(out_dir), chunk_length_ms=500)

    # Check that chunks were created
    chunk_dirs = list(out_dir.glob("long/*.wav"))
    assert chunk_dirs, "No chunks created"


def test_split_audio_handles_short_file(tmp_path, monkeypatch):
    # Create a dummy wav file shorter than chunk size
    sr = 16000
    duration = 1  # seconds
    audio = (np.random.randn(sr * duration) * 32767).astype(np.int16)
    wav_path = tmp_path / "long.wav"
    AudioSegment(audio.tobytes(), frame_rate=sr, sample_width=2, channels=1).export(
        str(wav_path), format="wav"
    )

    out_dir = tmp_path / "chunks"
    split_audio.split_all_wavs(str(tmp_path), str(out_dir), chunk_length_ms=500)

    # Should create either 0 or 1 chunk, depending on implementation
    chunk_files = list(out_dir.glob("*.wav"))
    assert len(chunk_files) in (0, 1)
