import pytest
from config import (
    SAMPLE_RATE,
    CURRICULUM,
    LATENT_DIM,
    DEVICE,
    CHECKPOINT_DIR,
    GENERATED_DIR,
)


def test_sample_rate_positive():
    assert SAMPLE_RATE > 0, "Sample rate must be positive"


def test_latent_dim():
    assert isinstance(LATENT_DIM, int) and LATENT_DIM > 0


def test_curriculum_format():
    for stage, dur, epochs in CURRICULUM:
        assert isinstance(stage, str)
        assert isinstance(dur, (int, float)) and dur > 0
        assert isinstance(epochs, int) and epochs > 0
