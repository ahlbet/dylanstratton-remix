import pytest
from config import (
    SAMPLE_RATE,
    CURRICULUM,
    LATENT_DIM,
    DEVICE,
    CHECKPOINT_DIR,
    GENERATED_DIR,
)
import torch


def test_sample_rate_positive():
    assert SAMPLE_RATE > 0, "Sample rate must be positive"


def test_latent_dim():
    assert isinstance(LATENT_DIM, int) and LATENT_DIM > 0


def test_curriculum_format():
    for stage, dur, epochs in CURRICULUM:
        assert isinstance(stage, str)
        assert isinstance(dur, (int, float)) and dur > 0
        assert isinstance(epochs, int) and epochs > 0


def test_device_string():
    assert isinstance(DEVICE, torch.device)
    assert str(DEVICE) in ("cpu", "cuda") or str(DEVICE).startswith("cuda:"), (
        "DEVICE should be 'cpu', 'cuda', or 'cuda:N'"
    )


def test_checkpoint_dir_is_str():
    assert isinstance(CHECKPOINT_DIR, str)
    assert CHECKPOINT_DIR != ""


def test_generated_dir_is_str():
    assert isinstance(GENERATED_DIR, str)
    assert GENERATED_DIR != ""


def test_curriculum_not_empty():
    assert len(CURRICULUM) > 0, "CURRICULUM should not be empty"


def test_curriculum_stage_names_unique():
    stage_names = [stage for stage, _, _ in CURRICULUM]
    assert len(stage_names) == len(set(stage_names)), (
        "Stage names in CURRICULUM should be unique"
    )


def test_sample_rate_reasonable():
    assert 8000 <= SAMPLE_RATE <= 192000, (
        "Sample rate should be within a reasonable range"
    )


def test_latent_dim_reasonable():
    assert 1 <= LATENT_DIM <= 4096, "LATENT_DIM should be within a reasonable range"
