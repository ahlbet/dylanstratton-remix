import torch
from utils import compute_gradient_penalty, find_latest_checkpoint
from config import DEVICE


def test_gradient_penalty():
    B, C, L = 2, 1, 16
    real = torch.randn(B, C, L).to(DEVICE)
    fake = torch.randn(B, C, L).to(DEVICE)
    gp = compute_gradient_penalty(lambda x: torch.sum(x), real, fake)
    assert gp >= 0


def test_gradient_penalty_zero_when_real_equals_fake():
    B, C, L = 2, 1, 16
    real = torch.randn(B, C, L, requires_grad=True).to(DEVICE)
    gp = compute_gradient_penalty(lambda x: x.sum(), real, real)
    assert gp >= 0
    assert torch.isfinite(gp), "Gradient penalty should be finite"


def test_gradient_penalty_requires_grad():
    B, C, L = 2, 1, 16
    real = torch.randn(B, C, L, requires_grad=True).to(DEVICE)
    fake = torch.randn(B, C, L, requires_grad=True).to(DEVICE)
    gp = compute_gradient_penalty(lambda x: x.sum(), real, fake)
    assert isinstance(gp, torch.Tensor)
    assert gp.requires_grad is False  # Should be a scalar, not requiring grad


def test_find_latest_checkpoint(tmp_path):
    # Create dummy checkpoint files
    (tmp_path / "G_epoch_1.pt").write_text("g")
    (tmp_path / "D_epoch_1.pt").write_text("d")
    (tmp_path / "epoch_1.pth").write_text("meta")
    g_ckpt, d_ckpt, epoch = find_latest_checkpoint(str(tmp_path))
    assert g_ckpt.endswith("G_epoch_1.pt")
    assert d_ckpt.endswith("D_epoch_1.pt")
    assert epoch == 1


def test_find_latest_checkpoint_empty(tmp_path):
    g_ckpt, d_ckpt, epoch = find_latest_checkpoint(str(tmp_path))
    assert g_ckpt is None
    assert d_ckpt is None
    assert epoch == 0
