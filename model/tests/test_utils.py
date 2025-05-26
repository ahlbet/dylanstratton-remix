import torch
from utils import compute_gradient_penalty


def test_gradient_penalty():
    B, C, L = 2, 1, 16
    real = torch.randn(B, C, L)
    fake = torch.randn(B, C, L)
    gp = compute_gradient_penalty(lambda x: torch.sum(x), real, fake)
    assert gp >= 0
