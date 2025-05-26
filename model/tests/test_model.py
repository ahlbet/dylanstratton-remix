import torch
from model import build_generator_for, build_discriminator_for, SAMPLE_RATE


def test_gan_shapes():
    seq_len = SAMPLE_RATE
    G = build_generator_for(1).to("cpu")
    D = build_discriminator_for(1).to("cpu")
    z = torch.randn(2, 100, 1)
    fake = G(z)
    assert fake.shape == (2, 1, seq_len)
    out = D(fake)
    assert out.shape == (2, 1)
