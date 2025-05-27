import torch
from model import build_generator_for, build_discriminator_for, SAMPLE_RATE
from config import LATENT_DIM


def test_gan_shapes():
    seq_len = SAMPLE_RATE
    G = build_generator_for(1).to("cpu")
    D = build_discriminator_for(1).to("cpu")
    z = torch.randn(2, 100, 1)
    fake = G(z)
    assert fake.shape == (2, 1, seq_len)
    out = D(fake)
    assert out.shape == (2, 1)


def test_generator_output_range():
    G = build_generator_for(1).to("cpu")
    z = torch.randn(4, LATENT_DIM, 1)
    fake = G(z)
    # Check output shape
    assert fake.shape == (4, 1, SAMPLE_RATE)
    # Check output is finite
    assert torch.isfinite(fake).all()
    # Optional: check output range if using Tanh
    assert fake.max() <= 1.0 and fake.min() >= -1.0


def test_discriminator_output_shape():
    D = build_discriminator_for(1).to("cpu")
    x = torch.randn(3, 1, SAMPLE_RATE)
    out = D(x)
    assert out.shape == (3, 1)
    assert torch.isfinite(out).all()


def test_generator_gradients():
    G = build_generator_for(1).to("cpu")
    z = torch.randn(2, LATENT_DIM, 1, requires_grad=True)
    fake = G(z)
    loss = fake.sum()
    loss.backward()
    assert z.grad is not None
    assert torch.isfinite(z.grad).all()


def test_discriminator_gradients():
    D = build_discriminator_for(1).to("cpu")
    x = torch.randn(2, 1, SAMPLE_RATE, requires_grad=True)
    out = D(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_generator_batch_independence():
    G = build_generator_for(1).to("cpu")
    z1 = torch.randn(1, LATENT_DIM, 1)
    z2 = torch.randn(1, LATENT_DIM, 1)
    out1 = G(z1)
    out2 = G(z2)
    assert not torch.allclose(out1, out2), (
        "Generator should produce different outputs for different inputs"
    )
