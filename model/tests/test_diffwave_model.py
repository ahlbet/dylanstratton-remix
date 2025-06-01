import torch
from diffwave_model import DiffWave


model_config = {
    "timesteps": 200,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "time_emb_dim": 128,
    "channels": 64,
    "num_res_blocks": 6,
}


def test_diffwave_forward_shape():
    model = DiffWave(model_config).cpu()
    batch_size = 2
    channels = 1
    seq_len = 16000
    x = torch.randn(batch_size, channels, seq_len)
    t = torch.randint(0, 50, (batch_size,))
    mel = torch.randn(batch_size, 80, 100)
    out = model(x, t, mel)
    assert out.shape == (batch_size, channels, seq_len)


def test_diffwave_requires_grad():
    model = DiffWave(model_config).cpu()
    x = torch.randn(1, 1, 8000, requires_grad=True)
    t = torch.randint(0, 50, (1,))
    mel = torch.randn(1, 80, 100)
    out = model(x, t, mel)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None


def test_diffwave_eval_mode():
    model = DiffWave(model_config).cpu()
    model.eval()
    x = torch.randn(1, 1, 8000)
    t = torch.randint(0, 50, (1,))
    mel = torch.randn(1, 80, 100)
    with torch.no_grad():
        out = model(x, t, mel)
    assert out.shape == x.shape


def test_diffwave_zero_input():
    model = DiffWave(model_config).cpu()
    x = torch.zeros(1, 1, 8000)
    t = torch.zeros(1, dtype=torch.long)
    mel = torch.randn(1, 80, 100)
    out = model(x, t, mel)
    assert torch.isfinite(out).all()
