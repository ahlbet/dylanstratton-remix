import math
import torch
import torch.nn as nn

from config import SAMPLE_RATE, LATENT_DIM, DEVICE


# Sinusoidal timestep embedding
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb  # [B, dim]


# Basic Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels, time_emb_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)
        # add time embedding
        time_proj = self.time_mlp(t_emb)[:, :, None]
        h = h + time_proj
        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)
        return x + h


class DiffWave(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.device = DEVICE
        # diffusion hyperparameters
        self.timesteps = model_config.get("timesteps", 1000)
        beta_start = model_config.get("beta_start", 1e-4)
        beta_end = model_config.get("beta_end", 0.02)
        self.betas = torch.linspace(
            beta_start, beta_end, self.timesteps, device=self.device
        )
        self.alphas = 1.0 - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)
        # model architecture
        time_emb_dim = model_config.get("time_emb_dim", 128)
        channels = model_config.get("channels", 64)
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        self.mlp_time = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.initial_conv = nn.Conv1d(1, channels, kernel_size=3, padding=1)
        # stack residual blocks
        self.res_blocks = nn.ModuleList(
            [
                ResidualBlock(channels, time_emb_dim)
                for _ in range(model_config.get("num_res_blocks", 6))
            ]
        )
        self.final_norm = nn.GroupNorm(8, channels)
        self.final_act = nn.SiLU()
        self.final_conv = nn.Conv1d(channels, 1, kernel_size=3, padding=1)

    def forward(self, x, t):
        """
        x: [B, 1, L], t: [B] timestep indices
        returns predicted noise epsilon
        """
        t_emb = self.time_emb(t)
        t_emb = self.mlp_time(t_emb)
        h = self.initial_conv(x)
        for block in self.res_blocks:
            h = block(h, t_emb)
        h = self.final_norm(h)
        h = self.final_act(h)
        return self.final_conv(h)

    def compute_loss(self, clean):
        """
        clean: [B, 1, L]
        returns MSE loss between predicted and true noise
        """
        b, _, l = clean.shape
        device = clean.device
        # sample random timesteps
        t = torch.randint(0, self.timesteps, (b,), device=device)
        # gather alpha_hat
        a_hat = self.alpha_hat[t][:, None, None]
        noise = torch.randn_like(clean)
        x_t = torch.sqrt(a_hat) * clean + torch.sqrt(1 - a_hat) * noise
        pred_noise = self.forward(x_t, t)
        return nn.functional.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def inference(self, seq_len):
        """
        returns: [1, 1, seq_len]
        """
        device = self.device
        x = torch.randn(1, 1, seq_len, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((1,), i, device=device, dtype=torch.long)
            a_hat = self.alpha_hat[i]
            beta = self.betas[i]
            # predict noise
            eps = self.forward(x, t)
            # compute posterior mean
            coef1 = 1 / torch.sqrt(self.alphas[i])
            coef2 = beta / torch.sqrt(1 - self.alpha_hat[i])
            mu = coef1 * (x - coef2 * eps)
            if i > 0:
                z = torch.randn_like(x)
                sigma = torch.sqrt(beta)
                x = mu + sigma * z
            else:
                x = mu
        return x
