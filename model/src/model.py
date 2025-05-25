import math
import torch.nn as nn
from torch.nn.utils import spectral_norm
from src.config import SAMPLE_RATE, LATENT_DIM

# Generator builder: duration in seconds → sample count internally
def build_generator_for(duration_s, latent_dim=LATENT_DIM, base_channels=512):
    seq_len = int(duration_s * SAMPLE_RATE)
    n_layers = max(1, math.ceil(math.log(seq_len, 4)))
    layers = []
    in_ch = latent_dim
    for i in range(n_layers):
        out_ch = 1 if i == n_layers - 1 else base_channels // (2**i)
        layers.append(nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=4))
        if i < n_layers - 1:
            layers += [nn.BatchNorm1d(out_ch), nn.ReLU(True)]
        else:
            layers.append(nn.Tanh())
        in_ch = out_ch

    conv_net = nn.Sequential(*layers)
    class Generator(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = conv_net
            self.seq_len = seq_len
        def forward(self, z):
            x = self.net(z)
            return x[:, :, :self.seq_len]
    return Generator()

# Discriminator builder: duration in seconds → architecture
def build_discriminator_for(duration_s, base_channels=512):
    seq_len = int(duration_s * SAMPLE_RATE)
    n_layers = max(1, math.ceil(math.log(seq_len, 4)))
    layers = []
    in_ch = 1
    for i in range(n_layers):
        out_ch = min(base_channels, 2 ** (i + 4))
        layers += [
            spectral_norm(nn.Conv1d(in_ch, out_ch, 4, stride=4)),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        in_ch = out_ch
    layers += [
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
        nn.Linear(in_ch, 1)
    ]
    return nn.Sequential(*layers)