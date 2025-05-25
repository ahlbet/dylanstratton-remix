import math
import torch.nn as nn
from torch.nn.utils import spectral_norm
from config import SAMPLE_RATE, LATENT_DIM


# Generator builder: duration in seconds → sample count internally
def build_generator_for(seq_len, latent_dim=LATENT_DIM, base_channels=512):
    """
    Returns a nn.Module that maps z:[B,latent_dim,1] → [B,1,seq_len] via
    successive 4× ConvTranspose1d layers (kernel_size=4, stride=4).
    """
    seq_len = int(seq_len * SAMPLE_RATE)  # convert seconds to samples
    # how many 4× upsamples are needed to reach or exceed seq_len
    n_layers = math.ceil(math.log(seq_len, 4))

    layers = []
    in_ch = latent_dim
    for i in range(n_layers):
        # for each layer, halve channels until we hit 1 at the last layer
        out_ch = base_channels // (2**i) if i < n_layers - 1 else 1
        if i == n_layers - 1:
            out_ch = 1

        layers.append(
            nn.ConvTranspose1d(
                in_ch, out_ch, kernel_size=4, stride=4, padding=0, output_padding=0
            )
        )
        if i < n_layers - 1:
            layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(True))
        else:
            layers.append(nn.Tanh())

        in_ch = out_ch

    net = nn.Sequential(*layers)

    class Generator(nn.Module):
        def __init__(self, net, seq_len):
            super().__init__()
            self.net = net
            self.seq_len = seq_len

        def forward(self, z):
            # z: [B, latent_dim, 1]
            x = self.net(z)  # [B, 1, 4**n_layers]
            return x[:, :, : self.seq_len]  # crop to exactly seq_len

    return Generator(net, seq_len)


# Discriminator builder: duration in seconds → architecture
def build_discriminator_for(seq_len, base_channels=512):
    """
    Returns an nn.Module that maps x:[B,1,seq_len] → [B,1] via
    successive 4× Conv1d downsampling layers (kernel_size=4, stride=4),
    spectral norm, LeakyReLU, then global average pooling + linear.
    """
    # 1) how many 4× downsamples to go from seq_len → ~1
    n_layers = math.ceil(math.log(seq_len, 4))

    layers = []
    in_ch = 1
    for i in range(n_layers):
        # double channels each layer but cap at base_channels
        out_ch = min(base_channels, 2 ** (i + 4))  # starts at 16, 32, 64, ...
        layers.append(spectral_norm(nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=4)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        in_ch = out_ch

    # global avg‐pool to length 1, flatten, then linear → 1 logit
    layers += [
        nn.AdaptiveAvgPool1d(1),  # [B, out_ch, 1]
        nn.Flatten(),  # [B, out_ch]
        nn.Linear(in_ch, 1),  # [B, 1]
    ]

    net = nn.Sequential(*layers)

    class Discriminator(nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net = net

        def forward(self, x):
            # x: [B,1,seq_len]
            return self.net(x)

    return Discriminator(net)
