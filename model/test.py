import os
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from scipy.signal import butter, lfilter
from torch.utils.data import DataLoader, Dataset


class AmbientDataset(Dataset):
    def __init__(self, folder, sample_rate=8000, duration=2):
        print(folder)

        print(glob(os.path.join(folder, "**/*.wav"), recursive=True))

        print(os.curdir)

        #    Check if directory exists
        print(f"Directory exists: {os.path.exists(folder)}")

        # List all files in the directory
        print("Files in directory:", os.listdir(folder))
        abs_path = os.path.abspath(folder)

        self.paths = glob(os.path.join(folder, "**/*.wav"), recursive=True)
        self.sample_rate = sample_rate
        self.samples = int(sample_rate * duration)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.paths[idx])
        waveform = waveform.mean(dim=0)  # mono
        waveform = waveform / torch.max(torch.abs(waveform))  # normalize
        waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        waveform = waveform[: self.samples]  # truncate/pad
        if waveform.shape[0] < self.samples:
            padding = self.samples - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        return waveform

    def lowpass_filter(self, data, cutoff=4000, order=6):
        fs = self.sample_rate
        nyq = 0.5 * fs
        # print(f"Nyquist frequency: {nyq}")
        normal_cutoff = cutoff / nyq
        # print(f"Normal cutoff: {normal_cutoff}")
        b, a = butter(order, normal_cutoff, btype="low")
        return lfilter(b, a, data)


duration = 4  # seconds
sample_rate = 16000
output_size = int(sample_rate * duration)
latent_dim = 100

dataset = AmbientDataset(
    "training-samples/split", sample_rate=sample_rate, duration=duration
)
loader = DataLoader(dataset, batch_size=1, shuffle=True)


class Conv1dGenerator(nn.Module):
    def __init__(self, latent_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            # Input: [B, 100, 1]
            nn.ConvTranspose1d(latent_dim, 512, 4, stride=1),  # [B, 512, 4]
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, 4, stride=4),  # [B, 256, 16]
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, 4, stride=4),  # [B, 128, 64]
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, 4, stride=4),  # [B, 64, 256]
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, 4, stride=4),  # [B, 32, 1024]
            nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 4, stride=4),  # [B, 16, 4096]
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, 4, stride=4),  # [B, 1, 16384]
            nn.Tanh(),
        )

    def forward(self, z):
        return self.net(z)


class Conv1dDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, 4, stride=4),  # [B, 16, 4000]
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, 4, stride=4),  # [B, 32, 1000]
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, 4, stride=4),  # [B, 64, 250]
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, 4, stride=4),  # [B, 128, 62]
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 4, stride=4),  # [B, 256, 15]
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(256 * 15, 1),
        )

    def forward(self, x):
        return self.net(x)


G = Conv1dGenerator()
D = Conv1dDiscriminator()
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
optimizer_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))


def gradient_penalty(D, real_data, fake_data, device):
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1).to(device)
    interpolates = (alpha * real_data + ((1 - alpha) * fake_data)).requires_grad_(True)

    d_interpolates = D(interpolates)
    fake = torch.ones_like(d_interpolates, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    penalty = ((grad_norm - 1) ** 2).mean()
    return penalty


lambda_gp = 10  # Gradient penalty coefficient
n_critic = 5  # Train D more often than G

for epoch in range(1):
    for i, real in enumerate(loader):
        real = real.to(torch.float32)
        real = real.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        G.to(real.device)
        D.to(real.device)

        batch_size = real.size(0)
        z = torch.randn(batch_size, 100).to(real.device)
        fake = G(z)

        real_labels = torch.ones(batch_size, 1).to(real.device)
        fake_labels = torch.zeros(batch_size, 1).to(real.device)

        # Train Discriminator
        # D_real = D(real)
        # D_fake = D(fake.detach())
        # loss_D = criterion(D_real, real_labels) + criterion(D_fake, fake_labels)
        # optimizer_D.zero_grad()
        # loss_D.backward()
        # optimizer_D.step()
        for _ in range(n_critic):
            z = torch.randn(batch_size, latent_dim).to(real.device)
            fake = G(z).detach()
            d_real = D(real).mean()
            d_fake = D(fake).mean()
            gp = gradient_penalty(D, real, fake, real.device)

            d_loss = d_fake - d_real + lambda_gp * gp
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

        # Train Generator
        # D_fake = D(fake)
        # loss_G = criterion(D_fake, real_labels)
        # optimizer_G.zero_grad()
        # loss_G.backward()
        # optimizer_G.step()
        z = torch.randn(batch_size, latent_dim, 1).to(real.device)
        fake = G(z)
        g_loss = -D(fake).mean()

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    torch.save(
        {
            "epoch": epoch,
            "generator_state_dict": G.state_dict(),
            "discriminator_state_dict": D.state_dict(),
            "g_optimizer_state_dict": optimizer_G.state_dict(),
            "d_optimizer_state_dict": optimizer_D.state_dict(),
            "loss_G": g_loss,
            "loss_D": d_loss,
        },
        f"checkpoints/ambient_checkpoint_epoch{epoch}.pth",
    )

    print(
        f"Epoch {epoch + 1} - Loss D: {d_loss.item():.4f}, Loss G: {g_loss.item():.4f}"
    )

G.eval()
z = torch.randn(1, 100)
sample = G(z).detach().cpu().numpy().squeeze()
torchaudio.save("generated.wav", torch.tensor(sample).unsqueeze(0), sample_rate)
