import os
import math
import re
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchaudio.transforms import Resample
from tqdm import tqdm
from glob import glob
from torch.nn.utils import spectral_norm
import soundfile as sf
import librosa
import numpy as np

# ----- Hyperparameters -----
# each tuple is (stage_name, seq_len_seconds, epochs_to_train)
CURRICULUM = [
    ("stage1s", 1, 30),  # train 1 s for 30 epochs
    ("stage2s", 2, 20),  # then 2 s for 20 more
    ("stage4s", 4, 10),  # then 4 s for 10 more
    ("stage8s", 8, 10),  # then 8 s for 10 more
    ("stage16s", 16, 10),  # then 16 s for 10 more
    ("stage32s", 32, 10),  # then 32 s for 10 more
]
SAMPLE_RATE = 16000
LATENT_DIM = 100
SEQ_LEN = SAMPLE_RATE * 4  # 4 seconds
BATCH_SIZE = 16
N_CRITIC = 5
EPOCHS = 10
LR = 5e-5
LAMBDA_GP = 10
CHECKPOINT_PATH = "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----- Dataset Loader -----
class AudioDataset(Dataset):
    def __init__(self, folder, seq_len=SEQ_LEN):
        self.seq_len = seq_len
        self.paths = glob(os.path.join(folder, "**/*.wav"), recursive=True)
        self.files = []
        for p in self.paths:
            audio = self._quick_load(p)
            if self._is_valid(audio):
                self.files.append(p)
            else:
                print(f"⚠️  Skipping bad file: {p}")
        self.resample = Resample(orig_freq=44100, new_freq=SAMPLE_RATE)

    def _quick_load(self, path):
        # load, resample, mono, pad/trim—same as your process_file but minimal
        wav, sr = sf.read(path)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != SAMPLE_RATE:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
        # pad/trim
        if len(wav) < self.seq_len:
            wav = np.pad(wav, (0, self.seq_len - len(wav)))
        else:
            wav = wav[: self.seq_len]
        return wav.astype(np.float32)

    def _is_valid(self, audio):
        # no NaNs/Infs, not silent, not clipped
        if not np.isfinite(audio).all():
            return False
        peak = np.max(np.abs(audio))
        if peak < 1e-5:  # essentially silent
            return False
        if peak > 1.0 + 1e-3:  # clipped
            return False
        return True

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        wav, sr = torchaudio.load(self.files[idx])
        if sr != SAMPLE_RATE:
            wav = self.resample(wav)
        wav = wav.mean(dim=0, keepdim=True)  # mono
        if wav.shape[-1] < self.seq_len:
            pad_len = self.seq_len - wav.shape[-1]
            wav = nn.functional.pad(wav, (0, pad_len))
        else:
            wav = wav[:, : self.seq_len]
        wav = wav / wav.abs().max()
        return wav


def build_generator_for(seq_len, latent_dim=100, base_channels=512):
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


# ----- Gradient Penalty -----
def compute_gradient_penalty(D, real, fake):
    alpha = torch.rand(real.size(0), 1, 1).to(DEVICE)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)
    fake_output = torch.ones_like(d_interpolates).to(DEVICE)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def find_latest_checkpoint(
    ckpt_dir, prefix="G_epoch_", suffix=".pt", latest_stage=None
):
    """
    Scan ckpt_dir for files like 'epoch_{N}.pt' and return the highest N
    and paths to both G and D checkpoints.
    """
    print(f"Searching for checkpoints in: {ckpt_dir}")
    print(f"Looking for pattern: {prefix}*{suffix}")

    pattern = os.path.join(ckpt_dir, f"{prefix}*{suffix}")
    files = glob(pattern)
    print(f"Found files: {files}")

    if not files:
        print("No checkpoint files found")
        return None, None, 0

    # extract epoch numbers
    epochs = []
    for f in files:
        print(f"Processing file: {f}")
        m = re.search(rf"{re.escape(prefix)}(\d+){re.escape(suffix)}", f)
        if m:
            epoch_num = int(m.group(1))
            epochs.append(epoch_num)
            print(f"  Extracted epoch number: {epoch_num}")
        else:
            print(f"  Failed to extract epoch number from: {f}")

    if not epochs:
        print("No valid epoch numbers found")
        return None, None, 0

    latest = max(epochs)
    print(f"Latest epoch: {latest}")

    path = os.path.join(ckpt_dir, f"{prefix}{latest}{suffix}")
    d_path = os.path.join(ckpt_dir, f"D_epoch_{latest}{suffix}")

    print(f"Generator checkpoint path: {path}")
    print(f"Discriminator checkpoint path: {d_path}")

    return path, d_path, latest


# ----- Training -----
def train_stage(stage_name, seq_len, epochs, resume_ckpt=None, is_latest=False):
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs("generated", exist_ok=True)

    dataset = AudioDataset("data_processed/split", seq_len=seq_len * SAMPLE_RATE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    # gen = build_generator_for(seq_len=16000, latent_dim=100)
    # print("Generator parameter count:", sum(p.numel() for p in gen.parameters()))
    generator = build_generator_for(seq_len, latent_dim=100).to(DEVICE)
    print(generator.parameters())
    discriminator = build_discriminator_for(seq_len).to(DEVICE)

    # DEBUG: how many params do we really have?
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"G params: {g_params:,}, D params: {d_params:,}")
    # optim_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.0, 0.9))
    # optim_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.0, 0.9))

    # 5) set up optimizers
    optim_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.0, 0.9))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.0, 0.9))

    # Attempt to resume
    if is_latest:
        latest_stage = stage_name
    else:
        latest_stage = None
    g_ckpt, d_ckpt, start_epoch = find_latest_checkpoint(CHECKPOINT_PATH, latest_stage)
    if g_ckpt:
        print(f"Resuming from epoch {start_epoch}")
        generator.load_state_dict(torch.load(g_ckpt))
        discriminator.load_state_dict(torch.load(d_ckpt))
    else:
        start_epoch = 0
        print("No checkpoint found — starting from scratch")
    if resume_ckpt:
        ckpt = torch.load(resume_ckpt)
        generator.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        optim_G.load_state_dict(ckpt["optim_G"])
        optim_D.load_state_dict(ckpt["optim_D"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs):
        print(f"Epoch {epoch}/{epochs} - Stage: {stage_name}, Seq Len: {seq_len}s")
        for i, real_audio in enumerate(tqdm(loader)):
            real_audio = real_audio.to(DEVICE)

            # --- Train Discriminator ---
            for _ in range(N_CRITIC):
                z = torch.randn(real_audio.size(0), LATENT_DIM, 1).to(DEVICE)
                fake_audio = generator(z).detach()

                d_real = discriminator(real_audio)
                d_fake = discriminator(fake_audio)

                gp = compute_gradient_penalty(discriminator, real_audio, fake_audio)
                loss_D = d_fake.mean() - d_real.mean() + LAMBDA_GP * gp

                optim_D.zero_grad()
                loss_D.backward()
                optim_D.step()

            # --- Train Generator ---
            z = torch.randn(real_audio.size(0), LATENT_DIM, 1).to(DEVICE)
            fake_audio = generator(z)
            loss_G = -discriminator(fake_audio).mean()

            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

        # Save checkpoint
        # Save checkpoint at end of this epoch
        torch.save(generator.state_dict(), f"{CHECKPOINT_PATH}/G_epoch_{epoch+1}.pt")
        torch.save(
            discriminator.state_dict(), f"{CHECKPOINT_PATH}/D_epoch_{epoch+1}.pt"
        )
        ckpt_path = f"{CHECKPOINT_PATH}/epoch_{epoch}.pth"
        torch.save(
            {
                "epoch": epoch,
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "optim_G": optim_G.state_dict(),
                "optim_D": optim_D.state_dict(),
            },
            ckpt_path,
        )
        print(f"Checkpoint saved to {ckpt_path}")
        # torch.save(generator.state_dict(), f"{CHECKPOINT_PATH}/G_epoch_{epoch}.pt")
        # torch.save(discriminator.state_dict(), f"{CHECKPOINT_PATH}/D_epoch_{epoch}.pt")

        # Generate example audio
        with torch.no_grad():
            test_z = torch.randn(1, LATENT_DIM, 1).to(DEVICE)
            sample = generator(test_z)[:, :, :SEQ_LEN].squeeze().cpu()
            torchaudio.save(
                f"generated/{stage_name}_sample_epoch_{epoch}.wav",
                sample.unsqueeze(0),
                SAMPLE_RATE,
            )

        print(
            f"[Epoch {epoch}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}"
        )

        # return ckpt_path


if __name__ == "__main__":
    last_ckpt = None

    for i, stage_name, seconds, epochs in CURRICULUM:
        print(f"→ Starting {stage_name}: {seconds}s for {epochs} epochs")
        if last_ckpt and os.path.exists(last_ckpt):
            resume = last_ckpt
        else:
            resume = None

        if i == len(CURRICULUM) - 1:
            is_latest = True
        else:
            is_latest = False
        last_ckpt = train_stage(
            stage_name, seconds, epochs, resume_ckpt=None, is_latest=is_latest
        )
        print(f"✓ Finished {stage_name}, checkpoint saved at {last_ckpt}")
