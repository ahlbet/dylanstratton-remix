import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.config import (
    CHECKPOINT_DIR,
    GENERATED_DIR,
    CURRICULUM,
    DEVICE,
    LR,
    SAMPLE_RATE,
    BATCH_SIZE,
    N_CRITIC,
    LATENT_DIM,
    LAMBDA_GP,
    PROCESSED_DATA_DIR,
)
from src.dataset import AudioDataset
from src.model import build_generator_for, build_discriminator_for
from src.utils import compute_gradient_penalty, find_latest_checkpoint


def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)

    # resume if available
    ckpt_path, start_epoch = find_latest_checkpoint()
    generator = build_generator_for(CURRICULUM[-1][1]).to(DEVICE)
    discriminator = build_discriminator_for(CURRICULUM[-1][1]).to(DEVICE)
    optim_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.0, 0.9))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.0, 0.9))
    if ckpt_path:
        state = torch.load(ckpt_path)
        generator.load_state_dict(state)

    for stage_name, duration_s, epochs in CURRICULUM:
        seq_len = int(duration_s * SAMPLE_RATE)
        dataset = AudioDataset(PROCESSED_DATA_DIR, seq_len)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        for epoch in range(start_epoch, start_epoch + epochs):
            for real in tqdm(loader, desc=f"{stage_name}"):
                real = real.unsqueeze(1).to(DEVICE)
                # D steps
                for _ in range(N_CRITIC):
                    z = torch.randn(real.size(0), LATENT_DIM, 1, device=DEVICE)
                    fake = generator(z).detach()
                    loss_D = discriminator(fake).mean() - discriminator(real).mean()
                    gp = compute_gradient_penalty(
                        discriminator, real, fake, DEVICE, LAMBDA_GP
                    )
                    (loss_D + gp).backward()
                    optim_D.step()
                    optim_D.zero_grad()
                # G step
                z = torch.randn(BATCH_SIZE, LATENT_DIM, 1, device=DEVICE)
                loss_G = -discriminator(generator(z)).mean()
                loss_G.backward()
                optim_G.step()
                optim_G.zero_grad()
        # save and generate
        ckpt = os.path.join(CHECKPOINT_DIR, f"G_epoch_{epoch + 1}.pt")
        torch.save(generator.state_dict(), ckpt)
        sample = generator(torch.randn(1, LATENT_DIM, 1, device=DEVICE))
        torchaudio.save(
            f"{GENERATED_DIR}/{stage_name}_epoch{epoch + 1}.wav",
            sample[:, :, :seq_len].cpu().squeeze(0),
            SAMPLE_RATE,
        )
        start_epoch += epochs
