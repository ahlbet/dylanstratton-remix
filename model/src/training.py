import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import (
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
    SEQ_LEN,
)
from dataset import AudioDataset
from model import build_generator_for, build_discriminator_for
from utils import compute_gradient_penalty, find_latest_checkpoint


def train_stage(stage_name, seq_len, epochs, resume_ckpt=None, is_latest=False):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs("generated", exist_ok=True)

    dataset = AudioDataset(PROCESSED_DATA_DIR, seq_len=seq_len * SAMPLE_RATE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    generator = build_generator_for(seq_len, latent_dim=100).to(DEVICE)
    print(generator.parameters())
    discriminator = build_discriminator_for(seq_len).to(DEVICE)

    # 5) set up optimizers
    optim_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.0, 0.9))
    optim_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.0, 0.9))

    # Attempt to resume
    if is_latest:
        latest_stage = stage_name
    else:
        latest_stage = None
    g_ckpt, d_ckpt, start_epoch = find_latest_checkpoint(CHECKPOINT_DIR)
    if g_ckpt:
        print(f"Resuming from epoch {start_epoch}")
        generator.load_state_dict(torch.load(g_ckpt, map_location=DEVICE))
        discriminator.load_state_dict(torch.load(d_ckpt, map_location=DEVICE))
    else:
        start_epoch = 0
        print("No checkpoint found — starting from scratch")
    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location=DEVICE)
        generator.load_state_dict(ckpt["generator"])
        discriminator.load_state_dict(ckpt["discriminator"])
        optim_G.load_state_dict(ckpt["optim_G"])
        optim_D.load_state_dict(ckpt["optim_D"])
        start_epoch = ckpt["epoch"] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + start_epoch):
        print(
            f"Epoch {epoch}/{epochs + start_epoch} - Stage: {stage_name}, Seq Len: {seq_len}s"
        )
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
        torch.save(generator.state_dict(), f"{CHECKPOINT_DIR}/G_epoch_{epoch + 1}.pt")
        torch.save(
            discriminator.state_dict(), f"{CHECKPOINT_DIR}/D_epoch_{epoch + 1}.pt"
        )
        ckpt_path = f"{CHECKPOINT_DIR}/epoch_{epoch}.pth"
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


def train():
    last_ckpt = None

    for i, (stage_name, seconds, epochs) in enumerate(CURRICULUM):
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
