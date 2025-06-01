import os
import torch
from torch.utils.data import DataLoader
import torchaudio
from config import (
    CHECKPOINT_DIR,
    GENERATED_DIR,
    CURRICULUM,
    DEVICE,
    SAMPLE_RATE,
    BATCH_SIZE,
    DR_LR,
    PROCESSED_DATA_DIR,
)
from dataset import AudioDataset
from diffwave_model import DiffWave
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from glob import glob
import re


def train_diffwave(checkpoint_dir=None):
    if checkpoint_dir is not None:
        global CHECKPOINT_DIR
        CHECKPOINT_DIR = checkpoint_dir
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join("logs", CHECKPOINT_DIR))

    model_config = {
        "timesteps": 1000,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "time_emb_dim": 128,
        "channels": 64,
        "num_res_blocks": 6,
    }
    model = DiffWave(model_config).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scaler = GradScaler()

    for stage_name, duration_s, epochs in CURRICULUM:
        seq_len = int(duration_s * SAMPLE_RATE)
        # Determine start epoch for this stage by scanning existing checkpoints
        pattern = os.path.join(CHECKPOINT_DIR, f"DW_{stage_name}_e*.pt")
        ckpts = glob(pattern)
        if ckpts:
            done = (
                max(
                    int(
                        re.search(
                            rf"DW_{stage_name}_e(\d+).pt", os.path.basename(p)
                        ).group(1)
                    )
                    for p in ckpts
                )
                + 1
            )
            print(f"Resuming DiffWave {stage_name} from epoch {done}")
        else:
            done = 0
        dataset = AudioDataset(PROCESSED_DATA_DIR, seq_len)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        for epoch in tqdm(range(done, epochs + done), desc=f"Stage {stage_name}"):
            ckpt = os.path.join(CHECKPOINT_DIR, f"DW_{stage_name}_e{epoch}.pt")
            saved_mel = None
            for clean, mel in tqdm(loader, desc=f"{stage_name} Epoch {epoch}"):
                clean, mel = clean.to(DEVICE), mel.to(DEVICE)
                saved_mel = mel
                loss = model.compute_loss(clean, mel)
                optimizer.zero_grad()
                with autocast(device_type=DEVICE.type, dtype=torch.float16):
                    # Forward pass
                    loss = model.compute_loss(clean, mel)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if epoch % 2 == 0 or epoch == epochs - 1:
                print(f"Saving checkpoint to {ckpt}")
                with torch.no_grad():
                    sample = model.inference(seq_len, saved_mel)
                # Ensure we have [channels, samples] for saving
                audio_save = sample.cpu().squeeze().view(1, -1)  # Force to [1, seq_len]

                # Save WAV file - keep original values
                torchaudio.save(
                    os.path.join(
                        GENERATED_DIR, CHECKPOINT_DIR, f"dw_{stage_name}_e{epoch}.wav"
                    ),
                    audio_save,
                    SAMPLE_RATE,
                )

                # For tensorboard, normalize to [-1, 1] range
                audio_norm = audio_save.clamp(-1, 1)  # Ensure values are in [-1, 1]
                writer.add_audio(
                    os.path.join(CHECKPOINT_DIR, f"sample/{stage_name}_epoch_{epoch}"),
                    audio_norm,
                    global_step=epoch,
                    sample_rate=SAMPLE_RATE,
                )
            # Save checkpoint and generate sample
            torch.save(model.state_dict(), ckpt)

            print(f"Stage {stage_name} Epoch {epoch}: loss={loss.item():.4f}")
            writer.add_scalar(f"loss/{stage_name}", loss.item(), epoch)
            writer.flush()

    writer.close()


if __name__ == "__main__":
    train_diffwave()
    print("Training complete. Checkpoints and samples saved.")
