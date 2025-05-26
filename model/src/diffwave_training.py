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


def train_diffwave():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(GENERATED_DIR, exist_ok=True)

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

    for stage_name, duration_s, epochs in CURRICULUM:
        seq_len = int(duration_s * SAMPLE_RATE)
        dataset = AudioDataset(PROCESSED_DATA_DIR, seq_len)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
        for epoch in tqdm(range(epochs), desc=f"Stage {stage_name}"):
            for clean in tqdm(loader, desc=f"{stage_name} Epoch {epoch}"):
                clean = clean.to(DEVICE)
                loss = model.compute_loss(clean)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # Save checkpoint and generate sample
            ckpt = os.path.join(CHECKPOINT_DIR, f"DW_{stage_name}_e{epoch}.pt")
            torch.save(model.state_dict(), ckpt)
            sample = model.inference(seq_len)
            torchaudio.save(
                os.path.join(GENERATED_DIR, f"dw_{stage_name}_e{epoch}.wav"),
                sample.cpu(),
                SAMPLE_RATE,
            )
            print(f"Stage {stage_name} Epoch {epoch}: loss={loss.item():.4f}")


if __name__ == "__main__":
    train_diffwave()
