#!/usr/bin/env python3
"""
Generate audio using the latest DiffWave checkpoint.
"""

import os
import torch

import torchaudio
from diffwave_model import DiffWave
from utils import find_latest_checkpoint
from config import SAMPLE_RATE, GENERATED_DIR, DEVICE

# Default DiffWave model config (should match training)
MODEL_CONFIG = {
    "timesteps": 1000,
    "beta_start": 1e-4,
    "beta_end": 0.02,
    "time_emb_dim": 128,
    "channels": 64,
    "num_res_blocks": 6,
}


def main(duration_s=4):
    # Instantiate model
    seq_len = int(SAMPLE_RATE * duration_s)
    model = DiffWave(MODEL_CONFIG).to(DEVICE)
    # Load latest checkpoint
    ckpt, _ = find_latest_checkpoint()
    if ckpt is None:
        raise FileNotFoundError(
            "No DiffWave checkpoint found in checkpoints directory."
        )
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()
    # Generate
    audio = model.inference(seq_len)
    # Save
    os.makedirs(GENERATED_DIR, exist_ok=True)
    out_path = os.path.join(GENERATED_DIR, f"diffwave_{duration_s}s.wav")
    torchaudio.save(out_path, audio.cpu(), SAMPLE_RATE)
    print(f"Saved DiffWave sample to {out_path}")


if __name__ == "__main__":
    main()
