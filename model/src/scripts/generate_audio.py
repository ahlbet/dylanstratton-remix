#!/usr/bin/env python3
import os
import torch
import torchaudio
from model import build_generator_for
from utils import find_latest_checkpoint
from config import (
    SAMPLE_RATE,
    LATENT_DIM,
    CHECKPOINT_DIR,
    DEVICE,
    GENERATED_DIR,
    SEQ_LEN,
)


def main(duration_s=1):
    generator = build_generator_for(duration_s).to(DEVICE)
    g_ckpt, d_ckpt, _ = find_latest_checkpoint(CHECKPOINT_DIR)
    generator.load_state_dict(torch.load(g_ckpt, map_location=DEVICE))
    generator.eval()
    with torch.no_grad():
        z = torch.randn(1, LATENT_DIM, 1)
        sample = generator(z)[:, :, :SEQ_LEN].squeeze().cpu()
        os.makedirs(GENERATED_DIR, exist_ok=True)
        torchaudio.save(
            f"{GENERATED_DIR}/sample_{duration_s}s.wav",
            sample.unsqueeze(0),
            SAMPLE_RATE,
        )


if __name__ == "__main__":
    main()
