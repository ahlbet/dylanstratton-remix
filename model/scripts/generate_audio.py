#!/usr/bin/env python3
import os
import torch
import torchaudio
from src.model import build_generator_for
from src.utils import find_latest_checkpoint
from src.config import SAMPLE_RATE, LATENT_DIM, CHECKPOINT_DIR, DEVICE, GENERATED_DIR

def main(duration_s=4):
    generator = build_generator_for(duration_s).to(DEVICE)
    ckpt, _ = find_latest_checkpoint()
    generator.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    generator.eval()
    z = torch.randn(1, LATENT_DIM, 1)
    audio = generator(z)[:, :, :int(duration_s * SAMPLE_RATE)].squeeze(0)
    os.makedirs(GENERATED_DIR, exist_ok=True)
    torchaudio.save(f"{GENERATED_DIR}/sample_{duration_s}s.wav", audio.unsqueeze(0), SAMPLE_RATE)

if __name__ == "__main__": main()