import os
import torch

# Audio settings
SAMPLE_RATE = 16000  # Hz
SEQ_LEN = SAMPLE_RATE * 4  # 4 seconds

# Training hyperparameters
LATENT_DIM = 100
BATCH_SIZE = 16
N_CRITIC = 5
LR = 5e-5
LAMBDA_GP = 10

# Progressive curriculum: (stage_name, duration_s, epochs)
CURRICULUM = [
    ("stage1s", 1, 30),
    ("stage2s", 2, 20),
    ("stage4s", 4, 10),
]

# Paths
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed/split"
CHECKPOINT_DIR = "checkpoints"
GENERATED_DIR = "generated"

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
