"""
Initialize the package and provide convenient imports.
"""

# Version
__version__ = "0.1.0"

# Expose core components
from .config import (
    SAMPLE_RATE,
    LATENT_DIM,
    BATCH_SIZE,
    N_CRITIC,
    LR,
    LAMBDA_GP,
    CURRICULUM,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    CHECKPOINT_DIR,
    GENERATED_DIR,
    DEVICE,
)
from .dataset import AudioDataset
from .model import build_generator_for, build_discriminator_for
from .utils import compute_gradient_penalty, find_latest_checkpoint
from .training import train as main
