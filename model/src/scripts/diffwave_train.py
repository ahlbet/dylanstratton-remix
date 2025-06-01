#!/usr/bin/env python3
import argparse
from diffwave_training import train_diffwave


DEFAULT_CHECKPOINT_DIR = "checkpoints"


def main():
    parser = argparse.ArgumentParser(
        prog="train-diffwave", description="Train or resume your DiffWave model"
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        default=DEFAULT_CHECKPOINT_DIR,
        help="Directory where new checkpoints will be saved",
    )
    parser.add_argument(
        "--resume",
        "-r",
        default=None,
        help="Path to a specific checkpoint to resume from (overrides auto-find)",
    )
    args = parser.parse_args()
    print(f"Checkpoint directory: {args.out_dir}")

    train_diffwave(checkpoint_dir=args.out_dir)


if __name__ == "__main__":
    main()
