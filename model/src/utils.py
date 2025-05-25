import os
from glob import glob
import re
import torch
from torch import autograd
from config import CHECKPOINT_DIR, DEVICE


# ----- Gradient Penalty -----
def compute_gradient_penalty(D, real, fake):
    alpha = torch.rand(real.size(0), 1, 1).to(DEVICE)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)

    d_interpolates = D(interpolates)
    fake_output = torch.ones_like(d_interpolates).to(DEVICE)

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp


def find_latest_checkpoint(
    ckpt_dir, prefix="G_epoch_", suffix=".pt", latest_stage=None
):
    """
    Scan ckpt_dir for files like 'epoch_{N}.pt' and return the highest N
    and paths to both G and D checkpoints.
    """
    print(f"Searching for checkpoints in: {ckpt_dir}")
    print(f"Looking for pattern: {prefix}*{suffix}")

    pattern = os.path.join(ckpt_dir, f"{prefix}*{suffix}")
    files = glob(pattern)
    print(f"Found files: {files}")

    if not files:
        print("No checkpoint files found")
        return None, None, 0

    # extract epoch numbers
    epochs = []
    for f in files:
        print(f"Processing file: {f}")
        m = re.search(rf"{re.escape(prefix)}(\d+){re.escape(suffix)}", f)
        if m:
            epoch_num = int(m.group(1))
            epochs.append(epoch_num)
            print(f"  Extracted epoch number: {epoch_num}")
        else:
            print(f"  Failed to extract epoch number from: {f}")

    if not epochs:
        print("No valid epoch numbers found")
        return None, None, 0

    latest = max(epochs)
    print(f"Latest epoch: {latest}")

    path = os.path.join(ckpt_dir, f"{prefix}{latest}{suffix}")
    d_path = os.path.join(ckpt_dir, f"D_epoch_{latest}{suffix}")

    print(f"Generator checkpoint path: {path}")
    print(f"Discriminator checkpoint path: {d_path}")

    return path, d_path, latest
