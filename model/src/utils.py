import os
import glob
import re
import torch
from torch import autograd
from src.config import CHECKPOINT_DIR


def compute_gradient_penalty(D, real, fake, device, lambda_gp):
    batch_size = real.size(0)
    alpha = torch.rand(batch_size, 1, 1, device=device)
    interp = alpha * real + (1 - alpha) * fake
    interp.requires_grad_(True)
    d_interp = D(interp)
    grads = autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]
    gp = ((grads.view(batch_size, -1).norm(2, 1) - 1) ** 2).mean()
    return lambda_gp * gp


def find_latest_checkpoint(ckpt_dir=CHECKPOINT_DIR):
    files = glob.glob(os.path.join(ckpt_dir, "G_epoch_*.pt"))
    if not files:
        return None, 0
    epochs = [int(re.search(r"G_epoch_(\d+).pt", f).group(1)) for f in files]
    latest = max(epochs)
    return os.path.join(ckpt_dir, f"G_epoch_{latest}.pt"), latest
