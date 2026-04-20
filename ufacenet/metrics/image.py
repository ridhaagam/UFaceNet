"""Image reconstruction metrics."""

from __future__ import annotations

import torch
from torch import Tensor


def psnr(pred: Tensor, target: Tensor, max_value: float = 1.0) -> Tensor:
    """Peak signal-to-noise ratio for tensors in a common value range."""

    mse = torch.mean((pred - target) ** 2).clamp_min(1e-12)
    return 20 * torch.log10(torch.tensor(max_value, device=pred.device)) - 10 * torch.log10(mse)


def simple_ssim(pred: Tensor, target: Tensor, max_value: float = 1.0) -> Tensor:
    """Global SSIM approximation for smoke tests and quick validation."""

    c1 = (0.01 * max_value) ** 2
    c2 = (0.03 * max_value) ** 2
    pred_mu = pred.mean()
    target_mu = target.mean()
    pred_var = pred.var(unbiased=False)
    target_var = target.var(unbiased=False)
    cov = ((pred - pred_mu) * (target - target_mu)).mean()
    return ((2 * pred_mu * target_mu + c1) * (2 * cov + c2)) / (
        (pred_mu**2 + target_mu**2 + c1) * (pred_var + target_var + c2)
    )
