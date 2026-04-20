"""Task consistency metrics for reconstructed faces."""

from __future__ import annotations

import torch
from torch import Tensor


def landmark_nme(pred: Tensor, target: Tensor, normalizer: Tensor | float = 1.0) -> Tensor:
    """Normalized mean landmark error for tensors shaped B x 68 x 2 or B x 136."""

    pred = pred.reshape(pred.shape[0], -1, 2)
    target = target.reshape(target.shape[0], -1, 2)
    errors = torch.linalg.norm(pred - target, dim=-1).mean(dim=-1)
    return errors / torch.as_tensor(normalizer, device=pred.device, dtype=pred.dtype)


def pose_mae(pred: Tensor, target: Tensor) -> Tensor:
    """Head-pose mean absolute error over pitch, yaw, and roll."""

    return torch.abs(pred - target).mean(dim=-1)


def parsing_pixel_agreement(pred_logits: Tensor, target_logits: Tensor) -> Tensor:
    """Pixel agreement between predicted segmentation label maps."""

    pred = pred_logits.argmax(dim=1)
    target = target_logits.argmax(dim=1)
    return (pred == target).float().mean(dim=(1, 2))
