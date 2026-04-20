"""Losses for FRec reconstruction and consistency training."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def reconstruction_l1(pred: Tensor, target: Tensor) -> Tensor:
    """Pixel L1 loss for paired aligned face reconstruction."""

    return F.l1_loss(pred, target)


def identity_cosine_loss(pred_embeddings: Tensor, target_embeddings: Tensor) -> Tensor:
    """Identity loss from frozen face-recognition embeddings."""

    pred = F.normalize(pred_embeddings, dim=-1)
    target = F.normalize(target_embeddings, dim=-1)
    return (1.0 - (pred * target).sum(dim=-1)).mean()


def total_variation_loss(image: Tensor) -> Tensor:
    """Total variation regularizer for smoother generated maps."""

    vertical = torch.abs(image[:, :, 1:, :] - image[:, :, :-1, :]).mean()
    horizontal = torch.abs(image[:, :, :, 1:] - image[:, :, :, :-1]).mean()
    return vertical + horizontal


def landmark_consistency_loss(pred: Tensor, target: Tensor) -> Tensor:
    """L1 loss over normalized landmark coordinates."""

    return F.l1_loss(pred.reshape(pred.shape[0], -1, 2), target.reshape(target.shape[0], -1, 2))


def pose_consistency_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean absolute error over pitch, yaw, and roll."""

    return torch.abs(pred - target).mean()
