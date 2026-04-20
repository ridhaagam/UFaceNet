"""Face embedding identity metrics."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def cosine_similarity(left: Tensor, right: Tensor) -> Tensor:
    """Cosine similarity between two batches of embeddings."""

    return (F.normalize(left, dim=-1) * F.normalize(right, dim=-1)).sum(dim=-1)
