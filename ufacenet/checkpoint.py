"""Checkpoint helpers for UFaceNet weights."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract a PyTorch state dict from common checkpoint layouts."""

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(isinstance(k, str) for k in checkpoint):
            tensors = {k: v for k, v in checkpoint.items() if torch.is_tensor(v)}
            if tensors:
                return tensors
    raise ValueError("Could not find a state dict in checkpoint")


def load_checkpoint(model: nn.Module, path: str | Path, strict: bool = False) -> dict[str, list[str]]:
    """Load UFaceNet weights and report key mismatches."""

    checkpoint = torch.load(Path(path), map_location="cpu")
    state = extract_state_dict(checkpoint)
    result = model.load_state_dict(state, strict=strict)
    return {"missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys)}


def save_checkpoint(model: nn.Module, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    """Save model weights with lightweight metadata."""

    payload = {"state_dict": model.state_dict(), "metadata": metadata or {}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
