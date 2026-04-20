"""License-aware dataset and asset download helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .registry import DATASET_REGISTRY


def write_dataset_manifest(path: str | Path = "data/manifests/datasets.json") -> Path:
    """Write the dataset registry as a local manifest."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: spec.__dict__ for key, spec in DATASET_REGISTRY.items()}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def fetch_lfw(root: str | Path = "data/raw/lfw_home", min_faces_per_person: int = 20) -> dict[str, Any]:
    """Download LFW through sklearn for smoke recognition/FRec checks."""

    from sklearn.datasets import fetch_lfw_people

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    dataset = fetch_lfw_people(data_home=str(root), min_faces_per_person=min_faces_per_person, resize=0.5, color=True)
    return {"dataset": "lfw", "images": int(dataset.images.shape[0]), "root": str(root)}

