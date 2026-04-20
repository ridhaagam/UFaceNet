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


def download_facexformer_checkpoint(target_dir: str | Path = "checkpoints/facexformer") -> dict[str, str]:
    """Download the upstream FaceXFormer checkpoint from Hugging Face."""

    from huggingface_hub import hf_hub_download

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id="kartiknarayan/facexformer", filename="ckpts/model.pt", local_dir=str(target_dir))
    return {"asset": "facexformer_checkpoint", "path": path}
