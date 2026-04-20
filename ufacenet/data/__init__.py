"""Dataset registry and simple image datasets for UFaceNet."""

from .image_folder import FaceImageFolder
from .registry import DATASET_REGISTRY, DatasetSpec

__all__ = ["DATASET_REGISTRY", "DatasetSpec", "FaceImageFolder"]
