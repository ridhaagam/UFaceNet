"""Image-folder dataset for paired face reconstruction starts."""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FaceImageFolder(Dataset[Tensor]):
    """Load aligned face images from a directory as tensors in [0, 1]."""

    def __init__(self, root: str | Path, image_size: int = 224) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Image folder does not exist: {self.root}")
        self.paths = sorted(path for path in self.root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        if not self.paths:
            raise FileNotFoundError(f"No images found under: {self.root}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image)
