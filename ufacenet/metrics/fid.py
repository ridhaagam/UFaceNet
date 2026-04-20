"""Frechet distance metrics for image folders and face crops."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from scipy import linalg


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _image_paths(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def _color_moment_features(path: Path, image_size: int = 64) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((image_size, image_size))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    q25 = np.quantile(arr.reshape(-1, 3), 0.25, axis=0)
    q75 = np.quantile(arr.reshape(-1, 3), 0.75, axis=0)
    return np.concatenate([means, stds, q25, q75], axis=0)


def image_directory_statistics(directory: str | Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute lightweight feature statistics for all images in a directory."""

    paths = _image_paths(directory)
    if not paths:
        raise FileNotFoundError(f"No images found under {directory}")
    features = np.stack([_color_moment_features(path) for path in paths], axis=0)
    cov = np.cov(features, rowvar=False) if len(paths) > 1 else np.eye(features.shape[1], dtype=np.float64) * 1e-6
    return features.mean(axis=0), cov, len(paths)


def frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    """Compute the Frechet distance between two Gaussian feature fits."""

    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1 @ sigma2)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def directory_fid(real_dir: str | Path, generated_dir: str | Path) -> dict[str, float | int | str]:
    """Compute a deterministic folder-level Frechet score for smoke evaluation."""

    real_mu, real_cov, real_count = image_directory_statistics(real_dir)
    gen_mu, gen_cov, gen_count = image_directory_statistics(generated_dir)
    return {
        "fid": frechet_distance(real_mu, real_cov, gen_mu, gen_cov),
        "real_count": real_count,
        "generated_count": gen_count,
        "feature_model": "color_moments_smoke",
    }
