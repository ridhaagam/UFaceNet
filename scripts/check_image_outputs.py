#!/usr/bin/env python
"""Fail if generated image outputs are blank or nearly constant."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check generated images for blank or near-constant outputs.")
    parser.add_argument("paths", nargs="+", help="Image files or directories to inspect.")
    parser.add_argument("--min-std", type=float, default=0.03, help="Minimum normalized pixel standard deviation.")
    parser.add_argument("--output", help="Optional JSON report path.")
    return parser.parse_args()


def iter_images(paths: list[str]) -> list[Path]:
    images: list[Path] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            images.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS))
        elif path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    if not images:
        raise FileNotFoundError("No image files found in requested paths")
    return images


def image_stats(path: Path) -> dict[str, object]:
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return {
        "path": str(path),
        "size": list(image.size),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def main() -> None:
    args = parse_args()
    report = [image_stats(path) for path in iter_images(args.paths)]
    failed = [item for item in report if item["std"] < args.min_std]
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failed:
        failed_paths = ", ".join(str(item["path"]) for item in failed)
        raise SystemExit(f"Blank or near-constant images below std {args.min_std}: {failed_paths}")


if __name__ == "__main__":
    main()
