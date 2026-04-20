#!/usr/bin/env python
"""Prepare LFW funneled images as a lightweight FRec training split."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a lightweight aligned-face FRec split from LFW.")
    parser.add_argument("--lfw-root", default="data/raw/lfw_home", help="Root produced by sklearn LFW download.")
    parser.add_argument("--output-root", default="data/processed/aligned_faces/train", help="Output image folder.")
    parser.add_argument("--max-images", type=int, default=1024, help="Maximum images to link or copy.")
    parser.add_argument("--copy", action="store_true", help="Copy files instead of creating symlinks.")
    return parser.parse_args()


def find_lfw_images(root: Path) -> list[Path]:
    candidates = list(root.rglob("lfw_funneled"))
    if not candidates:
        raise FileNotFoundError(f"Could not find lfw_funneled under {root}")
    image_root = candidates[0]
    return sorted(path for path in image_root.rglob("*.jpg"))


def main() -> None:
    args = parse_args()
    source_images = find_lfw_images(Path(args.lfw_root))[: args.max_images]
    if not source_images:
        raise SystemExit("No LFW images found. Run scripts/download_datasets.py --download-lfw first.")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for idx, source in enumerate(source_images):
        target = output_root / f"lfw_{idx:06d}.jpg"
        if target.exists():
            continue
        if args.copy:
            shutil.copy2(source, target)
        else:
            target.symlink_to(source.resolve())
    manifest = {"source": str(Path(args.lfw_root)), "output": str(output_root), "images": len(source_images)}
    (output_root.parent / "lfw_frec_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
