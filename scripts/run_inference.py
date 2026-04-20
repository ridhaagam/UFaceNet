#!/usr/bin/env python
"""Run one-pass UFaceNet inference on an image or random smoke input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image

from ufacenet import UFaceNet, UFaceNetConfig
from ufacenet.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run UFaceNet one-pass inference.")
    parser.add_argument("--image", help="Input image path. If omitted, uses random smoke input.")
    parser.add_argument("--checkpoint", help="Optional UFaceNet checkpoint.")
    parser.add_argument("--tasks", default="all", help="Comma-separated tasks or alias: all, analysis, frec, local.")
    parser.add_argument("--output-dir", default="runs/inference", help="Output directory.")
    parser.add_argument("--image-size", type=int, default=224, help="Input and FRec output size.")
    parser.add_argument("--backbone", choices=["tiny", "swin_b"], default="tiny", help="Backbone to instantiate.")
    parser.add_argument("--refiner", action="store_true", help="Enable the lightweight high-fidelity refiner interface.")
    return parser.parse_args()


def load_image(path: str | None, image_size: int) -> torch.Tensor:
    if path is None:
        return torch.rand(1, 3, image_size, image_size)
    transform = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor()])
    return transform(Image.open(path).convert("RGB")).unsqueeze(0)


def tensor_shapes(value: object) -> object:
    if torch.is_tensor(value):
        return list(value.shape)
    if isinstance(value, dict):
        return {key: tensor_shapes(item) for key, item in value.items() if item is not None}
    if isinstance(value, (list, tuple)):
        return [tensor_shapes(item) for item in value]
    return value


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = UFaceNetConfig(image_size=args.image_size, backbone=args.backbone, enable_refiner=args.refiner)
    model = UFaceNet(config).eval()
    load_report = None
    if args.checkpoint:
        load_report = load_checkpoint(model, args.checkpoint, strict=False)

    image = load_image(args.image, args.image_size)
    with torch.no_grad():
        outputs = model(image, tasks=args.tasks)

    frec = outputs.get("frec")
    if isinstance(frec, dict):
        save_image(frec["rgb"], output_dir / "frec_rgb.png")
        if frec.get("refined_rgb") is not None:
            save_image(frec["refined_rgb"], output_dir / "frec_refined_rgb.png")
        if frec.get("depth") is not None:
            save_image(torch.sigmoid(frec["depth"]), output_dir / "frec_depth.png")
        if frec.get("mask") is not None:
            save_image(frec["mask"], output_dir / "frec_mask.png")

    report = {"outputs": tensor_shapes(outputs), "checkpoint": load_report}
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
