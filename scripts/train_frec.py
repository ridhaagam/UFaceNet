#!/usr/bin/env python
"""Train the FRec branch for paired aligned-face reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import yaml

from ufacenet import UFaceNet, UFaceNetConfig
from ufacenet.checkpoint import save_checkpoint
from ufacenet.data import FaceImageFolder
from ufacenet.losses import reconstruction_l1, total_variation_loss
from ufacenet.metrics.image import psnr, simple_ssim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train UFaceNet FRec on an aligned image folder.")
    parser.add_argument("--config", default="configs/ufacenet_frec_frozen.yaml", help="Training config YAML.")
    parser.add_argument("--data-root", help="Override image folder root.")
    parser.add_argument("--output-dir", default="runs/frec_train", help="Output directory.")
    parser.add_argument("--max-steps", type=int, help="Override training.max_steps from config.")
    parser.add_argument("--smoke", action="store_true", help="Run a two-step synthetic smoke training loop.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def freeze_non_frec(model: UFaceNet) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("frec_head") or "task_tokens.frec" in name


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    model_cfg = cfg.get("model", {})
    image_size = int(model_cfg.get("image_size", 224))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UFaceNet(
        UFaceNetConfig(
            image_size=image_size,
            backbone=str(model_cfg.get("backbone", "tiny")),
            enable_geometry=bool(model_cfg.get("enable_geometry", True)),
            enable_refiner=bool(model_cfg.get("enable_refiner", False)),
            frec_input_skip_init=float(model_cfg.get("frec_input_skip_init", 0.85)),
        )
    ).to(device)
    if bool(train_cfg.get("freeze_non_frec", True)):
        freeze_non_frec(model)

    if args.smoke:
        batches = [torch.rand(2, 3, image_size, image_size) for _ in range(2)]
        loader = None
    else:
        data_root = args.data_root or str(cfg.get("data", {}).get("train_root", ""))
        if not data_root:
            raise SystemExit("Missing data root. Pass --data-root or set data.train_root in config.")
        dataset = FaceImageFolder(data_root, image_size=image_size)
        loader = DataLoader(dataset, batch_size=int(train_cfg.get("batch_size", 4)), shuffle=True, num_workers=0)
        batches = []

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    max_steps = int(args.max_steps if args.max_steps is not None else train_cfg.get("max_steps", 100))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[dict[str, float]] = []
    model.train()
    data_iter = iter(loader) if loader is not None else None
    for step in range(max_steps if not args.smoke else 2):
        if data_iter is None:
            images = batches[step % len(batches)].to(device)
        else:
            try:
                images = next(data_iter).to(device)
            except StopIteration:
                data_iter = iter(loader)
                images = next(data_iter).to(device)
        outputs = model(images, tasks="frec")
        rgb = outputs["frec"]["rgb"]
        loss = reconstruction_l1(rgb, images) + 0.01 * total_variation_loss(rgb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        metrics.append(
            {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                "psnr": float(psnr(rgb.detach(), images).cpu()),
                "ssim": float(simple_ssim(rgb.detach(), images).cpu()),
            }
        )

    model.eval()
    with torch.no_grad():
        sample = batches[0].to(device) if args.smoke else images.detach()
        sample_out = model(sample, tasks="frec")["frec"]["rgb"].detach().cpu()
    sample_cpu = sample.detach().cpu()
    save_image(sample_cpu, output_dir / "sample_input.png")
    save_image(sample_out, output_dir / "sample_reconstruction.png")
    paired = torch.stack([sample_cpu, sample_out], dim=1).flatten(0, 1)
    save_image(make_grid(paired, nrow=2), output_dir / "sample_input_reconstruction_grid.png")
    save_checkpoint(model, output_dir / "model.pt", metadata={"config": cfg, "smoke": args.smoke})
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics[-1], indent=2))


if __name__ == "__main__":
    main()
