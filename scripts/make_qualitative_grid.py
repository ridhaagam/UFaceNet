#!/usr/bin/env python
"""Make a simple qualitative grid from reconstruction outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a qualitative input/reconstruction grid.")
    parser.add_argument("--images", nargs="+", required=True, help="Image paths to place in the grid.")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels matching image paths.")
    parser.add_argument("--output", default="runs/qualitative_grid.png", help="Output PNG path.")
    parser.add_argument("--tile-size", type=int, default=224, help="Tile size in pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = args.labels or [Path(path).stem for path in args.images]
    if len(labels) != len(args.images):
        raise SystemExit("--labels must match --images length")
    tiles = []
    for path in args.images:
        tiles.append(Image.open(path).convert("RGB").resize((args.tile_size, args.tile_size)))
    canvas = Image.new("RGB", (args.tile_size * len(tiles), args.tile_size + 28), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, (tile, label) in enumerate(zip(tiles, labels)):
        x = idx * args.tile_size
        canvas.paste(tile, (x, 28))
        draw.text((x + 6, 6), label, fill=(0, 0, 0))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(output)


if __name__ == "__main__":
    main()
