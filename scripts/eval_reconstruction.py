#!/usr/bin/env python
"""Evaluate paired reconstruction folders with smoke-safe metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ufacenet.metrics.fid import directory_fid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate UFaceNet reconstruction folders.")
    parser.add_argument("--real-dir", required=True, help="Directory with real aligned face crops.")
    parser.add_argument("--generated-dir", required=True, help="Directory with reconstructed or generated crops.")
    parser.add_argument("--output", default="runs/reconstruction_eval/metrics.json", help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = directory_fid(args.real_dir, args.generated_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
