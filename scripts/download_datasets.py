#!/usr/bin/env python
"""Prepare license-aware dataset manifests and optional public downloads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ufacenet.data.download import fetch_lfw, write_dataset_manifest
from ufacenet.data.registry import DATASET_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare UFaceNet datasets and public startup assets.")
    parser.add_argument("--manifest", default="data/manifests/datasets.json", help="Path for the dataset manifest JSON.")
    parser.add_argument("--download-lfw", action="store_true", help="Download LFW through sklearn for smoke checks.")
    parser.add_argument("--blocker", default="runs/dataset_blocker.md", help="Where to write manual-access blockers.")
    return parser.parse_args()


def write_blocker(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Dataset Access Blockers",
        "",
        "These datasets are required for full UFaceNet training/evaluation but cannot be downloaded automatically without licenses, requests, or manual terms acceptance.",
        "",
    ]
    for spec in DATASET_REGISTRY.values():
        if spec.access.startswith("manual"):
            lines.extend(
                [
                    f"## {spec.key}",
                    f"- tasks: {', '.join(spec.tasks)}",
                    f"- expected_root: `{spec.root}`",
                    f"- source: {spec.source}",
                    f"- reason: {spec.notes}",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    args = parse_args()
    results: list[dict[str, object]] = [{"manifest": str(write_dataset_manifest(args.manifest))}]
    if args.download_lfw:
        results.append(fetch_lfw())
    blocker = write_blocker(args.blocker)
    results.append({"manual_access_blocker": str(blocker)})
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
