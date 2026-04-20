#!/usr/bin/env python
"""Validate UFaceNet dataset roots against the registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ufacenet.data.registry import DATASET_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate configured UFaceNet dataset roots.")
    parser.add_argument("--output", default="runs/dataset_report.json", help="JSON report path.")
    parser.add_argument("--blocker", default="runs/dataset_blocker.md", help="Markdown blocker path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {}
    blockers = ["# Dataset Validation Blockers", ""]
    for key, spec in DATASET_REGISTRY.items():
        root = Path(spec.root)
        image_count = sum(1 for path in root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}) if root.exists() else 0
        ok = root.exists() and image_count > 0
        report[key] = {
            "ok": ok,
            "root": str(root),
            "access": spec.access,
            "tasks": spec.tasks,
            "image_count": image_count,
            "source": spec.source,
            "notes": spec.notes,
        }
        if not ok:
            blockers.extend(
                [
                    f"## {key}",
                    f"- expected_root: `{root}`",
                    f"- access: {spec.access}",
                    f"- source: {spec.source}",
                    f"- next_step: {spec.notes}",
                    "",
                ]
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    blocker = Path(args.blocker)
    blocker.parent.mkdir(parents=True, exist_ok=True)
    blocker.write_text("\n".join(blockers), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
