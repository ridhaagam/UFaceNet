#!/usr/bin/env python
"""Record current status of original FaceXFormer task evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ufacenet.tasks import TASK_REGISTRY


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a task-evaluation readiness report.")
    parser.add_argument("--output", default="runs/original_task_eval/status.json", help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    report = {}
    for key, spec in TASK_REGISTRY.items():
        if key == "frec":
            continue
        report[key] = {
            "short_name": spec.short_name,
            "metric": spec.metric,
            "status": "blocked_until_dataset_and_protocol_are_configured",
        }
    output = Path(parse_args().output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
