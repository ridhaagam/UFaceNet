#!/usr/bin/env python
"""Validate the local UFaceNet Python environment."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
from pathlib import Path


MODULES = ["torch", "torchvision", "numpy", "PIL", "cv2", "yaml", "scipy", "ufacenet"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate imports and CUDA availability for UFaceNet.")
    parser.add_argument("--output", default="runs/environment_report.json", help="JSON report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report: dict[str, object] = {"python": platform.python_version(), "modules": {}}
    for module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
            report["modules"][module_name] = {"ok": True, "version": getattr(module, "__version__", "unknown")}
        except Exception as exc:
            report["modules"][module_name] = {"ok": False, "error": str(exc)}

    try:
        import torch

        report["cuda"] = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }
    except Exception as exc:
        report["cuda"] = {"available": False, "error": str(exc)}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
