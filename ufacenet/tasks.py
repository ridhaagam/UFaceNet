"""Task registry for unified face analysis and FRec outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class TaskSpec:
    """Static metadata for one UFaceNet task."""

    key: str
    short_name: str
    display_name: str
    output_key: str
    metric: str
    token_count: int = 1
    is_analysis: bool = True


TASK_REGISTRY: dict[str, TaskSpec] = {
    "parsing": TaskSpec("parsing", "FP", "Face Parsing", "segmentation", "mean F1", token_count=11),
    "landmarks": TaskSpec("landmarks", "LD", "Landmark Detection", "landmarks", "NME"),
    "headpose": TaskSpec("headpose", "HPE", "Head Pose Estimation", "headpose", "MAE"),
    "attributes": TaskSpec("attributes", "Attr", "Facial Attributes", "attributes", "accuracy"),
    "age": TaskSpec("age", "Age", "Age Estimation", "age", "MAE"),
    "gender": TaskSpec("gender", "Gen", "Gender Estimation", "gender", "accuracy"),
    "race": TaskSpec("race", "Race", "Race Estimation", "race", "accuracy"),
    "visibility": TaskSpec("visibility", "Vis", "Face Visibility", "visibility", "recall@80P"),
    "expression": TaskSpec("expression", "Exp", "Expression Recognition", "expression", "accuracy"),
    "recognition": TaskSpec("recognition", "FR", "Face Recognition", "recognition", "verification accuracy"),
    "frec": TaskSpec("frec", "FRec", "Face Reconstruction/Generation", "frec", "rFID/FID-face", is_analysis=False),
}

ALIASES = {
    "all": tuple(TASK_REGISTRY),
    "analysis": tuple(k for k, spec in TASK_REGISTRY.items() if spec.is_analysis),
    "frec": ("frec",),
    "reconstruction": ("frec",),
    "generation": ("frec",),
    "local": ("parsing", "landmarks", "headpose", "attributes", "age", "gender", "race", "visibility"),
}


def normalize_task_request(tasks: str | Iterable[str] | None) -> tuple[str, ...]:
    """Resolve task names and aliases into canonical task keys."""

    if tasks is None:
        return ALIASES["all"]
    if isinstance(tasks, str):
        raw_items = [item.strip().lower() for item in tasks.split(",") if item.strip()]
    else:
        raw_items = [str(item).strip().lower() for item in tasks if str(item).strip()]

    resolved: list[str] = []
    for item in raw_items:
        for key in ALIASES.get(item, (item,)):
            if key not in TASK_REGISTRY:
                raise KeyError(f"Unknown task '{item}'. Known tasks: {', '.join(TASK_REGISTRY)}")
            if key not in resolved:
                resolved.append(key)
    return tuple(resolved)


def analysis_task_keys() -> tuple[str, ...]:
    """Return canonical analysis task keys without FRec."""

    return ALIASES["analysis"]
