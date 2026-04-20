"""License-aware dataset registry for UFaceNet."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    """Dataset acquisition and usage metadata."""

    key: str
    tasks: tuple[str, ...]
    access: str
    root: str
    source: str
    notes: str


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "celeba_mask_hq": DatasetSpec(
        key="celeba_mask_hq",
        tasks=("parsing", "frec"),
        access="manual_license",
        root="data/raw/CelebAMask-HQ",
        source="https://github.com/switchablenorms/CelebAMask-HQ",
        notes="Requires following upstream download instructions and CelebA license constraints.",
    ),
    "300w": DatasetSpec(
        key="300w",
        tasks=("landmarks",),
        access="manual_license",
        root="data/raw/300W",
        source="https://ibug.doc.ic.ac.uk/resources/300-W/",
        notes="Requires acceptance of dataset terms.",
    ),
    "biwi": DatasetSpec(
        key="biwi",
        tasks=("headpose",),
        access="manual_license",
        root="data/raw/BIWI",
        source="https://data.vision.ee.ethz.ch/cvl/gfanelli/head_pose/head_forest.html",
        notes="BIWI head pose dataset has academic access constraints.",
    ),
    "celeba": DatasetSpec(
        key="celeba",
        tasks=("attributes",),
        access="manual_or_torchvision",
        root="data/raw/CelebA",
        source="https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html",
        notes="Large dataset with Google Drive mirrors that may require manual handling.",
    ),
    "fairface": DatasetSpec(
        key="fairface",
        tasks=("age", "gender", "race"),
        access="manual_license",
        root="data/raw/FairFace",
        source="https://github.com/joojs/fairface",
        notes="Follow upstream instructions and license.",
    ),
    "raf_db": DatasetSpec(
        key="raf_db",
        tasks=("expression",),
        access="manual_license",
        root="data/raw/RAF-DB",
        source="http://www.whdeng.cn/RAF/model1.html",
        notes="Requires request/registration.",
    ),
    "affectnet": DatasetSpec(
        key="affectnet",
        tasks=("expression",),
        access="manual_license",
        root="data/raw/AffectNet",
        source="http://mohammadmahoor.com/affectnet/",
        notes="Requires request/registration.",
    ),
    "ms1mv3": DatasetSpec(
        key="ms1mv3",
        tasks=("recognition",),
        access="manual_license",
        root="data/raw/MS1MV3",
        source="https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_",
        notes="Large identity dataset with redistribution constraints.",
    ),
    "lfw": DatasetSpec(
        key="lfw",
        tasks=("recognition", "frec"),
        access="public_sklearn",
        root="data/raw/lfw_home",
        source="https://vis-www.cs.umass.edu/lfw/",
        notes="Can be fetched with sklearn for smoke and verification experiments.",
    ),
    "now": DatasetSpec(
        key="now",
        tasks=("frec",),
        access="manual_license",
        root="data/raw/NoW",
        source="https://github.com/soubhiksanyal/now_evaluation",
        notes="Benchmark requires data access and official protocol.",
    ),
}
