from pathlib import Path

import torch

from ufacenet import UFaceNet, UFaceNetConfig
from ufacenet.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path: Path):
    model = UFaceNet(UFaceNetConfig(image_size=64, backbone="tiny"))
    path = tmp_path / "model.pt"
    save_checkpoint(model, path, metadata={"test": True})
    report = load_checkpoint(model, path, strict=False)
    assert report["unexpected_keys"] == []


def test_dummy_facexformer_layout_loads_non_strict(tmp_path: Path):
    model = UFaceNet(UFaceNetConfig(image_size=64, backbone="tiny"))
    path = tmp_path / "legacy.pt"
    torch.save({"state_dict_backbone": {}}, path)
    report = load_checkpoint(model, path, strict=False)
    assert report["missing_keys"]
