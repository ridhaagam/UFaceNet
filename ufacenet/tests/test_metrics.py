from pathlib import Path

import torch
from PIL import Image

from ufacenet.metrics.fid import directory_fid
from ufacenet.metrics.image import psnr, simple_ssim


def test_image_metrics_are_finite():
    left = torch.rand(1, 3, 16, 16)
    right = left.clone()
    assert torch.isfinite(psnr(left, right))
    assert torch.isfinite(simple_ssim(left, right))


def test_directory_fid_smoke(tmp_path: Path):
    real = tmp_path / "real"
    fake = tmp_path / "fake"
    real.mkdir()
    fake.mkdir()
    Image.new("RGB", (16, 16), (255, 0, 0)).save(real / "a.png")
    Image.new("RGB", (16, 16), (254, 0, 0)).save(fake / "a.png")
    metrics = directory_fid(real, fake)
    assert metrics["real_count"] == 1
    assert metrics["generated_count"] == 1
    assert metrics["fid"] >= 0
