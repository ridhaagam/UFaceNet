import sys

import torch

from ufacenet import UFaceNet, UFaceNetConfig


def test_ufacenet_import_does_not_import_external_baseline():
    assert "facexformer" not in sys.modules


def test_one_pass_all_tasks_with_frec():
    model = UFaceNet(UFaceNetConfig(image_size=64, backbone="tiny", enable_geometry=True, enable_refiner=True))
    image = torch.rand(1, 3, 64, 64)
    with torch.no_grad():
        output = model(image, tasks="all")
    assert "analysis" in output
    assert "frec" in output
    assert output["analysis"]["landmarks"].shape == (1, 136)
    assert output["analysis"]["recognition"].shape == (1, 512)
    assert output["frec"]["rgb"].shape == (1, 3, 64, 64)
    assert output["frec"]["refined_rgb"].shape == (1, 3, 64, 64)
    assert output["frec"]["depth"].shape == (1, 1, 64, 64)


def test_frec_disabled_raises_when_requested():
    model = UFaceNet(UFaceNetConfig(image_size=64, backbone="tiny", enable_frec=False))
    image = torch.rand(1, 3, 64, 64)
    try:
        model(image, tasks="frec")
    except RuntimeError as exc:
        assert "enable_frec=False" in str(exc)
    else:
        raise AssertionError("Expected FRec disabled runtime error")
