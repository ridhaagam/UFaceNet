"""One-pass UFaceNet model for analysis and FRec outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .decoder import FaceXDecoder, PositionEmbeddingRandom
from .reconstruction import FRecHead
from .tasks import normalize_task_request


@dataclass
class UFaceNetConfig:
    """Configuration for UFaceNet architecture and output branches."""

    image_size: int = 224
    backbone: str = "tiny"
    pretrained_backbone: bool = False
    transformer_dim: int = 256
    decoder_depth: int = 2
    enable_frec: bool = True
    enable_geometry: bool = True
    enable_refiner: bool = False
    frec_input_skip_init: float = 0.85


class TinyFaceEncoder(nn.Module):
    """Small convolutional encoder used for smoke tests and CPU-safe training starts."""

    def __init__(self) -> None:
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.stage2 = self._stage(64, 128)
        self.stage3 = self._stage(128, 256)
        self.stage4 = self._stage(256, 512)
        self.channels = (64, 128, 256, 512)

    @staticmethod
    def _stage(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> list[Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        f4 = self.stage4(f3)
        return [f1, f2, f3, f4]


class SwinFaceEncoder(nn.Module):
    """Torchvision Swin-B encoder wrapper for UFaceNet multi-scale features."""

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        from torchvision.models import Swin_B_Weights, swin_b

        weights = Swin_B_Weights.IMAGENET1K_V1 if pretrained else None
        swin = swin_b(weights=weights)
        self.backbone = nn.Sequential(*(list(swin.children())[:-1]))
        self.target_layer_names = {"0.1", "0.3", "0.5", "0.7"}
        self.channels = (128, 256, 512, 1024)
        self._features: list[Tensor] = []
        for name, module in self.backbone.named_modules():
            if name in self.target_layer_names:
                module.register_forward_hook(self._save_feature)

    def _save_feature(self, _module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        self._features.append(output.permute(0, 3, 1, 2).contiguous())

    def forward(self, x: Tensor) -> list[Tensor]:
        self._features.clear()
        _ = self.backbone(x)
        if len(self._features) != 4:
            raise RuntimeError(f"Expected 4 Swin feature maps, got {len(self._features)}")
        return list(self._features)


class FeatureProjector(nn.Module):
    """Project and fuse encoder feature maps into the FaceX token dimension."""

    def __init__(self, in_channels: tuple[int, int, int, int], out_channels: int) -> None:
        super().__init__()
        self.projections = nn.ModuleList(nn.Conv2d(ch, out_channels, kernel_size=1) for ch in in_channels)
        self.fuse = nn.Conv2d(out_channels * len(in_channels), out_channels, kernel_size=1, bias=False)

    def forward(self, features: list[Tensor]) -> Tensor:
        target_size = features[0].shape[-2:]
        projected = []
        for feature, projection in zip(features, self.projections):
            x = projection(feature)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
            projected.append(x)
        return self.fuse(torch.cat(projected[::-1], dim=1))


class UFaceNet(nn.Module):
    """Unified model that returns analysis and FRec outputs from one forward pass."""

    def __init__(self, config: UFaceNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or UFaceNetConfig()
        if self.config.backbone == "tiny":
            self.encoder = TinyFaceEncoder()
        elif self.config.backbone == "swin_b":
            self.encoder = SwinFaceEncoder(pretrained=self.config.pretrained_backbone)
        else:
            raise ValueError(f"Unknown backbone '{self.config.backbone}'")

        self.projector = FeatureProjector(tuple(self.encoder.channels), self.config.transformer_dim)
        self.pe_layer = PositionEmbeddingRandom(self.config.transformer_dim // 2)
        self.face_decoder = FaceXDecoder(
            transformer_dim=self.config.transformer_dim,
            enable_geometry_tokens=self.config.enable_geometry,
        )
        self.frec_head = (
            FRecHead(
                dim=self.config.transformer_dim,
                image_size=self.config.image_size,
                geometry=self.config.enable_geometry,
                refiner=self.config.enable_refiner,
                input_skip_init=self.config.frec_input_skip_init,
            )
            if self.config.enable_frec
            else None
        )

    def forward(self, images: Tensor, tasks: str | list[str] | tuple[str, ...] | None = "all") -> dict[str, object]:
        requested = normalize_task_request(tasks)
        features = self.encoder(images)
        fused = self.projector(features)
        image_pe = self.pe_layer((fused.shape[-2], fused.shape[-1])).unsqueeze(0).to(fused)
        decoded = self.face_decoder(fused, image_pe)

        analysis = self._select_analysis(decoded.analysis, requested)
        output: dict[str, object] = {
            "requested_tasks": requested,
            "analysis": analysis,
            "tokens": {name: token for name, token in decoded.task_tokens.items() if name in requested or name == "frec"},
        }
        if "frec" in requested:
            if self.frec_head is None:
                raise RuntimeError("FRec was requested but enable_frec=False")
            frec = self.frec_head(decoded.task_tokens["frec"], decoded.refined_features, source_image=images)
            output["frec"] = {
                "rgb": frec.rgb,
                "refined_rgb": frec.refined_rgb,
                "depth": frec.depth,
                "normals": frec.normals,
                "mask": frec.mask,
                "latent": frec.latent,
            }
        return output

    @staticmethod
    def _select_analysis(outputs: dict[str, Tensor], requested: tuple[str, ...]) -> dict[str, Tensor]:
        mapping = {
            "parsing": "segmentation",
            "landmarks": "landmarks",
            "headpose": "headpose",
            "attributes": "attributes",
            "age": "age",
            "gender": "gender",
            "race": "race",
            "visibility": "visibility",
            "expression": "expression",
            "recognition": "recognition",
        }
        return {key: outputs[value] for key, value in mapping.items() if key in requested}
