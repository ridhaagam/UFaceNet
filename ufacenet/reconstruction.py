"""FRec reconstruction and high-fidelity refinement modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ReconstructionOutput:
    """Outputs from the FRec branch."""

    rgb: Tensor
    refined_rgb: Tensor | None
    depth: Tensor | None
    normals: Tensor | None
    mask: Tensor | None
    latent: Tensor


class ReconstructionConsistencyBlock(nn.Module):
    """Fuse FRec token and pooled face features before decoding."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, frec_token: Tensor, refined_features: Tensor) -> Tensor:
        pooled = refined_features.flatten(2).mean(dim=-1)
        return self.proj(torch.cat([frec_token, pooled], dim=-1))


class RGBReconstructionDecoder(nn.Module):
    """Decode UFaceNet features into aligned RGB and optional geometry maps."""

    def __init__(self, dim: int = 256, image_size: int = 224, geometry: bool = True) -> None:
        super().__init__()
        self.image_size = image_size
        self.geometry = geometry
        self.token_to_bias = nn.Linear(dim, dim)
        self.shared_up = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
            nn.GELU(),
        )
        self.rgb_head = nn.Conv2d(dim // 4, 3, kernel_size=3, padding=1)
        if geometry:
            self.depth_head = nn.Conv2d(dim // 4, 1, kernel_size=3, padding=1)
            self.normal_head = nn.Conv2d(dim // 4, 3, kernel_size=3, padding=1)
            self.mask_head = nn.Conv2d(dim // 4, 1, kernel_size=3, padding=1)

    def forward(self, refined_features: Tensor, frec_latent: Tensor) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
        bias = self.token_to_bias(frec_latent)[:, :, None, None]
        conditioned = refined_features + bias
        shared = self.shared_up(conditioned)
        rgb = torch.sigmoid(self.rgb_head(shared))
        rgb = F.interpolate(rgb, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        if not self.geometry:
            return rgb, None, None, None
        depth = F.interpolate(self.depth_head(shared), size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        normals = F.normalize(self.normal_head(shared), dim=1)
        normals = F.interpolate(normals, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        mask = torch.sigmoid(self.mask_head(shared))
        mask = F.interpolate(mask, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return rgb, depth, normals, mask


class ConditionedRefiner(nn.Module):
    """Small refiner interface that can be replaced by VAE, VQ, or diffusion backends."""

    def __init__(self, dim: int = 256, enabled: bool = False) -> None:
        super().__init__()
        self.enabled = enabled
        self.to_scale = nn.Linear(dim, 3)
        self.refine = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
        )

    def forward(self, rgb: Tensor, frec_latent: Tensor) -> Tensor | None:
        if not self.enabled:
            return None
        scale = torch.tanh(self.to_scale(frec_latent))[:, :, None, None]
        return torch.clamp(rgb + 0.05 * torch.tanh(self.refine(rgb)) + 0.02 * scale, 0.0, 1.0)


class FRecHead(nn.Module):
    """One-pass FRec branch conditioned by FaceX task tokens and face features."""

    def __init__(self, dim: int = 256, image_size: int = 224, geometry: bool = True, refiner: bool = False) -> None:
        super().__init__()
        self.consistency = ReconstructionConsistencyBlock(dim)
        self.decoder = RGBReconstructionDecoder(dim=dim, image_size=image_size, geometry=geometry)
        self.refiner = ConditionedRefiner(dim=dim, enabled=refiner)

    def forward(self, frec_token: Tensor, refined_features: Tensor) -> ReconstructionOutput:
        latent = self.consistency(frec_token, refined_features)
        rgb, depth, normals, mask = self.decoder(refined_features, latent)
        refined_rgb = self.refiner(rgb, latent)
        return ReconstructionOutput(rgb=rgb, refined_rgb=refined_rgb, depth=depth, normals=normals, mask=mask, latent=latent)
