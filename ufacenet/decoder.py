"""FaceX-style task decoder with first-class FRec tokens."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Type

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .transformer import LayerNorm2d, TwoWayTransformer


class MLP(nn.Module):
    """Small prediction MLP used by task heads."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int) -> None:
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList(nn.Linear(dims[idx], dims[idx + 1]) for idx in range(num_layers))

    def forward(self, x: Tensor) -> Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = F.relu(x, inplace=False)
        return x


class PositionEmbeddingRandom(nn.Module):
    """Random Fourier positional encoding for image-token grids."""

    def __init__(self, num_pos_feats: int = 128, scale: float = 1.0) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError("scale must be positive")
        self.register_buffer("gaussian_matrix", scale * torch.randn((2, num_pos_feats)))

    def _pe_encoding(self, coords: Tensor) -> Tensor:
        coords = 2 * coords - 1
        coords = 2 * math.pi * (coords @ self.gaussian_matrix)
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: tuple[int, int]) -> Tensor:
        height, width = size
        y = (torch.arange(height, device=self.gaussian_matrix.device, dtype=torch.float32) + 0.5) / height
        x = (torch.arange(width, device=self.gaussian_matrix.device, dtype=torch.float32) + 0.5) / width
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        return self._pe_encoding(torch.stack([grid_x, grid_y], dim=-1)).permute(2, 0, 1)


@dataclass(frozen=True)
class FaceXDecoderOutput:
    """Structured output from the FaceX decoder."""

    analysis: dict[str, Tensor]
    task_tokens: dict[str, Tensor]
    mask_tokens: Tensor
    refined_features: Tensor


class FaceXDecoder(nn.Module):
    """Unified token decoder for face analysis and FRec control."""

    def __init__(
        self,
        transformer_dim: int = 256,
        transformer: nn.Module | None = None,
        activation: Type[nn.Module] = nn.GELU,
        enable_geometry_tokens: bool = False,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.enable_geometry_tokens = enable_geometry_tokens
        self.transformer = transformer or TwoWayTransformer(
            depth=2,
            embedding_dim=transformer_dim,
            mlp_dim=2048,
            num_heads=8,
        )

        token_names = [
            "landmarks",
            "headpose",
            "attributes",
            "visibility",
            "age",
            "gender",
            "race",
            "expression",
            "recognition",
            "frec",
        ]
        if enable_geometry_tokens:
            token_names.extend(["geometry", "texture", "render"])
        self.token_names = tuple(token_names)
        self.task_tokens = nn.ParameterDict(
            {name: nn.Parameter(torch.randn(1, transformer_dim) * 0.02) for name in self.token_names}
        )
        self.mask_tokens = nn.Embedding(11, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.mask_hypernet = MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)

        self.landmarks_head = MLP(transformer_dim, transformer_dim, 136, 3)
        self.headpose_head = MLP(transformer_dim, transformer_dim, 3, 3)
        self.attributes_head = MLP(transformer_dim, transformer_dim, 40, 3)
        self.visibility_head = MLP(transformer_dim, transformer_dim, 29, 3)
        self.age_head = MLP(transformer_dim, transformer_dim, 8, 3)
        self.gender_head = MLP(transformer_dim, transformer_dim, 2, 3)
        self.race_head = MLP(transformer_dim, transformer_dim, 5, 3)
        self.expression_head = MLP(transformer_dim, transformer_dim, 7, 3)
        self.recognition_head = MLP(transformer_dim, transformer_dim, 512, 3)

    def _tokens(self, batch_size: int) -> tuple[Tensor, list[str]]:
        task_weights = [self.task_tokens[name] for name in self.token_names]
        weights = torch.cat(task_weights + [self.mask_tokens.weight], dim=0)
        return weights.unsqueeze(0).expand(batch_size, -1, -1), list(self.token_names)

    def forward(self, image_embeddings: Tensor, image_pe: Tensor) -> FaceXDecoderOutput:
        batch, channels, height, width = image_embeddings.shape
        tokens, token_names = self._tokens(batch)
        hs, refined = self.transformer(image_embeddings, image_pe.expand(batch, -1, -1, -1), tokens)

        token_count = len(token_names)
        task_token_values = {name: hs[:, idx, :] for idx, name in enumerate(token_names)}
        mask_token_values = hs[:, token_count:, :]

        refined_features = refined.transpose(1, 2).reshape(batch, channels, height, width)
        upscaled = self.output_upscaling(refined_features)
        hyper = self.mask_hypernet(mask_token_values)
        seg = (hyper @ upscaled.flatten(2)).reshape(batch, -1, upscaled.shape[-2], upscaled.shape[-1])

        recognition = F.normalize(self.recognition_head(task_token_values["recognition"]), dim=-1)
        analysis = {
            "landmarks": self.landmarks_head(task_token_values["landmarks"]),
            "headpose": self.headpose_head(task_token_values["headpose"]),
            "attributes": self.attributes_head(task_token_values["attributes"]),
            "visibility": self.visibility_head(task_token_values["visibility"]),
            "age": self.age_head(task_token_values["age"]),
            "gender": self.gender_head(task_token_values["gender"]),
            "race": self.race_head(task_token_values["race"]),
            "expression": self.expression_head(task_token_values["expression"]),
            "recognition": recognition,
            "segmentation": seg,
        }
        return FaceXDecoderOutput(
            analysis=analysis,
            task_tokens=task_token_values,
            mask_tokens=mask_token_values,
            refined_features=refined_features,
        )
