"""Two-way transformer blocks for UFaceNet task-token decoding."""

from __future__ import annotations

import math
from typing import Type

import torch
from torch import Tensor, nn


class MLPBlock(nn.Module):
    """Feed-forward block used inside two-way attention layers."""

    def __init__(self, embedding_dim: int, mlp_dim: int, act: Type[nn.Module] = nn.GELU) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: Tensor) -> Tensor:
        return self.lin2(self.act(self.lin1(x)))


class LayerNorm2d(nn.Module):
    """LayerNorm over channel dimension for image tensors."""

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class Attention(nn.Module):
    """Multi-head attention with optional internal dimension downsampling."""

    def __init__(self, embedding_dim: int, num_heads: int, downsample_rate: int = 1) -> None:
        super().__init__()
        internal_dim = embedding_dim // downsample_rate
        if internal_dim % num_heads != 0:
            raise ValueError("num_heads must divide embedding_dim // downsample_rate")
        self.internal_dim = internal_dim
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embedding_dim, internal_dim)
        self.k_proj = nn.Linear(embedding_dim, internal_dim)
        self.v_proj = nn.Linear(embedding_dim, internal_dim)
        self.out_proj = nn.Linear(internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor) -> Tensor:
        batch, tokens, channels = x.shape
        x = x.reshape(batch, tokens, self.num_heads, channels // self.num_heads)
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        batch, heads, tokens, channels = x.shape
        return x.transpose(1, 2).reshape(batch, tokens, heads * channels)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q = self._separate_heads(self.q_proj(q))
        k = self._separate_heads(self.k_proj(k))
        v = self._separate_heads(self.v_proj(v))
        scale = math.sqrt(q.shape[-1])
        attn = torch.softmax((q @ k.transpose(-2, -1)) / scale, dim=-1)
        return self.out_proj(self._recombine_heads(attn @ v))


class TwoWayAttentionBlock(nn.Module):
    """Attention block that lets task tokens and face tokens update each other."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm4 = nn.LayerNorm(embedding_dim)
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor) -> tuple[Tensor, Tensor]:
        q = queries if self.skip_first_layer_pe else queries + query_pe
        queries = self.norm1(queries + self.self_attn(q=q, k=q, v=queries))

        queries = self.norm2(
            queries + self.cross_attn_token_to_image(q=queries + query_pe, k=keys + key_pe, v=keys)
        )
        queries = self.norm3(queries + self.mlp(queries))
        keys = self.norm4(keys + self.cross_attn_image_to_token(q=keys + key_pe, k=queries + query_pe, v=queries))
        return queries, keys


class TwoWayTransformer(nn.Module):
    """Transformer decoder for joint task-token and face-token processing."""

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            TwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                activation=activation,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=(idx == 0),
            )
            for idx in range(depth)
        )
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(self, image_embedding: Tensor, image_pe: Tensor, point_embedding: Tensor) -> tuple[Tensor, Tensor]:
        batch, channels, height, width = image_embedding.shape
        keys = image_embedding.flatten(2).permute(0, 2, 1)
        key_pe = image_pe.flatten(2).permute(0, 2, 1)
        queries = point_embedding

        for layer in self.layers:
            queries, keys = layer(queries=queries, keys=keys, query_pe=point_embedding, key_pe=key_pe)

        queries = self.norm_final_attn(
            queries + self.final_attn_token_to_image(q=queries + point_embedding, k=keys + key_pe, v=keys)
        )
        return queries, keys.reshape(batch, height * width, channels)
