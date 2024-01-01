from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .block import VisualEncoderBlock

__all__ = ["VisualEncoder"]


@dataclass(eq=False)
class VisualEncoder(nn.Module):
    dim: int
    patch_size: int
    image_size: int
    num_layers: int
    num_heads: int
    embedding_dim: int
    pooling: str

    def __post_init__(self):
        super().__init__()

        seq_len = (self.image_size // self.patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, self.dim, self.patch_size, self.patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, self.dim) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))

        self.blocks = nn.Sequential(
            *[
                VisualEncoderBlock(self.dim, self.num_heads)
                for _ in range(self.num_layers)
            ]
        )

        self.norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.embedding_projection = nn.Linear(self.dim, self.embedding_dim, bias=False)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x).flatten(start_dim=2).transpose(2, 1)
        x = x + self.pos_embed
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.blocks(x)

        return self.norm(x)

    def forward_embedding(self, x: Tensor) -> Tensor:
        if self.pooling == "cls":
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        return self.embedding_projection(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        embeddings = self.forward_embedding(features)
        return features, embeddings
