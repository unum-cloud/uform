from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor

from ..network_layers import MLP, Attention, LayerScale

__all__ = ["VisualEncoderBlock"]


@dataclass(eq=False)
class VisualEncoderBlock(nn.Module):
    dim: int
    num_heads: int

    def __post_init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(self.dim, eps=1e-6)
        self.attn = Attention(self.dim, self.num_heads)
        self.ls1 = LayerScale(self.dim)

        self.norm2 = nn.LayerNorm(self.dim, eps=1e-6)
        self.mlp = MLP(self.dim)
        self.ls2 = LayerScale(self.dim)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x
