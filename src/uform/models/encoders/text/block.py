from dataclasses import dataclass
from typing import Optional

import torch.nn as nn
from torch import Tensor

from ..network_layers import MLP, Attention

__all__ = ["TextEncoderBlock"]


@dataclass(eq=False)
class TextEncoderBlock(nn.Module):
    dim: int
    num_heads: int
    dropout_prob: float
    cross_attention: bool = False

    def __post_init__(self):
        super().__init__()

        self.norm_attn = nn.LayerNorm(self.dim, eps=1e-12)
        self.attention = Attention(self.dim, self.num_heads, self.dropout_prob)

        if self.cross_attention:
            self.norm_crossattn = nn.LayerNorm(self.dim, eps=1e-12)
            self.crossattn = Attention(self.dim, self.num_heads, self.dropout_prob)

        self.norm_mlp = nn.LayerNorm(self.dim, eps=1e-12)
        self.mlp = MLP(self.dim)

        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(
        self,
        x: Tensor,
        attn_mask: Tensor,
        context: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.norm_attn(x + self.dropout(self.attention(x, attn_mask)))

        if self.cross_attention and context is not None:
            x = self.norm_crossattn(
                x + self.dropout(self.crossattn(x, context=context)),
            )

        return self.norm_mlp(x + self.dropout(self.mlp(x)))
