from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["Attention", "MLP", "LayerScale"]


@dataclass(eq=False)
class Attention(nn.Module):
    dim: int
    num_heads: int
    dropout_prob: float = 0

    def __post_init__(self):
        super().__init__()

        self.use_sdp = int(torch.__version__[0]) > 1

        self.query = nn.Linear(self.dim, self.dim)
        self.key = nn.Linear(self.dim, self.dim)
        self.value = nn.Linear(self.dim, self.dim)
        self.out = nn.Linear(self.dim, self.dim)

        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim**-0.5

    def forward(
        self,
        x: Tensor,
        attn_mask: Optional[Tensor] = None,
        context: Optional[Tensor] = None,
        is_causal: bool = False,
    ) -> Tensor:
        query = self.reshape(self.query(x))
        key = self.reshape(self.key(x if context is None else context))
        value = self.reshape(self.value(x if context is None else context))

        if self.use_sdp:
            x = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask,
                dropout_p=self.dropout_prob if self.training else 0,
                is_causal=is_causal,
            )
        else:
            attn = query @ key.transpose(-2, -1) * self.scale
            if attn_mask is not None:
                attn += attn_mask

            attn = attn.softmax(dim=-1)
            x = attn @ value

        return self.out(x.transpose(2, 1).flatten(2))

    def reshape(self, x: Tensor) -> Tensor:
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(2, 1)


@dataclass(eq=False)
class MLP(nn.Module):
    dim: int
    dim_expand_factor: int = 4

    def __post_init__(self):
        super().__init__()

        self.hidden_layer = nn.Linear(self.dim, self.dim * self.dim_expand_factor)
        self.output_layer = nn.Linear(self.dim * self.dim_expand_factor, self.dim)

    def forward(self, x: Tensor) -> Tensor:
        x = F.gelu(self.hidden_layer(x))
        return self.output_layer(x)


@dataclass(eq=False)
class LayerScale(nn.Module):
    dim: int
    init_values: float = 1e-5
    inplace: bool = False

    def __post_init__(self):
        super().__init__()
        self.gamma = nn.Parameter(self.init_values * torch.ones(self.dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
