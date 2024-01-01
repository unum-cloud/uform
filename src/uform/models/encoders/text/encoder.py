import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .block import TextEncoderBlock

__all__ = ["TextEncoder"]


class TextEncoder(nn.Module):
    model_type: str
    dim: int
    context_dim: int
    vocab_size: int
    padding_idx: int
    num_layers: int
    num_heads: int
    embedding_dim: int
    multimodal_layers_ids: tuple
    head_one_neuron: bool
    pooling: str = "cls"
    max_position_embeddings: int = 77
    dropout_prob: float = 0

    def __post_init__(self):
        super().__init__()

        self.word_embeddings = nn.Embedding(
            self.vocab_size, self.dim, padding_idx=self.padding_idx
        )
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.dim)

        if self.model_type == "bert":
            self.register_buffer(
                "position_ids",
                torch.arange(self.max_position_embeddings).unsqueeze(0),
                persistent=False,
            )

        self.layer_norm = nn.LayerNorm(self.dim, eps=1e-12)
        self.dropout = nn.Dropout(self.dropout_prob)

        self.blocks = nn.ModuleList(
            [
                TextEncoderBlock(
                    self.dim,
                    self.num_heads,
                    self.dropout_prob,
                    layer_id in self.multimodal_layers_ids,
                )
                for layer_id in range(self.num_layers)
            ]
        )

        self.embedding_projection = nn.Linear(self.dim, self.embedding_dim, bias=False)
        self.matching_head = nn.Linear(self.dim, 1 if self.head_one_neuron else 2)

        if self.context_dim != self.dim:
            self.context_projection = nn.Linear(self.context_dim, self.dim, bias=False)
        else:
            self.context_projection = nn.Identity()

    def forward_features(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        x = self.embed_text(x)
        attn_mask = self.get_attention_mask(attn_mask, x.dtype)

        for block in self.blocks:
            if not block.cross_attention:
                x = block(x, attn_mask)

        return x

    def forward_multimodal(
        self, x: Tensor, attn_mask: Tensor, context: Tensor
    ) -> Tensor:
        context = self.context_projection(context)
        expanded_attn_mask = self.get_attention_mask(attn_mask, x.dtype)
        for block in self.blocks:
            if block.cross_attention:
                x = block(x, expanded_attn_mask, context)

        return self.pool_features(x, attn_mask)

    def forward_embedding(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        return self.embedding_projection(self.pool_features(x, attn_mask))

    def forward_matching(self, x: Tensor) -> Tensor:
        logits = self.matching_head(x)
        if self.head_one_neuron:
            return torch.sigmoid(logits)[:, 0]

        return F.softmax(logits, dim=1)[:, 1]

    def pool_features(self, x: Tensor, attn_mask: Tensor) -> Tensor:
        if self.pooling == "cls":
            return x[:, 0]

        attn_mask = attn_mask.unsqueeze(2).type_as(x)

        return (x * attn_mask).sum(dim=1) / attn_mask.sum(dim=1)

    def get_attention_mask(self, attn_mask: Tensor, dtype: torch.dtype) -> Tensor:
        attn_mask = attn_mask.to(dtype)
        attn_mask = (1.0 - attn_mask) * torch.finfo(dtype).min
        return attn_mask.unsqueeze(1).expand(-1, attn_mask.shape[1], -1).unsqueeze(1)

    def get_position_ids(self, x: Tensor) -> Tensor:
        if self.model_type == "roberta":
            mask = x.ne(self.padding_idx).int()
            return (
                torch.cumsum(mask, dim=1).type_as(mask) * mask
            ).long() + self.padding_idx

        return self.position_ids[:, : x.shape[1]]

    def embed_text(self, x: Tensor) -> Tensor:
        positional_embedding = self.position_embeddings(self.get_position_ids(x))
        x = self.word_embeddings(x) + positional_embedding
        return self.dropout(self.layer_norm(x))

    def forward(self, x: dict) -> torch.Tensor:
        features = self.forward_features(x["input_ids"], x["attention_mask"])
        embeddings = self.forward_embedding(features, x["attention_mask"])
        return features, embeddings
