from dataclasses import dataclass
from os import PathLike
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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


@dataclass(eq=False)
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
            self.vocab_size,
            self.dim,
            padding_idx=self.padding_idx,
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
            ],
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
        self,
        x: Tensor,
        attn_mask: Tensor,
        context: Tensor,
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
            return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + self.padding_idx

        return self.position_ids[:, : x.shape[1]]

    def embed_text(self, x: Tensor) -> Tensor:
        positional_embedding = self.position_embeddings(self.get_position_ids(x))
        x = self.word_embeddings(x) + positional_embedding
        return self.dropout(self.layer_norm(x))

    def forward(self, x: dict) -> Tensor:
        features = self.forward_features(x["input_ids"], x["attention_mask"])
        embeddings = self.forward_embedding(features, x["attention_mask"])
        return features, embeddings


@dataclass(eq=False)
class VisualEncoder(nn.Module):
    dim: int
    patch_size: int
    image_size: int
    num_layers: int
    num_heads: int
    embedding_dim: int
    pooling: str
    num_reg_tokens: int = 0

    def __post_init__(self):
        super().__init__()

        seq_len = (self.image_size // self.patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, self.dim, self.patch_size, self.patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, self.dim) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.dim))

        if self.num_reg_tokens > 0:
            self.reg_token = nn.Parameter(torch.zeros(1, self.num_reg_tokens, self.dim))

        self.blocks = nn.Sequential(
            *[VisualEncoderBlock(self.dim, self.num_heads) for _ in range(self.num_layers)],
        )

        self.norm = nn.LayerNorm(self.dim, eps=1e-6)
        self.embedding_projection = nn.Linear(self.dim, self.embedding_dim, bias=False)

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x).flatten(start_dim=2).transpose(2, 1)
        x = x + self.pos_embed

        special_tokens = [self.cls_token.expand(x.shape[0], -1, -1)]

        if self.num_reg_tokens > 0:
            special_tokens.append(self.reg_token.expand(x.shape[0], -1, -1))

        x = torch.cat(special_tokens + [x], dim=1)

        x = self.blocks(x)

        return self.norm(x)

    def forward_embedding(self, x: Tensor) -> Tensor:
        if self.pooling == "cls":
            x = x[:, 0]
        else:
            x = x.mean(dim=1)

        return self.embedding_projection(x)

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        embeddings = self.forward_embedding(features)
        return features, embeddings


class VLM(nn.Module):
    """
    Vision-Language Model for Multimodal embeddings.
    """

    def __init__(self, config: Dict, tokenizer_path: PathLike):
        """
        :param config: Model config
        """

        super().__init__()
        self._embedding_dim = config["text_encoder"]["embedding_dim"]

        self.text_encoder = TextEncoder(**config["text_encoder"])
        self.image_encoder = VisualEncoder(**config["image_encoder"])

    def encode_image(
        self,
        images: Tensor,
        return_features: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Passes the pre-processed images through `image_encoder` to produce images features (optional) and embeddings.

        :param images: Preprocessed image
        :param return_features: Whether to return images features or return only embeddings
        """

        features = self.image_encoder.forward_features(images)
        embeddings = self.image_encoder.forward_embedding(features)

        if return_features:
            return features, embeddings

        return embeddings

    def encode_text(
        self,
        texts: Dict[str, Tensor],
        return_features: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Passes the pre-processed texts through `text_encoder` to produce texts features (optional) and embeddings.

        :param texts: Dictionary with tokenized texts and attention masks
        :param return_features: Whether to return texts features or return only embeddings
        """

        features = self.text_encoder.forward_features(
            texts["input_ids"],
            texts["attention_mask"],
        )
        embeddings = self.text_encoder.forward_embedding(
            features,
            texts["attention_mask"],
        )

        if return_features:
            return features, embeddings

        return embeddings

    def encode_multimodal(
        self,
        image: Optional[Tensor] = None,
        text: Optional[Dict] = None,
        image_features: Optional[Tensor] = None,
        text_features: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        return_scores: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Passes preprocessed texts (or precomputed texts features) and
            preprocessed images (or precomputed images features) through multimodal encoded to produce multimodal joint embeddings.

        :param image: Preprocessed images
        :param text: Preprocessed texts
        :param image_features: Precomputed images features
        :param text_features: Precomputed text features
        :param attention_mask: Attention masks, not required if pass `text` instead of text_features
        """

        assert image is not None or image_features is not None, "Either `image` or `image_features` should be non None"
        assert text is not None or text_features is not None, "Either `text_data` or `text_features` should be non None"

        if text_features is not None:
            assert attention_mask is not None, "if `text_features` is not None, then you should pass `attention_mask`"

        if image_features is None:
            image_features = self.image_encoder.forward_features(image)

        if text_features is None:
            text_features = self.text_encoder.forward_features(
                text["input_ids"],
                text["attention_mask"],
            )

        embeddings = self.text_encoder.forward_multimodal(
            text_features,
            attention_mask if attention_mask is not None else text["attention_mask"],
            image_features,
        )
        
        if return_scores:
            return self.get_matching_scores(embeddings), embeddings

        return embeddings

    def get_matching_scores(self, embeddings: Tensor) -> Tensor:
        """Computes the probability that there is a match between images and texts based on their multimodal embeddings

        :param embeddings: multimodal joint embeddings
        """

        return self.text_encoder.forward_matching(embeddings)

    def forward(
        self,
        images: Tensor,
        texts: Dict[str, Tensor],
    ) -> Union[Tensor, Tensor]:
        """Inference forward method

        :param images: Preprocessed images
        :param texts: Preprocessed texts
        :return: embeddings for images and texts
        """
        _, image_embeddings = self.image_encoder(images)
        _, text_embeddings = self.text_encoder(texts)
        return image_embeddings, text_embeddings

    @property
    def text_features_dim(self) -> int:
        """Dimensionality of the text encoder features."""

        return self.text_encoder.dim

    @property
    def image_features_dim(self) -> int:
        """Dimensionality of the image encoder features."""

        return self.image_encoder.dim

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of shared space embedding."""

        return self._embedding_dim

    @property
    def multimodal_embedding_dim(self) -> int:
        """Dimensionality of multimodal joint embedding."""
        return self.text_encoder.dim
