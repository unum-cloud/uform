from os import PathLike
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    InterpolationMode,
)

from tokenizers import Tokenizer
from PIL.Image import Image


# lambda is not pickable
def convert_to_rgb(image):
    return image.convert("RGB")


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
        self, x: Tensor, attn_mask: Tensor, context: Optional[Tensor] = None
    ) -> Tensor:
        x = self.norm_attn(x + self.dropout(self.attention(x, attn_mask)))

        if self.cross_attention and context is not None:
            x = self.norm_crossattn(
                x + self.dropout(self.crossattn(x, context=context))
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


class VLM(nn.Module):
    """
    Vision-Language Model for multi-modal embeddings.
    """

    def __init__(self, config: Dict, tokenizer_path: PathLike):
        """
        :param config: Model config
        """

        super().__init__()
        self._max_seq_len = config["text_encoder"]["max_position_embeddings"]
        self._embedding_dim = config["text_encoder"]["embedding_dim"]
        self._image_size = config["image_encoder"]["image_size"]

        self.text_encoder = TextEncoder(**config["text_encoder"])
        self.image_encoder = VisualEncoder(**config["image_encoder"])

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.no_padding()
        self._pad_token_idx = self.text_encoder.padding_idx

        self._image_transform = Compose(
            [
                Resize(self._image_size, interpolation=InterpolationMode.BICUBIC),
                convert_to_rgb,
                CenterCrop(self._image_size),
                ToTensor(),
                Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

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
            texts["input_ids"], texts["attention_mask"]
        )
        embeddings = self.text_encoder.forward_embedding(
            features, texts["attention_mask"]
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
    ) -> Tensor:
        """Passes preprocessed texts (or precomputed texts features) and
            preprocessed images (or precomputed images features) through multimodal encoded to produce multimodal joint embeddings.

        :param image: Preprocessed images
        :param text: Preprocesses texts
        :param image_features: Precomputed images features
        :param text_features: Precomputed text features
        :param attention_mask: Attention masks, not required if pass `text` instead of text_features
        """

        assert (
            image is not None or image_features is not None
        ), "Either `image` or `image_features` should be non None"
        assert (
            text is not None or text_features is not None
        ), "Either `text_data` or `text_features` should be non None"

        if text_features is not None:
            assert (
                attention_mask is not None
            ), "if `text_features` is not None, then you should pass `attention_mask`"

        if image_features is None:
            image_features = self.image_encoder.forward_features(image)

        if text_features is None:
            text_features = self.text_encoder.forward_features(
                text["input_ids"], text["attention_mask"]
            )

        return self.text_encoder.forward_multimodal(
            text_features,
            attention_mask if attention_mask is not None else text["attention_mask"],
            image_features,
        )

    def get_matching_scores(self, embeddings: Tensor) -> Tensor:
        """Computes the probability that there is a match between images and texts based on their multimodal embeddings

        :param embeddings: multimodal joint embeddings
        """

        return self.text_encoder.forward_matching(embeddings)

    def preprocess_text(self, texts: Union[str, List[str]]) -> Dict[str, Tensor]:
        """Transforms one or more strings into dictionary with tokenized strings and attention masks.

        :param texts: text of list of texts to tokenizer
        """
        if isinstance(texts, str):
            texts = [texts]

        input_ids = torch.full(
            (len(texts), self.text_encoder.max_position_embeddings),
            fill_value=self._pad_token_idx,
            dtype=torch.int64,
        )

        attention_mask = torch.zeros(
            len(texts), self.text_encoder.max_position_embeddings, dtype=torch.int32
        )
        encoded = self._tokenizer.encode_batch(texts)

        for i, seq in enumerate(encoded):
            seq_len = min(len(seq), self.text_encoder.max_position_embeddings)
            input_ids[i, :seq_len] = torch.LongTensor(
                seq.ids[: self.text_encoder.max_position_embeddings]
            )
            attention_mask[i, :seq_len] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def preprocess_image(self, images: Union[Image, List[Image]]) -> Tensor:
        """Transforms one or more Pillow images into Torch Tensors.

        :param images: image or list of images to preprocess
        """

        if isinstance(images, list):
            batch_images = torch.empty(
                (len(images), 3, self._image_size, self._image_size),
                dtype=torch.float32,
            )

            for i, image in enumerate(images):
                batch_images[i] = self._image_transform(image)

            return batch_images
        else:
            return self._image_transform(images).unsqueeze(0)

    def forward(
        self,
        images: torch.Tensor,
        texts: dict,
    ):
        """Inference forward method

        :param images: Preprocessed images
        :param texts: Preprocessed texts
        :return: embeddings for images and texts
        """
        _, embs_imgs = self.image_encoder(images)
        _, embs_txts = self.text_encoder(texts)
        return embs_imgs, embs_txts

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


class TritonClient(VLM):
    """
    Nvidia Triton client to connect to the remote VLM inference server.
    """

    def __init__(self, tokenizer_path, pad_token_idx, url: str = "localhost:7001"):
        import tritonclient.http as httpclient

        self._client = httpclient
        self._triton_client = self._client.InferenceServerClient(url=url)

        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.no_padding()
        self._pad_token_idx = pad_token_idx

        self._image_transform = Compose(
            [
                Resize(self._image_size, interpolation=InterpolationMode.BICUBIC),
                convert_to_rgb,
                CenterCrop(self._image_size),
                ToTensor(),
                Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def encode_image(
        self,
        imgs: Tensor,
    ):
        """
        Passes the pre-processed images through `image_encoder` to produce image embeddings.

        :param imgs: Preprocessed image
        """

        # images prep
        inputs = []
        outputs = []
        imgs = imgs.cpu().detach().numpy()
        inputs.append(self._client.InferInput("inputs", imgs.shape, "FP32"))
        inputs[0].set_data_from_numpy(imgs)
        outputs.append(self._client.InferRequestedOutput("output"))

        # Querying the server
        results = self._triton_client.infer(
            model_name="vit", inputs=inputs, outputs=outputs
        )
        output_data = torch.from_numpy(results.as_numpy("output"))
        return output_data

    def encode_text(
        self,
        text: Dict[str, Tensor],
    ):
        """
        Passes the pre-processed texts through `text_encoder` to produce texts embeddings.

        :param text: Dictionary with tokenized texts and attention masks
        """

        # texts prep
        inputs = []
        input_ids, attention_mask = text["input_ids"], text["attention_mask"]
        input_ids = input_ids.type(dtype=torch.int32).cpu().detach().numpy()
        attention_mask = attention_mask.type(dtype=torch.int32).cpu().detach().numpy()
        inputs.append(
            self._client.InferInput("attention_mask", attention_mask.shape, "INT32")
        )
        inputs.append(self._client.InferInput("input_ids", input_ids.shape, "INT32"))
        inputs[0].set_data_from_numpy(attention_mask)
        inputs[1].set_data_from_numpy(input_ids)
        test_output = self._client.InferRequestedOutput("output")

        # Querying the server
        results = self._triton_client.infer(
            model_name="albef", inputs=inputs, outputs=[test_output]
        )
        output_vec = torch.from_numpy(results.as_numpy("output"))
        return output_vec

    def encode_multimodal(self, *args, **kwargs):
        raise NotImplementedError("Multimodal encodings coming soon!")


class VLM_IPU(VLM):
    """
    Code for GraphCore IPUs.
    Please read User Guide if you want UForm to work on GraphCore hardware.
    (https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/intro.html)
    """

    def __init__(self, config: Dict, tokenizer_path: PathLike):
        import poptorch

        self.poptorch = poptorch
        super().__init__(config, tokenizer_path)

    def recomputation_checkpoint(self, module):
        """
        Annotates the output of a module to be checkpointed instead of recomputed
        """

        def recompute_outputs(module, inputs, outputs):
            if type(outputs) is tuple:
                return tuple(self.poptorch.recomputationCheckpoint(y) for y in outputs)
            else:
                return self.poptorch.recomputationCheckpoint(outputs)

        module.register_forward_hook(recompute_outputs)

    def parallelize(self):
        """
        Splits the model layers between IPU devices.
        """
        print("---------- Device Allocation -----------")
        print("image_encoder 0 ~ 6--> IPU 0")
        for index in range(4):
            layer = self.image_encoder.blocks[index]
            self.recomputation_checkpoint(layer)
            self.image_encoder.blocks[index] = self.poptorch.BeginBlock(
                layer,
                f"image_encoder_layer{index}",
                ipu_id=0,
            )

        print("image_encoder 4 ~ 8 --> IPU 1")
        for index in range(4, 8):
            layer = self.image_encoder.blocks[index]
            self.recomputation_checkpoint(layer)
            self.image_encoder.blocks[index] = self.poptorch.BeginBlock(
                layer,
                f"image_encoder_layer{index}",
                ipu_id=1,
            )

        print("image_encoder 8 ~ 12 --> IPU 2")
        for index in range(8, 12):
            layer = self.image_encoder.blocks[index]
            self.recomputation_checkpoint(layer)
            self.image_encoder.blocks[index] = self.poptorch.BeginBlock(
                layer,
                f"image_encoder_layer{index}",
                ipu_id=2,
            )

        print("text_enocder 0 ~ 4 --> IPU 3")
        for index in range(0, 4):
            layer = self.text_encoder.blocks[index]
            self.recomputation_checkpoint(layer)
            self.text_encoder.blocks[index] = self.poptorch.BeginBlock(
                layer,
                f"text_encoder_layer{index}",
                ipu_id=3,
            )

        return self
