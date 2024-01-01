from os import PathLike
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from PIL.Image import Image
from tokenizers import Tokenizer
from torch import Tensor
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    Resize,
    ToTensor,
)

from .encoders import TextEncoder, VisualEncoder
from .image_utils import convert_to_rgb

__all__ = ["VLM", "VLM_IPU"]


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
            ],
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
                text["input_ids"],
                text["attention_mask"],
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
            len(texts),
            self.text_encoder.max_position_embeddings,
            dtype=torch.int32,
        )
        encoded = self._tokenizer.encode_batch(texts)

        for i, seq in enumerate(encoded):
            seq_len = min(len(seq), self.text_encoder.max_position_embeddings)
            input_ids[i, :seq_len] = torch.LongTensor(
                seq.ids[: self.text_encoder.max_position_embeddings],
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
