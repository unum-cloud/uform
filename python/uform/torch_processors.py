from os import PathLike
from typing import Dict, List, Union, Sequence
import json

import torch
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


# lambda is not pickle-able
def convert_to_rgb(image):
    return image.convert("RGB")


class TextProcessor:
    def __init__(self, config_path: PathLike, tokenizer_path: PathLike):
        """
        :param config: model config
        :param tokenizer_path: path to tokenizer file
        """

        config = json.load(open(config_path, "r"))
        if "text_encoder" in config:
            config = config["text_encoder"]

        self._max_seq_len = config["max_position_embeddings"]
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.no_padding()
        self._pad_token_idx = config["padding_idx"]

    def __call__(self, texts: Union[str, List[str]]) -> Dict[str, Tensor]:
        """Transforms one or more strings into dictionary with tokenized strings and attention masks.

        :param texts: text of list of texts to tokenizer
        :return: dictionary with tokenized strings and attention masks as values
        """
        if isinstance(texts, str):
            texts = [texts]

        input_ids = torch.full(
            (len(texts), self._max_seq_len),
            fill_value=self._pad_token_idx,
            dtype=torch.int64,
        )

        attention_mask = torch.zeros(
            len(texts),
            self._max_seq_len,
            dtype=torch.int32,
        )
        encoded = self._tokenizer.encode_batch(texts)

        for i, seq in enumerate(encoded):
            seq_len = min(len(seq), self._max_seq_len)
            input_ids[i, :seq_len] = torch.LongTensor(
                seq.ids[:seq_len],
            )
            attention_mask[i, :seq_len] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class ImageProcessor:
    def __init__(self, config_path: PathLike):
        """
        :param config: model config
        """

        config = json.load(open(config_path, "r"))
        if "image_encoder" in config:
            config = config["image_encoder"]

        self._image_size = config["image_size"]
        self._normalization_means = config["normalization_means"]
        self._normalization_deviations = config["normalization_deviations"]

        assert isinstance(self._image_size, int) and self._image_size > 0
        assert isinstance(self._normalization_means, list) and isinstance(self._normalization_deviations, list)
        assert len(self._normalization_means) == len(self._normalization_deviations) == 3

        self._image_transform = Compose(
            [
                Resize(self._image_size, interpolation=InterpolationMode.BICUBIC),
                convert_to_rgb,
                CenterCrop(self._image_size),
                ToTensor(),
                Normalize(
                    mean=tuple(self._normalization_means),
                    std=tuple(self._normalization_deviations),
                ),
            ],
        )

    def __call__(self, images: Union[Image, Sequence[Image]]) -> Dict[str, Tensor]:
        """Transforms one or more Pillow images into Torch Tensors.

        :param images: image or list of images to preprocess
        :return: dictionary with float-represented images in tensors as values
        """

        if isinstance(images, Sequence):
            batch_images = torch.empty(
                (len(images), 3, self._image_size, self._image_size),
                dtype=torch.float32,
            )

            for i, image in enumerate(images):
                batch_images[i] = self._image_transform(image)

        else:
            batch_images = self._image_transform(images).unsqueeze(0)

        return {"images": batch_images}
