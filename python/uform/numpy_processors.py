from os import PathLike
from typing import Dict, List, Union
import json

from PIL.Image import Image, BICUBIC
from tokenizers import Tokenizer
import numpy as np


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

    def __call__(self, texts: Union[str, List[str]]) -> Dict[str, np.ndarray]:
        """Transforms one or more strings into dictionary with tokenized strings and attention masks.

        :param texts: text of list of texts to tokenizer
        """
        if isinstance(texts, str):
            texts = [texts]

        input_ids = np.full(
            (len(texts), self._max_seq_len),
            fill_value=self._pad_token_idx,
            dtype=np.int64,
        )

        attention_mask = np.zeros(
            (len(texts), self._max_seq_len),
            dtype=np.int32,
        )
        encoded = self._tokenizer.encode_batch(texts)

        for i, seq in enumerate(encoded):
            seq_len = min(len(seq), self._max_seq_len)
            input_ids[i, :seq_len] = seq.ids[:seq_len]

            attention_mask[i, :seq_len] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class ImageProcessor:
    def __init__(self, config_path: PathLike, tokenizer_path: PathLike = None):
        """
        :param config: model config
        :param tokenizer_path: path to tokenizer file
        :param tensor_type: which tensors to return, either pt (PyTorch) or np (NumPy)
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

        self.image_mean = np.array(self._normalization_means, dtype=np.float32)[None, None]
        self.image_std = np.array(self._normalization_deviations, dtype=np.float32)[None, None]

    def __call__(self, images: Union[Image, List[Image]]) -> np.ndarray:
        """Transforms one or more Pillow images into Torch Tensors.

        :param images: image or list of images to preprocess
        """

        if isinstance(images, list):
            batch_images = np.empty(
                (len(images), 3, self._image_size, self._image_size),
                dtype=np.float32,
            )

            for i, image in enumerate(images):
                batch_images[i] = self._resize_crop_normalize(image)

        else:
            batch_images = self._resize_crop_normalize(images)[None]

        return batch_images

    def _resize_crop_normalize(self, image: Image):
        width, height = image.size

        if width < height:
            width = self._image_size
            height = int(height / width * self._image_size)
        else:
            width = int(width / height * self._image_size)
            height = self._image_size

        image = image.resize((width, height), resample=BICUBIC)

        left = (width - self._image_size) / 2
        top = (height - self._image_size) / 2
        right = (width + self._image_size) / 2
        bottom = (height + self._image_size) / 2

        image = image.convert("RGB").crop((left, top, right, bottom))
        # At this point `image` is a PIL Image with RGB channels.
        # If you convert it to `np.ndarray` it will have shape (H, W, C) where C is the number of channels.
        image = (np.array(image).astype(np.float32) / 255.0 - self.image_mean) / self.image_std

        # To make it compatible with PyTorch, we need to transpose the image to (C, H, W).
        return np.transpose(image, (2, 0, 1))
