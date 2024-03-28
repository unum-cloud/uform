from os import PathLike
from typing import Dict, List, Union

import torch
from PIL import Image
from tokenizers import Tokenizer
from torch import Tensor
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, Resize, ToTensor)


# lambda is not pickable
def convert_to_rgb(image):
    return image.convert("RGB")


class Processor:
    def __init__(self, config: Dict, tokenizer_path: PathLike, tensor_type: str = "pt"):
        """
        :param config: model config
        :param tokenizer_path: path to tokenizer file
        :param tensor_type: which tensors to return, either pt (PyTorch) or np (NumPy)
        """

        assert tensor_type in ("pt", "np"), "`tensor_type` must be either `pt` or `np`"

        self._image_size = config["image_encoder"]["image_size"]
        self._max_seq_len = config["text_encoder"]["max_position_embeddings"]
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.no_padding()
        self._pad_token_idx = config["text_encoder"]["padding_idx"]

        self.tensor_type = tensor_type

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

    def preprocess_text(self, texts: Union[str, List[str]]) -> Dict[str, Tensor]:
        """Transforms one or more strings into dictionary with tokenized strings and attention masks.

        :param texts: text of list of texts to tokenizer
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
                seq.ids[: self._max_seq_len],
            )
            attention_mask[i, :seq_len] = 1

        if self.tensor_type == "np":
            return {
                "input_ids": input_ids.numpy(),
                "attention_mask": attention_mask.numpy(),
            }

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

        else:
            batch_images = self._image_transform(images).unsqueeze(0)

        if self.tensor_type == "np":
            return batch_images.numpy()

        return batch_images
