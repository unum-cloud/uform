from typing import Dict

import torch
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

from .image_utils import convert_to_rgb
from .vlm import VLM

__all__ = ["TritonClient"]


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
            ],
        )

    def encode_image(self, imgs: Tensor):
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
            model_name="vit",
            inputs=inputs,
            outputs=outputs,
        )
        output_data = torch.from_numpy(results.as_numpy("output"))
        return output_data

    def encode_text(self, text: Dict[str, Tensor]):
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
            self._client.InferInput("attention_mask", attention_mask.shape, "INT32"),
        )
        inputs.append(self._client.InferInput("input_ids", input_ids.shape, "INT32"))
        inputs[0].set_data_from_numpy(attention_mask)
        inputs[1].set_data_from_numpy(input_ids)
        test_output = self._client.InferRequestedOutput("output")

        # Querying the server
        results = self._triton_client.infer(
            model_name="albef",
            inputs=inputs,
            outputs=[test_output],
        )
        output_vec = torch.from_numpy(results.as_numpy("output"))
        return output_vec

    def encode_multimodal(self, *args, **kwargs):
        raise NotImplementedError("Multimodal encodings coming soon!")
