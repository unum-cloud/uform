from json import load
from os.path import dirname, join
from typing import Mapping, Optional, Tuple, Union

import torch
from huggingface_hub import hf_hub_download, snapshot_download

from uform.models import (MLP, VLM, VLM_IPU, Attention, LayerScale,
                          TextEncoder, TextEncoderBlock, TritonClient,
                          VisualEncoder, VisualEncoderBlock)
from uform.onnx import VLM_ONNX
from uform.preprocessing import Processor

__all__ = [
    "MLP",
    "VLM",
    "VLM_IPU",
    "VLM_ONNX",
    "Attention",
    "LayerScale",
    "TextEncoder",
    "TextEncoderBlock",
    "TritonClient",
    "VisualEncoder",
    "VisualEncoderBlock",
    "Processor",
    "get_checkpoint",
    "get_model",
    "get_client",
    "get_model_ipu",
]


def get_checkpoint(model_name: str, token: str) -> Tuple[str, Mapping, str]:
    model_path = snapshot_download(repo_id=model_name, token=token)
    config_path = join(model_path, "torch_config.json")

    state = torch.load(join(model_path, "torch_weight.pt"))
    return config_path, state, join(model_path, "tokenizer.json")


def get_model(model_name: str, token: Optional[str] = None) -> Union[VLM, Processor]:
    config_path, state, tokenizer_path = get_checkpoint(model_name, token)

    with open(config_path) as f:
        config = load(f)
    
    model = VLM(config, tokenizer_path)
    model.image_encoder.load_state_dict(state["image_encoder"])
    model.text_encoder.load_state_dict(state["text_encoder"])
    processor = Processor(config, tokenizer_path)

    return model.eval(), processor

def get_model_onnx(model_name: str, device: str, dtype: str, token: Optional[str] = None) -> Union[VLM_ONNX, Processor]:
    assert device in ("cpu", "gpu"), f"Invalid `device`: {device}. Must be either `cpu` or `gpu`"
    assert dtype in ("fp32", "fp16"), f"Invalid `dtype`: {dtype}. Must be either `fp32` or `fp16` (only for gpu)"
    assert (device == "cpu" and dtype == "fp32") or device == "gpu", "Combination `device`=`cpu` & `dtype=fp16` is not supported"

    files_to_download = [
        "config.json",
        "tokenizer.json",
        f"image_encoder_{device}_{dtype}.onnx",
        f"text_encoder_{device}_{dtype}.onnx",
        f"reranker_{device}_{dtype}.onnx",
    ]

    for file in files_to_download:
        model_path = hf_hub_download(
            model_name,
            file,
            token=token
        )

    model_path = dirname(model_path)

    with open(join(model_path, "config.json")) as f:
        config = load(f)
    
    model = VLM_ONNX(model_path, config, device, dtype)
    processor = Processor(config, join(model_path, "tokenizer.json"), "np")

    return model, processor


def get_client(
    url: str,
    model_name: str = "unum-cloud/uform-vl-english",
    token: Optional[str] = None,
) -> TritonClient:
    config_path, _, tokenizer_path = get_checkpoint(model_name, token)

    with open(config_path) as f:
        pad_token_idx = load(f)["text_encoder"]["padding_idx"]

    return TritonClient(tokenizer_path, pad_token_idx, url)


def get_model_ipu(model_name: str, token: Optional[str] = None) -> VLM_IPU:
    config_path, state, tokenizer_path = get_checkpoint(model_name, token)

    with open(config_path) as f:
        model = VLM_IPU(load(f), tokenizer_path)

    model.image_encoder.load_state_dict(state["image_encoder"])
    model.text_encoder.load_state_dict(state["text_encoder"])

    return model.eval()
