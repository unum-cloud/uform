from json import load
from typing import Optional

import torch
from huggingface_hub import hf_hub_download

from models import VLM, TritonClient, VLM_IPU


def get_model(model_name: str, token: Optional[str] = None) -> VLM:
    config_path = hf_hub_download(model_name, "torch_config.json", token=token)
    state = torch.load(hf_hub_download(model_name, "torch_weight.pt", token=token))
    tokenizer_path = hf_hub_download(model_name, "tokenizer.json", token=token)

    with open(config_path, "r") as f:
        model = VLM(load(f), tokenizer_path)

    model.image_encoder.load_state_dict(state["image_encoder"])
    model.text_encoder.load_state_dict(state["text_encoder"])

    return model.eval()


def get_client(
        url: str,
        model_name: str = "unum-cloud/uform-vl-english",
        token: Optional[str] = None) -> TritonClient:
    
    config_path = hf_hub_download(model_name, "torch_config.json", token=token)
    tokenizer_path = hf_hub_download(model_name, "tokenizer.json", token=token)

    with open(config_path, 'r') as f:
        pad_token_idx = load(f)["text_encoder"]["padding_idx"]

    return TritonClient(tokenizer_path, pad_token_idx, url)


def get_model_ipu(model_name: str, token: Optional[str] = None) -> VLM_IPU:
    config_path = hf_hub_download(model_name, "torch_config.json", token=token)
    state = torch.load(hf_hub_download(model_name, "torch_weight.pt", token=token))
    tokenizer_path = hf_hub_download(model_name, "tokenizer.json", token=token)
    
    with open(config_path, "r") as f:
        model = VLM_IPU(load(f), tokenizer_path)

    model.image_encoder.load_state_dict(state["image_encoder"])
    model.text_encoder.load_state_dict(state["text_encoder"])

    return model.eval()
