from json import load
from typing import Optional, Tuple, Mapping

import torch
from huggingface_hub import snapshot_download

from models import *


def get_checkpoint(model_name, token) -> Tuple[str, Mapping, str]:
    model_path = snapshot_download(repo_id=model_name, token=token)
    config_path = f"{model_path}/torch_config.json"
    state = torch.load(f"{model_path}/torch_weight.pt")

    return config_path, state, f"{model_path}/tokenizer.json"


def get_model(model_name: str, token: Optional[str] = None) -> VLM:
    config_path, state, tokenizer_path = get_checkpoint(model_name, token)

    with open(config_path, "r") as f:
        model = VLM(load(f), tokenizer_path)

    model.image_encoder.load_state_dict(state["image_encoder"])
    model.text_encoder.load_state_dict(state["text_encoder"])

    return model.eval()


def get_client(
    url: str,
    model_name: str = "unum-cloud/uform-vl-english",
    token: Optional[str] = None,
) -> TritonClient:
    config_path, _, tokenizer_path = get_checkpoint(model_name, token)

    with open(config_path, "r") as f:
        pad_token_idx = load(f)["text_encoder"]["padding_idx"]

    return TritonClient(tokenizer_path, pad_token_idx, url)


def get_model_ipu(model_name: str, token: Optional[str] = None) -> VLM_IPU:
    config_path, state, tokenizer_path = get_checkpoint(model_name, token)

    with open(config_path, "r") as f:
        model = VLM_IPU(load(f), tokenizer_path)

    model.image_encoder.load_state_dict(state["image_encoder"])
    model.text_encoder.load_state_dict(state["text_encoder"])

    return model.eval()
