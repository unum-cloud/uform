from json import load
from os.path import join, exists
from typing import Mapping, Optional, Tuple
from enum import Enum

from huggingface_hub import snapshot_download


class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"


def get_checkpoint(model_name: str, token: Optional[str], modalities: Tuple[str, Modality]) -> Tuple[str, Mapping, str]:
    import torch

    # It is not recommended to use `.pth` extension when checkpointing models
    # because it collides with Python path (`.pth`) configuration files.
    merged_model_names = ["torch_weight.pt", "weight.pt", "model.pt"]
    separate_modality_names = [(x.value if isinstance(x, Modality) else x) + ".pt" for x in modalities]
    config_names = ["torch_config.json", "config.json"]
    tokenizer_names = ["tokenizer.json"]

    # The download stats depend on the number of times the `config.json` is pulled
    # https://huggingface.co/docs/hub/models-download-stats
    model_path = snapshot_download(
        repo_id=model_name,
        token=token,
        allow_patterns=merged_model_names + separate_modality_names + config_names + tokenizer_names,
    )

    # Find the first name in `config_names` that is present
    config_path = None
    for config_name in config_names:
        if exists(join(model_path, config_name)):
            config_path = join(model_path, config_name)
            break

    # Same for the tokenizer
    tokenizer_path = None
    for tokenizer_name in tokenizer_names:
        if exists(join(model_path, tokenizer_name)):
            tokenizer_path = join(model_path, tokenizer_name)
            break

    # Ideally, we want to separately fetch all the models.
    # If those aren't available, aggregate separate modalities and merge them.
    state = None
    for file_name in merged_model_names:
        if exists(join(model_path, file_name)):
            state = torch.load(join(model_path, file_name))
            break

    if state is None:
        state = {}
        for file_name in separate_modality_names:
            if exists(join(model_path, file_name)):
                modality_name, _, _ = file_name.partition(".")
                property_name = modality_name + "_encoder"
                state[property_name] = torch.load(join(model_path, file_name))

    return config_path, state, tokenizer_path


def get_model(model_name: str, token: Optional[str] = None, modalities: Optional[Tuple[str]] = None):
    from python.uform.torch_encoders import TextVisualEncoder
    from python.uform.torch_processors import TorchProcessor

    if modalities is None:
        modalities = (Modality.TEXT, Modality.IMAGE)

    config_path, state, tokenizer_path = get_checkpoint(model_name, token, modalities)

    with open(config_path) as f:
        config = load(f)

    model = TextVisualEncoder(config, tokenizer_path)
    model.image_encoder.load_state_dict(state.get("image_encoder", None))
    model.text_encoder.load_state_dict(state.get("text_encoder", None))
    processor = TorchProcessor(config, tokenizer_path)

    return model.eval(), processor


def get_model_onnx(model_name: str, device: str, dtype: str, token: Optional[str] = None):
    from python.uform.onnx_encoders import TextVisualEncoder
    from python.uform.numpy_processors import NumPyProcessor

    assert device in (
        "cpu",
        "gpu",
    ), f"Invalid `device`: {device}. Must be either `cpu` or `gpu`"
    assert dtype in (
        "fp32",
        "fp16",
    ), f"Invalid `dtype`: {dtype}. Must be either `fp32` or `fp16` (only for gpu)"
    assert (
        device == "cpu" and dtype == "fp32"
    ) or device == "gpu", "Combination `device`=`cpu` & `dtype=fp16` is not supported"

    model_path = snapshot_download(repo_id=f"{model_name}-{device}-{dtype}", token=token)

    with open(join(model_path, "config.json")) as f:
        config = load(f)

    model = TextVisualEncoder(model_path, config, device, dtype)
    processor = NumPyProcessor(config, join(model_path, "tokenizer.json"))

    return model, processor
