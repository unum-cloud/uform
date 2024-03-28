from json import load
from os.path import join
from typing import Mapping, Optional, Tuple

from huggingface_hub import snapshot_download


def get_checkpoint(model_name: str, token: str) -> Tuple[str, Mapping, str]:
    import torch

    model_path = snapshot_download(repo_id=model_name, token=token)
    config_path = join(model_path, "torch_config.json")

    state = torch.load(join(model_path, "torch_weight.pt"))
    return config_path, state, join(model_path, "tokenizer.json")


def get_model(model_name: str, token: Optional[str] = None):
    from uform.torch_models import VLM
    from uform.torch_preprocessor import TorchProcessor

    config_path, state, tokenizer_path = get_checkpoint(model_name, token)

    with open(config_path) as f:
        config = load(f)

    model = VLM(config, tokenizer_path)
    model.image_encoder.load_state_dict(state["image_encoder"])
    model.text_encoder.load_state_dict(state["text_encoder"])
    processor = TorchProcessor(config, tokenizer_path)

    return model.eval(), processor


def get_model_onnx(model_name: str, device: str, dtype: str, token: Optional[str] = None):
    from uform.onnx_models import VLM_ONNX
    from uform.numpy_preprocessor import NumPyProcessor

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

    model = VLM_ONNX(model_path, config, device, dtype)
    processor = NumPyProcessor(config, join(model_path, "tokenizer.json"))

    return model, processor
