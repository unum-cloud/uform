from typing import Tuple
import os

import pytest
from PIL import Image
import uform

# PyTorch is a very heavy dependency, so we may want to skip these tests if it's not installed
try:
    import torch

    torch_available = True
except:
    torch_available = False

# ONNX is not a very light dependency either
try:
    import onnx

    onnx_available = True
except:
    onnx_available = False

torch_models = [
    "unum-cloud/uform2-vl-english-small",
    "unum-cloud/uform-vl-english",
    "unum-cloud/uform-vl-multilingual-v2",
]

onnx_models_and_providers = [
    ("unum-cloud/uform-vl-english-small", "cpu", "fp32"),
    ("unum-cloud/uform-vl-english-large", "cpu", "fp32"),
    ("unum-cloud/uform-vl-english-small", "gpu", "fp32"),
    ("unum-cloud/uform-vl-english-large", "gpu", "fp32"),
    ("unum-cloud/uform-vl-english-small", "gpu", "fp16"),
    ("unum-cloud/uform-vl-english-large", "gpu", "fp16"),
]

# Let's check if the HuggingFace Hub API token is set in the environment variable.
# If it's not there, check if the `.hf_token` file is present in the current working directory.
token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
if token is None:
    token_path = "./.hf_token"
    if os.path.exists(token_path):
        with open(token_path, "r") as file:
            token = file.read().strip()


@pytest.mark.skipif(not torch_available, reason="PyTorch is not installed")
@pytest.mark.parametrize("model_name", torch_models)
def test_torch_one_embedding(model_name: str):
    model, processor = uform.get_model(model_name, token=token)
    text = "a small red panda in a zoo"
    image_path = "assets/unum.png"

    image = Image.open(image_path)
    image_data = processor.preprocess_image(image)
    text_data = processor.preprocess_text(text)

    image_features, image_embedding = model.encode_image(image_data, return_features=True)
    text_features, text_embedding = model.encode_text(text_data, return_features=True)

    assert image_embedding.shape[0] == 1, "Image embedding batch size is not 1"
    assert text_embedding.shape[0] == 1, "Text embedding batch size is not 1"

    # Test reranking
    score, joint_embedding = model.encode_multimodal(
        image_features=image_features,
        text_features=text_features,
        attention_mask=text_data["attention_mask"],
        return_scores=True,
    )
    assert score.shape[0] == 1, "Matching score batch size is not 1"
    assert joint_embedding.shape[0] == 1, "Joint embedding batch size is not 1"


@pytest.mark.skipif(not torch_available, reason="PyTorch is not installed")
@pytest.mark.parametrize("model_name", torch_models)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_torch_many_embeddings(model_name: str, batch_size: int):
    model, processor = uform.get_model(model_name, token=token)
    texts = ["a small red panda in a zoo"] * batch_size
    image_paths = ["assets/unum.png"] * batch_size

    images = [Image.open(path) for path in image_paths]
    image_data = processor.preprocess_image(images)
    text_data = processor.preprocess_text(texts)

    image_embeddings = model.encode_image(image_data, return_features=False)
    text_embeddings = model.encode_text(text_data, return_features=False)

    assert image_embeddings.shape[0] == batch_size, "Image embedding is unexpected"
    assert text_embeddings.shape[0] == batch_size, "Text embedding is unexpected"


@pytest.mark.skipif(not onnx_available, reason="ONNX is not installed")
@pytest.mark.parametrize("model_specs", onnx_models_and_providers)
def test_onnx_one_embedding(model_specs: Tuple[str, str, str]):

    from uform.onnx_models import ExecutionProviderError

    try:

        model, processor = uform.get_model_onnx(*model_specs, token=token)
        text = "a small red panda in a zoo"
        image_path = "assets/unum.png"

        image = Image.open(image_path)
        image_data = processor.preprocess_image(image)
        text_data = processor.preprocess_text(text)

        image_features, image_embedding = model.encode_image(image_data, return_features=True)
        text_features, text_embedding = model.encode_text(text_data, return_features=True)

        assert image_embedding.shape[0] == 1, "Image embedding batch size is not 1"
        assert text_embedding.shape[0] == 1, "Text embedding batch size is not 1"

        score, joint_embedding = model.encode_multimodal(
            image_features=image_features,
            text_features=text_features,
            attention_mask=text_data["attention_mask"],
            return_scores=True,
        )
        assert score.shape[0] == 1, "Matching score batch size is not 1"
        assert joint_embedding.shape[0] == 1, "Joint embedding batch size is not 1"

    except ExecutionProviderError as e:
        pytest.skip(f"Execution provider error: {e}")


@pytest.mark.skipif(not onnx_available, reason="ONNX is not installed")
@pytest.mark.parametrize("model_specs", onnx_models_and_providers)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_onnx_many_embeddings(model_specs: Tuple[str, str, str], batch_size: int):

    from uform.onnx_models import ExecutionProviderError

    try:

        model, processor = uform.get_model_onnx(*model_specs, token=token)
        texts = ["a small red panda in a zoo"] * batch_size
        image_paths = ["assets/unum.png"] * batch_size

        images = [Image.open(path) for path in image_paths]
        image_data = processor.preprocess_image(images)
        text_data = processor.preprocess_text(texts)

        image_embeddings = model.encode_image(image_data, return_features=False)
        text_embeddings = model.encode_text(text_data, return_features=False)

        assert image_embeddings.shape[0] == batch_size, "Image embedding is unexpected"
        assert text_embeddings.shape[0] == batch_size, "Text embedding is unexpected"

    except ExecutionProviderError as e:
        pytest.skip(f"Execution provider error: {e}")


if __name__ == "__main__":
    pytest.main(["-s", "-x", __file__])
