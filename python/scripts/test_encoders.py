from typing import Tuple
import requests
from io import BytesIO
import os

import pytest
import numpy as np
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
    "unum-cloud/uform3-image-text-english-small",
    "unum-cloud/uform-vl-english",
    "unum-cloud/uform-vl-multilingual-v2",
]

onnx_models = [
    "unum-cloud/uform3-image-text-english-small",
]

# Let's check if the HuggingFace Hub API token is set in the environment variable.
# If it's not there, check if the `.hf_token` file is present in the current working directory.
token = os.getenv("HUGGINGFACE_HUB_TOKEN", None)
if token is None:
    token_path = "./.hf_token"
    if os.path.exists(token_path):
        with open(token_path, "r") as file:
            token = file.read().strip()


def cosine_similarity(x, y) -> float:
    if not isinstance(x, np.ndarray):
        x = x.detach().numpy()
    if not isinstance(y, np.ndarray):
        y = y.detach().numpy()

    # Unlike NumPy, SimSIMD can properly deal with integer types
    x = x.astype(np.float32).flatten()
    y = y.astype(np.float32).flatten()
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def cross_references_image_and_text_embeddings(text_to_embedding, image_to_embedding):
    """Test if the embeddings of text and image are semantically similar
    using a small set of example text-image pairs."""

    texts = [
        "A group of friends enjoy a barbecue on a sandy beach, with one person grilling over a large black grill, while the other sits nearby, laughing and enjoying the camaraderie.",
        "A white and orange cat stands on its hind legs, reaching towards a wicker basket filled with red raspberries on a wooden table in a garden, surrounded by orange flowers and a white teapot, creating a serene and whimsical scene.",
        "A young girl in a yellow dress stands in a grassy field, holding an umbrella and looking at the camera, amidst rain.",
        "This serene bedroom features a white bed with a black canopy, a gray armchair, a black dresser with a mirror, a vase with a plant, a window with white curtains, a rug, and a wooden floor, creating a tranquil and elegant atmosphere.",
        "The image captures the iconic Louvre Museum in Paris, illuminated by warm lights against a dark sky, with the iconic glass pyramid in the center, surrounded by ornate buildings and a large courtyard, showcasing the museum's grandeur and historical significance.",
    ]

    image_urls = [
        "https://github.com/ashvardanian/ashvardanian/blob/master/demos/bbq-on-beach.jpg?raw=true",
        "https://github.com/ashvardanian/ashvardanian/blob/master/demos/cat-in-garden.jpg?raw=true",
        "https://github.com/ashvardanian/ashvardanian/blob/master/demos/girl-and-rain.jpg?raw=true",
        "https://github.com/ashvardanian/ashvardanian/blob/master/demos/light-bedroom-furniture.jpg?raw=true",
        "https://github.com/ashvardanian/ashvardanian/blob/master/demos/louvre-at-night.jpg?raw=true",
    ]

    text_embeddings = []
    image_embeddings = []

    for text, image_url in zip(texts, image_urls):
        # Download and open the image
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))

        # Get embeddings
        text_embedding = text_to_embedding(text)
        image_embedding = image_to_embedding(image)

        text_embeddings.append(text_embedding)
        image_embeddings.append(image_embedding)

    # Evaluate cosine similarity
    for i in range(len(texts)):
        pair_similarity = cosine_similarity(text_embeddings[i], image_embeddings[i])
        other_text_similarities = [
            cosine_similarity(text_embeddings[j], image_embeddings[i]) for j in range(len(texts)) if j != i
        ]
        other_image_similarities = [
            cosine_similarity(text_embeddings[i], image_embeddings[j]) for j in range(len(texts)) if j != i
        ]

        assert pair_similarity > max(
            other_text_similarities
        ), "Text should be more similar to its corresponding image than to other images."
        assert pair_similarity > max(
            other_image_similarities
        ), "Image should be more similar to its corresponding text than to other texts."


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

    # Test if the model outputs actually make sense
    cross_references_image_and_text_embeddings(
        lambda text: model.encode_text(processor.preprocess_text(text)),
        lambda image: model.encode_image(processor.preprocess_image(image)),
    )


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
@pytest.mark.parametrize("model_name", onnx_models)
@pytest.mark.parametrize("device", ["CPUExecutionProvider"])
def test_onnx_one_embedding(model_name: str, device: str):

    from uform.onnx_encoders import ExecutionProviderError

    try:

        model, processor = uform.get_model_onnx(model_name, token=token, device=device)
        text = "a small red panda in a zoo"
        image_path = "assets/unum.png"

        image = Image.open(image_path)
        image_data = processor.preprocess_image(image)
        text_data = processor.preprocess_text(text)

        image_features, image_embedding = model.encode_image(image_data, return_features=True)
        text_features, text_embedding = model.encode_text(text_data, return_features=True)

        assert image_embedding.shape[0] == 1, "Image embedding batch size is not 1"
        assert text_embedding.shape[0] == 1, "Text embedding batch size is not 1"

        # Test if the model outputs actually make sense
        cross_references_image_and_text_embeddings(
            lambda text: model.encode_text(processor.preprocess_text(text)),
            lambda image: model.encode_image(processor.preprocess_image(image)),
        )

    except ExecutionProviderError as e:
        pytest.skip(f"Execution provider error: {e}")


@pytest.mark.skipif(not onnx_available, reason="ONNX is not installed")
@pytest.mark.parametrize("model_name", onnx_models)
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("device", ["CPUExecutionProvider"])
def test_onnx_many_embeddings(model_name: str, batch_size: int, device: str):

    from uform.onnx_encoders import ExecutionProviderError

    try:

        model, processor = uform.get_model_onnx(model_name, token=token, device=device)
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
