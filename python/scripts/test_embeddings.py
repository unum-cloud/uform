import pytest
from PIL import Image
import uform

torch_models = [
    "unum-cloud/uform-vl-english",
    "unum-cloud/uform-vl-multilingual-v2",
]

onnx_models_and_providers = [
    ("unum-cloud/uform-vl-english", "cpu"),
    ("unum-cloud/uform-vl-multilingual-v2", "cpu"),
]


@pytest.mark.parametrize("model_name", torch_models)
def test_one_embedding(model_name: str):
    model, processor = uform.get_model(model_name)
    text = "a small red panda in a zoo"
    image_path = "assets/unum.png"

    image = Image.open(image_path)
    image_data = processor.preprocess_image(image)
    text_data = processor.preprocess_text(text)

    _, image_embedding = model.encode_image(image_data, return_features=True)
    _, text_embedding = model.encode_text(text_data, return_features=True)

    assert image_embedding.shape[0] == 1, "Image embedding batch size is not 1"
    assert text_embedding.shape[0] == 1, "Text embedding batch size is not 1"


@pytest.mark.parametrize("model_name", torch_models)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_many_embeddings(model_name: str, batch_size: int):
    model, processor = uform.get_model(model_name)
    texts = ["a small red panda in a zoo"] * batch_size
    image_paths = ["assets/unum.png"] * batch_size

    images = [Image.open(path) for path in image_paths]
    image_data = processor.preprocess_image(images)
    text_data = processor.preprocess_text(texts)

    image_embeddings = model.encode_image(image_data, return_features=False)
    text_embeddings = model.encode_text(text_data, return_features=False)

    assert image_embeddings.shape[0] == batch_size, "Image embedding is unexpected"
    assert text_embeddings.shape[0] == batch_size, "Text embedding is unexpected"
