import pytest
from PIL import Image

# PyTorch is a very heavy dependency, so we may want to skip these tests if it's not installed
try:
    import torch

    torch_available = True
except:
    torch_available = False

torch_hf_models = [
    "unum-cloud/uform-gen2-qwen-500m",
]


@pytest.mark.skipif(not torch_available, reason="PyTorch is not installed")
@pytest.mark.parametrize("model_name", torch_hf_models)
def test_one_conversation(model_name: str):
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    prompt = "Describe the image in great detail."
    image = Image.open("assets/unum.png")

    inputs = processor(text=[prompt], images=[image], return_tensors="pt")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=10,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    decoded_text = processor.batch_decode(output[:, prompt_len:])[0]

    assert len(decoded_text), "No text was generated from the model."


@pytest.mark.skipif(not torch_available, reason="PyTorch is not installed")
@pytest.mark.parametrize("model_name", torch_hf_models)
@pytest.mark.parametrize("batch_size", [1, 2])
def test_many_conversations(model_name: str, batch_size: int):

    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    prompt = "Describe the image in great detail."
    image = Image.open("assets/unum.png")

    texts = [prompt] * batch_size
    images = [image] * batch_size
    inputs = processor(text=texts, images=images, return_tensors="pt")

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            use_cache=True,
            max_new_tokens=10,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    decoded_texts = processor.batch_decode(output[:, prompt_len:])

    assert all(len(decoded_text) for decoded_text in decoded_texts), "No text was generated from the model."
