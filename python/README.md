# UForm Python SDK

UForm multimodal AI SDK offers a simple way to integrate multimodal AI capabilities into your Python applications.
The SDK doesn't require any deep learning knowledge, PyTorch, or CUDA installation, and can run on almost any hardware.

## Installation

There are several ways to install the UForm Python SDK, depending on the backend you want to use.
PyTorch is by far the heaviest, but the most capable.
ONNX is a lightweight alternative that can run on any CPU, and on some GPUs.

```bash
pip install "uform[torch]"       # For PyTorch
pip install "uform[onnx]"        # For ONNX on CPU
pip install "uform[onnx-gpu]"    # For ONNX on GPU, available for some platforms
pip install "uform[torch,onnx]"  # For PyTorch and ONNX Python tests
```

## Quick Start

### Embeddings

Load the model:

```py
from uform import get_model, Modality

model_name = 'unum-cloud/uform3-image-text-english-small'
modalities = [Modality.TEXT_ENCODER, Modality.IMAGE_ENCODER]
processors, models = get_model(model_name, modalities=modalities)

model_text = models[Modality.TEXT_ENCODER]
model_image = models[Modality.IMAGE_ENCODER]
processor_text = processors[Modality.TEXT_ENCODER]
processor_image = processors[Modality.IMAGE_ENCODER]
```

Embed images:

```py
import requests
from io import BytesIO
from PIL import Image

image_url = 'https://media-cdn.tripadvisor.com/media/photo-s/1b/28/6b/53/lovely-armenia.jpg'
image_url = Image.open(BytesIO(requests.get(image_url).content))
image_data = processor_image(image)
image_features, image_embedding = model_image.encode(image_data, return_features=True)
```

Embed queries:

```py
text = 'a cityscape bathed in the warm glow of the sun, with varied architecture and a towering, snow-capped mountain rising majestically in the background'
text_data = processor_text(text)
text_features, text_embedding = model_text.encode(text_data, return_features=True)
```

### Generative Models

UForm generative models are fully compatible with the Hugging Face Transformers library, and can be used without installing the UForm library.
Those models can be used to caption images or power multimodal chat experiences.

```python
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained('unum-cloud/uform-gen2-dpo', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('unum-cloud/uform-gen2-dpo', trust_remote_code=True)

prompt = 'Question or Instruction'
image = Image.open('image.jpg')

inputs = processor(text=[prompt], images=[image], return_tensors='pt')

with torch.inference_mode():
     output = model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        max_new_tokens=256,
        eos_token_id=151645,
        pad_token_id=processor.tokenizer.pad_token_id
    )
prompt_len = inputs['input_ids'].shape[1]
decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
```

You can check examples of different prompts in our demo Gradio spaces on HuggingFace:

- for [`uform-gen2-qwen-500m`](https://huggingface.co/spaces/unum-cloud/uform-gen2-qwen-500m-demo)
- for [`uform-gen2-dpo`](https://huggingface.co/spaces/unum-cloud/uform-gen2-qwen-500m-dpo-demo)

## Technical Details

### Multi-GPU Parallelism

To achieve higher throughput, you can launch UForm on multiple GPUs.
For that pick the encoder of the model you want to run in parallel, and wrap it in `nn.DataParallel` (or `nn.DistributedDataParallel`).

```python
from uform import get_model, Modality

encoders, processors = uform.get_model('unum-cloud/uform-vl-english-small', backend='torch', device='gpu')

encoder_image = encoders[Modality.IMAGE_ENCODER]
encoder_image = nn.DataParallel(encoder_image)

_, res = encoder_image(images, 0)
```

### ONNX and CUDA

The configuration process may include a few additional steps, depending on the environment.
When using the CUDA and TensorRT backends with CUDA 12 or newer make sure to [install the Nvidia toolkit][install-nvidia-toolkit] and the `onnxruntime-gpu` package from the custom repository.

```sh
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12
pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/
export CUDA_PATH="/usr/local/cuda-12/bin"
export PATH="/usr/local/cuda-12/bin${PATH:+:${PATH}}"
export LD_LIBRARY_PATH="/usr/local/cuda-12/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
pytest python/scripts/ -s -x -Wd -v -k onnx
```

[install-nvidia-toolkit]: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#network-repo-installation-for-ubuntu
