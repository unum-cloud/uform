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

```py
from uform import get_model, Modality

import requests
from io import BytesIO
from PIL import Image

model_name = 'unum-cloud/uform3-image-text-english-small'
modalities = [Modality.TEXT_ENCODER, Modality.IMAGE_ENCODER]
processors, models = get_model(model_name, modalities=modalities)

model_text = models[Modality.TEXT_ENCODER]
model_image = models[Modality.IMAGE_ENCODER]
processor_text = processors[Modality.TEXT_ENCODER]
processor_image = processors[Modality.IMAGE_ENCODER]

# Download the image
text = 'a cityscape bathed in the warm glow of the sun, with varied architecture and a towering, snow-capped mountain rising majestically in the background'
image_url = 'https://media-cdn.tripadvisor.com/media/photo-s/1b/28/6b/53/lovely-armenia.jpg'
image_url = Image.open(BytesIO(requests.get(image_url).content))

# The actual inference
image_data = processor_image(image)
text_data = processor_text(text)
image_features, image_embedding = model_image.encode(image_data, return_features=True)
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

You can check examples of different prompts in our demo spaces:

- for [`uform-gen2-qwen-500m`](https://huggingface.co/spaces/unum-cloud/uform-gen2-qwen-500m-demo)
- for [`uform-gen2-dpo`](https://huggingface.co/spaces/unum-cloud/uform-gen2-qwen-500m-dpo-demo)

## Technical Details

### Down-casting, Quantization, Matryoshka, and Slicing

Depending on the application, the embeddings can be down-casted to smaller numeric representations without losing much recall.
Switching from `f32` to `f16` is recommended in almost all cases, unless you are running on very old hardware without half-precision support.
Switching to `i8` with linear scaling is also possible, but will be noticeable in the recall on larger collections with millions of searchable entries.
Similarly, for higher-dimensional embeddings (512 or 768), a common strategy is to quantize them into single-bit representations for faster search.

```python
import numpy as np

f32_embedding: np.ndarray = model.encode_text(text_data, return_features=False).detach().cpu().numpy()
f16_embedding: np.ndarray = f32_embedding.astype(np.float16)
i8_embedding: np.ndarray = (f32_embedding * 127).astype(np.int8)
b1_embedding: np.ndarray = np.packbits((f32_embedding > 0).astype(np.uint8))
```

Alternative approach to quantization is to use the Matryoshka embeddings, where the embeddings are sliced into smaller parts, and the search is performed in a hierarchical manner.

```python
import numpy as np

large_embedding: np.ndarray = model.encode_text(text_data, return_features=False).detach().cpu().numpy()
small_embedding: np.ndarray = large_embedding[:, :256]
tiny_embedding: np.ndarray = large_embedding[:, :64]
```

Both approaches are natively supported by the [USearch][github-usearch] vector-search engine and the [SimSIMD][github-simsimd] numerics libraries.
When dealing with small collections (up to millions of entries) and looking for low-latency cosine distance calculations, you can [achieve 5x-2500x performance improvement][report-simsimd] over Torch, NumPy, SciPy, and vanilla Python using SimSIMD.

```python
from simsimd import cosine, hamming

distance: float = cosine(f32_embedding, f32_embedding) # 32x SciPy performance on Apple M2 CPU
distance: float = cosine(f16_embedding, f16_embedding) # 79x SciPy performance on Apple M2 CPU
distance: float = cosine(i8_embedding, i8_embedding) # 133x SciPy performance on Apple M2 CPU
distance: float = hamming(b1_embedding, b1_embedding) # 17x SciPy performance on Apple M2 CPU
```

Similarly, when dealing with large collections (up to billions of entries per server) and looking for high-throughput search, you can [achieve 100x performance improvement][report-usearch] over FAISS and other vector-search solutions using USearch.
Here are a couple of examples:

```python
from usearch.index import Index

f32_index = Index(ndim=64, metric='cos', dtype='f32') # for Matryoshka embeddings
f16_index = Index(ndim=64, metric='cos', dtype='f16') # for Matryoshka embeddings
i8_index = Index(ndim=256, metric='cos', dtype='i8') # for quantized embeddings
b1_index = Index(ndim=768, metric='hamming', dtype='b1') # for binary embeddings
```

[github-usearch]: https://github.com/unum-cloud/usearch
[github-simsimd]: https://github.com/ashvardanian/simsimd
[report-usearch]: https://www.unum.cloud/blog/2023-11-07-scaling-vector-search-with-intel
[report-simsimd]: https://ashvardanian.com/posts/python-c-assembly-comparison/

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
