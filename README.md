<h1 align="center">UForm</h1>
<h3 align="center">
Pocket-Sized Multimodal AI<br/>
For Content Understanding and Generation<br/>
</h3>
<br/>

<p align="center">
<a href="https://discord.gg/jsMURnSFM2"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/discord.svg" alt="Discord"></a>
&nbsp; &nbsp; &nbsp;
<a href="https://www.linkedin.com/company/unum-cloud/"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/linkedin.svg" alt="LinkedIn"></a>
&nbsp; &nbsp; &nbsp;
<a href="https://twitter.com/unum_cloud"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/twitter.svg" alt="Twitter"></a>
&nbsp; &nbsp; &nbsp;
<a href="https://unum.cloud/post"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/blog.svg" alt="Blog"></a>
&nbsp; &nbsp; &nbsp;
<a href="https://github.com/unum-cloud/uform"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/github.svg" alt="GitHub"></a>
</p>

<p align="center">
Multimodal Embeddings from 64 to 768 Dimensions • 1B Parameter Chat
<br/>
Short Texts • Images • 🔜 Video Clips
<br/>
PyTorch • ONNX
</p>

---

![](https://github.com/ashvardanian/usearch-images/blob/main/assets/uform-gen-preview.jpg?raw=true)

Welcome to UForm, a __multimodal__ AI library that's as versatile as it is efficient.
UForm [tiny embedding models](#encoder) will help you understand and search visual and textual content across various languages.
UForm [small generative models](#decoder), on the other hand, don't only support conversational and chat use-cases, but are also capable of image captioning and Visual Question Answering (VQA).
With compact __custom pre-trained transformer models__, this can run anywhere from your server farm down to your smartphone.

## Features

- __Tiny Embeddings__: 64-dimensional [Matryoshaka][matryoshka]-style embeddings for extremely fast [search][usearch].
- __Throughput__: Thanks to the small size, the inference speed is [2-4x faster](#speed) than competitors.
- __Portable__: Models come with native ONNX support, making them easy to deploy on any platform.
- __Quantization Aware__: Down-cast embeddings from `f32` to `i8` without losing much recall.
- __Multilingual__: Trained on a balanced dataset, the recall is great across over [20 languages](#evaluation).

[usearch]: https://github.com/unum-cloud/usearch
[matryoshka]: https://arxiv.org/abs/2205.13147

## Models

### Embedding Models

| Model                                    | Parameters | Languages |                                 Architecture |
| :--------------------------------------- | ---------: | --------: | -------------------------------------------: |
| [`uform-vl-english-large`][model-e-l] 🆕  |       365M |         1 | 6 text layers, ViT-L/14, 6 multimodal layers |
| [`uform-vl-english`][model-e]            |       143M |         1 | 2 text layers, ViT-B/16, 2 multimodal layers |
| [`uform-vl-english-small`][model-e-s] 🆕  |        79M |         1 | 2 text layers, ViT-S/16, 2 multimodal layers |
| [`uform-vl-multilingual-v2`][model-m-v2] |       206M |        21 | 8 text layers, ViT-B/16, 4 multimodal layers |
| [`uform-vl-multilingual`][model-m]       |       206M |        12 | 8 text layers, ViT-B/16, 4 multimodal layers |

[model-e-l]: https://huggingface.co/unum-cloud/uform-vl-english-large/
[model-e]: https://huggingface.co/unum-cloud/uform-vl-english/
[model-e-s]: https://huggingface.co/unum-cloud/uform-vl-english-small/
[model-m]: https://huggingface.co/unum-cloud/uform-vl-multilingual/
[model-m-v2]: https://huggingface.co/unum-cloud/uform-vl-multilingual-v2/

### Generative Models

| Model                              | Parameters |                     Purpose |           Architecture |
| :--------------------------------- | ---------: | --------------------------: | ---------------------: |
| [`uform-gen2-dpo`][model-g2] 🆕     |       1.2B | Chat, Image Captioning, VQA | qwen1.5-0.5B, ViT-H/14 |
| [`uform-gen2-qwen-500m`][model-g2] |       1.2B | Chat, Image Captioning, VQA | qwen1.5-0.5B, ViT-H/14 |
| [`uform-gen`][model-g1]            |       1.5B |       Image Captioning, VQA |   llama-1.3B, ViT-B/16 |

[model-g2]: https://huggingface.co/unum-cloud/uform-gen2-qwen-500m/
[model-g1]: https://huggingface.co/unum-cloud/uform-gen/

## Producing Embeddings

Add UForm to your dependencies list, or just install it locally:

```bash
pip install uform
```

Then, you can use the following code to get embeddings for text and images.
You can do that either with the PyTorch reference model or the lighter cross-platform ONNX weights.

```python
import uform
from PIL import Image

# If you want to use the PyTorch model
model, processor = uform.get_model('unum-cloud/uform-vl-english-large') # Just English
model, processor = uform.get_model('unum-cloud/uform-vl-multilingual-v2') # 21 Languages

# If you want to use the light-weight portable ONNX model
# Available combinations: cpu & fp32, gpu & fp32, gpu & fp16
# Check out Unum's Hugging Face space for more details: https://huggingface.co/unum-cloud
model, processor = uform.get_model_onnx('unum-cloud/uform-vl-english-small', 'cpu', 'fp32')
model, processor = uform.get_model_onnx('unum-cloud/uform-vl-english-large', 'gpu', 'fp16')

text = 'a small red panda in a zoo'
image = Image.open('red_panda.jpg')

image_data = processor.preprocess_image(image)
text_data = processor.preprocess_text(text)

image_features, image_embedding = model.encode_image(image_data, return_features=True)
text_features, text_embedding = model.encode_text(text_data, return_features=True)
```

To search for similar items, the embeddings can be compared using cosine similarity.
The resulting value will fall within the range of `-1` to `1`, where `1` indicates a high likelihood of a match.
PyTorch provides a built-in function for calculating cosine similarity, while for ONNX, you can use NumPy.

```python
import torch.nn.functional as F

similarity = F.cosine_similarity(image_embedding, text_embedding)
```

ONNX has no such function, but you can calculate the cosine similarity using [SimSIMD](https://github.com/ashvardanian/simsimd) or manually, with NumPy:

```python
import numpy as np

image_embedding = image_embedding / np.linalg.norm(image_embedding, keepdims=True, axis=1)
text_embedding = text_embedding / np.linalg.norm(text_embedding, keepdims=True, axis=1)
similarity = (image_embedding * text_embedding).sum(axis=1)
```

### Reranking

Once the list of nearest neighbors (best matches) is obtained, the joint multimodal embeddings, created from both text and image features, can be used to better rerank (reorder) the list.
The model can calculate a "matching score" that falls within the range of `[0, 1]`, where `1` indicates a high likelihood of a match.

```python
score, joint_embedding = model.encode_multimodal(
    image_features=image_features,
    text_features=text_features,
    attention_mask=text_data['attention_mask'],
    return_scores=True,
)
```

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

### Compact Packaging

PyTorch is a heavy dependency to carry, especially if you run on Edge or IoT devices.
Using vanilla ONNX runtime, one can significantly reduce memory consumption and deployment latency.

```sh
$ conda create -n uform_torch python=3.10 -y
$ conda create -n uform_onnx python=3.10 -y
$ conda activate uform_torch && pip install -e ".[torch]" && conda deactivate
$ conda activate uform_onnx && pip install -e ".[onnx]" && conda deactivate
$ du -sh $(conda info --envs | grep 'uform_torch' | awk '{print $2}')
> 5.2G    ~/conda/envs/uform_torch
$ du -sh $(conda info --envs | grep 'uform_onnx' | awk '{print $2}')
> 461M    ~/conda/envs/uform_onnx
```

Most of that weight can be further reduced down to 100 MB for both the model and the runtime.
You can pick one of many supported [ONNX execution providers][onnx-providers], which includes XNNPACK, CUDA and TensorRT for Nvidia GPUs, OpenVINO on Intel, DirectML on Windows, ROCm on AMD, CoreML on Apple devices, and more to come.

[onnx-providers]: https://onnxruntime.ai/docs/execution-providers/

---

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

## Chat, Image Captioning and Question Answering

UForm generative models are fully compatible with the Hugging Face Transformers library, and can be used without installing the UForm library.
Those models can be used to caption images or power multimodal chat experiences.

```python
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained('unum-cloud/uform-gen2-qwen-500m', trust_remote_code=True)
processor = AutoProcessor.from_pretrained('unum-cloud/uform-gen2-qwen-500m', trust_remote_code=True)

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

You can check examples of different prompts in our [demo space](https://huggingface.co/spaces/unum-cloud/uform-gen2-qwen-500m-demo)


### Image Captioning and Question Answering

__It is the instruction for the first version of UForm-Gen model. We highly recommend you use the new model, instructions for which you can find above.__


The generative model can be used to caption images, summarize their content, or answer questions about them.
The exact behavior is controlled by prompts.

```python
from uform.gen_model import VLMForCausalLM, VLMProcessor

model = VLMForCausalLM.from_pretrained('unum-cloud/uform-gen')
processor = VLMProcessor.from_pretrained('unum-cloud/uform-gen')

# [cap] Narrate the contents of the image with precision.
# [cap] Summarize the visual content of the image.
# [vqa] What is the main subject of the image?
prompt = '[cap] Summarize the visual content of the image.'
image = Image.open('zebra.jpg')

inputs = processor(texts=[prompt], images=[image], return_tensors='pt')
with torch.inference_mode():
     output = model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        max_new_tokens=128,
        eos_token_id=32001,
        pad_token_id=processor.tokenizer.pad_token_id
    )

prompt_len = inputs['input_ids'].shape[1]
decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
```

### Multimodal Chat

The generative models can be used for chat-like experiences, where the user can provide both text and images as input.
To use that feature, you can start with the following CLI command:

```bash
uform-chat --model unum-cloud/uform-gen-chat --image=zebra.jpg
uform-chat --model unum-cloud/uform-gen-chat \
    --image="https://bit.ly/3tIVg9M" \
    --device="cuda:0" \
    --fp16
```

### Multi-GPU

To achieve higher throughput, you can launch UForm on multiple GPUs.
For that pick the encoder of the model you want to run in parallel (`text_encoder` or `image_encoder`), and wrap it in `nn.DataParallel` (or `nn.DistributedDataParallel`).

```python
import uform

model, processor = uform.get_model('unum-cloud/uform-vl-english')
model_image = nn.DataParallel(model.image_encoder)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_image.to(device)

_, res = model_image(images, 0)
```

## Evaluation

### Embedding Models

Few retrieval benchmarks exist for multimodal embeddings.
The most famous ones for English are "MS-COCO" and "Flickr30k".
Evaluating `uform-vl-english` model, one can expect the following numbers for search quality.

| Dataset  | Recall @ 1 | Recall @ 5 | Recall @ 10 |
| :------- | ---------: | ---------: | ----------: |
| Flickr   |      0.727 |      0.915 |       0.949 |
| MS-COCO¹ |      0.510 |      0.761 |       0.838 |


For multilingual benchmarks, we've created the [`unum-cloud/coco-sm`](https://github.com/unum-cloud/coco-sm) repository².
Evaluating the `unum-cloud/uform-vl-multilingual-v2` model, one can expect the following metrics for text-to-image search, compared against `xlm-roberta-base-ViT-B-32` [OpenCLIP](https://github.com/mlfoundations/open_clip) model.

| Language  | OpenCLIP @ 1 | UForm @ 1 | OpenCLIP @ 5 | UForm @ 5 | OpenCLIP @ 10 | UForm @ 10 | Speakers |
| :-------- | -----------: | --------: | -----------: | --------: | ------------: | ---------: | -------: |
| English 🇺🇸 |     __37.8__ |      37.7 |         63.5 |  __65.0__ |          73.5 |   __75.9__ |  1'452 M |
| Chinese 🇨🇳 |         27.3 |  __32.2__ |         51.3 |  __59.0__ |          62.1 |   __70.5__ |  1'118 M |
| Hindi 🇮🇳   |         20.7 |  __31.3__ |         42.5 |  __57.9__ |          53.7 |   __69.6__ |    602 M |
| Spanish 🇪🇸 |         32.6 |  __35.6__ |         58.0 |  __62.8__ |          68.8 |   __73.7__ |    548 M |
| Arabic 🇸🇦  |         22.7 |  __31.7__ |         44.9 |  __57.8__ |          55.8 |   __69.2__ |    274 M |
| French 🇫🇷  |         31.3 |  __35.4__ |         56.5 |  __62.6__ |          67.4 |   __73.3__ |    274 M |


<details>
<summary>All languages.</summary>
<br>

| Language             | OpenCLIP @ 1 |    UForm @ 1 | OpenCLIP @ 5 |    UForm @ 5 | OpenCLIP @ 10 |   UForm @ 10 | Speakers |
| :------------------- | -----------: | -----------: | -----------: | -----------: | ------------: | -----------: | -------: |
| Arabic 🇸🇦             |         22.7 |     __31.7__ |         44.9 |     __57.8__ |          55.8 |     __69.2__ |    274 M |
| Armenian 🇦🇲           |          5.6 |     __22.0__ |         14.3 |     __44.7__ |          20.2 |     __56.0__ |      4 M |
| Chinese 🇨🇳            |         27.3 |     __32.2__ |         51.3 |     __59.0__ |          62.1 |     __70.5__ |  1'118 M |
| English 🇺🇸            |     __37.8__ |         37.7 |         63.5 |     __65.0__ |          73.5 |     __75.9__ |  1'452 M |
| French 🇫🇷             |         31.3 |     __35.4__ |         56.5 |     __62.6__ |          67.4 |     __73.3__ |    274 M |
| German 🇩🇪             |         31.7 |     __35.1__ |         56.9 |     __62.2__ |          67.4 |     __73.3__ |    134 M |
| Hebrew 🇮🇱             |         23.7 |     __26.7__ |         46.3 |     __51.8__ |          57.0 |     __63.5__ |      9 M |
| Hindi 🇮🇳              |         20.7 |     __31.3__ |         42.5 |     __57.9__ |          53.7 |     __69.6__ |    602 M |
| Indonesian 🇮🇩         |         26.9 |     __30.7__ |         51.4 |     __57.0__ |          62.7 |     __68.6__ |    199 M |
| Italian 🇮🇹            |         31.3 |     __34.9__ |         56.7 |     __62.1__ |          67.1 |     __73.1__ |     67 M |
| Japanese 🇯🇵           |         27.4 |     __32.6__ |         51.5 |     __59.2__ |          62.6 |     __70.6__ |    125 M |
| Korean 🇰🇷             |         24.4 |     __31.5__ |         48.1 |     __57.8__ |          59.2 |     __69.2__ |     81 M |
| Persian 🇮🇷            |         24.0 |     __28.8__ |         47.0 |     __54.6__ |          57.8 |     __66.2__ |     77 M |
| Polish 🇵🇱             |         29.2 |     __33.6__ |         53.9 |     __60.1__ |          64.7 |     __71.3__ |     41 M |
| Portuguese 🇵🇹         |         31.6 |     __32.7__ |         57.1 |     __59.6__ |          67.9 |     __71.0__ |    257 M |
| Russian 🇷🇺            |         29.9 |     __33.9__ |         54.8 |     __60.9__ |          65.8 |     __72.0__ |    258 M |
| Spanish 🇪🇸            |         32.6 |     __35.6__ |         58.0 |     __62.8__ |          68.8 |     __73.7__ |    548 M |
| Thai 🇹🇭               |         21.5 |     __28.7__ |         43.0 |     __54.6__ |          53.7 |     __66.0__ |     61 M |
| Turkish 🇹🇷            |         25.5 |     __33.0__ |         49.1 |     __59.6__ |          60.3 |     __70.8__ |     88 M |
| Ukranian 🇺🇦           |         26.0 |     __30.6__ |         49.9 |     __56.7__ |          60.9 |     __68.1__ |     41 M |
| Vietnamese 🇻🇳         |         25.4 |     __28.3__ |         49.2 |     __53.9__ |          60.3 |     __65.5__ |     85 M |
|                      |              |              |              |              |               |              |          |
| Mean                 |     26.5±6.4 | __31.8±3.5__ |     49.8±9.8 | __58.1±4.5__ |     60.4±10.6 | __69.4±4.3__ |        - |
| Google Translate     |     27.4±6.3 | __31.5±3.5__ |     51.1±9.5 | __57.8±4.4__ |     61.7±10.3 | __69.1±4.3__ |        - |
| Microsoft Translator |     27.2±6.4 | __31.4±3.6__ |     50.8±9.8 | __57.7±4.7__ |     61.4±10.6 | __68.9±4.6__ |        - |
| Meta NLLB            |     24.9±6.7 | __32.4±3.5__ |    47.5±10.3 | __58.9±4.5__ |     58.2±11.2 | __70.2±4.3__ |        - |

</details>

### Generative Models

| Model                | LLM Size |  SQA |    MME | MMBench | Average¹ |
| :------------------- | -------: | ---: | -----: | ------: | -------: |
| UForm-Gen2-Qwen-500m |     0.5B | 45.5 |  880.1 |    42.0 |    29.31 |
| MobileVLM v2         |     1.4B | 52.1 | 1302.8 |    57.7 |    36.81 |
| LLaVA-Phi            |     2.7B | 68.4 | 1335.1 |    59.8 |    42.95 |

For captioning evaluation we measure CLIPScore and RefCLIPScore³.

| Model                               | Size | Caption Length | CLIPScore | RefCLIPScore |
| :---------------------------------- | ---: | -------------: | --------: | -----------: |
| `llava-hf/llava-1.5-7b-hf`          |   7B |           Long |     0.878 |        0.529 |
| `llava-hf/llava-1.5-7b-hf`          |   7B |          Short |     0.886 |        0.531 |
|                                     |
| `Salesforce/instructblip-vicuna-7b` |   7B |           Long |     0.902 |        0.534 |
| `Salesforce/instructblip-vicuna-7b` |   7B |          Short |     0.848 |        0.523 |
|                                     |
| `unum-cloud/uform-gen`              | 1.5B |           Long |     0.847 |        0.523 |
| `unum-cloud/uform-gen`              | 1.5B |          Short |     0.842 |        0.522 |
|                                     |
| `unum-cloud/uform-gen-chat`         | 1.5B |           Long |     0.860 |        0.525 |
| `unum-cloud/uform-gen-chat`         | 1.5B |          Short |     0.858 |        0.525 |

Results for VQAv2 evaluation.

| Model                      | Size | Accuracy |
| :------------------------- | ---: | -------: |
| `llava-hf/llava-1.5-7b-hf` |   7B |     78.5 |
| `unum-cloud/uform-gen`     | 1.5B |     66.5 |

<br/>

> ¹ Train split was in training data. <br/>
> ² Lacking a broad enough evaluation dataset, we translated the [COCO Karpathy test split](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits) with multiple public and proprietary translation services, averaging the scores across all sets, and breaking them down in the bottom section. <br/>
> ³ We used `apple/DFN5B-CLIP-ViT-H-14-378` CLIP model.

## Speed

On Nvidia RTX 3090, the following performance is expected on text encoding.

| Model                                     | Multilingual |                  Speed |    Speedup |
| :---------------------------------------- | -----------: | ---------------------: | ---------: |
| `bert-base-uncased`                       |           No | 1'612 sequences/second |            |
| `distilbert-base-uncased`                 |           No | 3'174 sequences/second |     x 1.96 |
| `sentence-transformers/all-MiniLM-L12-v2` |      __Yes__ | 3'604 sequences/second |     x 2.24 |
| `unum-cloud/uform-vl-multilingual-v2`     |      __Yes__ | 6'809 sequences/second | __x 4.22__ |

On Nvidia RTX 3090, the following performance is expected on text token generation using `float16`, equivalent PyTorch settings, and greedy decoding.

| Model                               | Size |               Speed |   Speedup |
| :---------------------------------- | ---: | ------------------: | --------: |
| `llava-hf/llava-1.5-7b-hf`          |   7B |  ~ 40 tokens/second |           |
| `Salesforce/instructblip-vicuna-7b` |   7B |  ~ 40 tokens/second |           |
| `unum-cloud/uform-gen`              | 1.5B | ~ 140 tokens/second | __x 3.5__ |

Given the small size of the model it also work well on mobile devices.
On Apple M2 Arm chips the energy efficiency of inference can exceed that of the RTX 3090 GPU and other Ampere-generation cards.

| Device                 |               Speed | Device TDP |        Efficiency |
| :--------------------- | ------------------: | ---------: | ----------------: |
| Nvidia RTX 3090        | ~ 140 tokens/second |     < 350W | 0.40 tokens/joule |
| Apple M2 Pro unplugged |  ~ 19 tokens/second |      < 20W | 0.95 tokens/joule |
| Apple M2 Max unplugged |  ~ 38 tokens/second |      < 36W | 1.06 tokens/joule |
| Apple M2 Max plugged   |  ~ 56 tokens/second |      < 89W | 0.63 tokens/joule |

> [!WARNING]
> The above numbers are for reference only and are not guaranteed to be accurate.

## License

All models come under the same license as the code - Apache 2.0.
