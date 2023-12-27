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

---

Welcome to UForm, a __multimodal__ AI library that's as versatile as it is efficient.
UForm [tiny embedding models](#encoder) will help you understand and search visual and textual content across various languages.
UForm [small generative models](#decoder), on the other hand, don't only support conversational and chat use-cases, but are also capable of image captioning and Visual Question Answering (VQA).
With compact __custom pre-trained transformer models__, this can run anywhere from your server farm down to your smartphone.

## Features

* __Throughput__: Thanks to the small size, the inference speed is [2-4x faster](#speed) than competitors.
* __Tiny Embeddings__: With just 256 dimensions, our vectors are 2-3x quicker to [search][usearch] than from CLIP-like models.
* __Quantization Aware__: Our embeddings can be downcasted from `f32` to `i8` without losing much recall.
* __Multilingual__: Trained on a balanced dataset, the recall is great across over [20 languages](#evaluation).
* __Hardware Friendly__: Whether it's Apple's CoreML or ONNX, [we've got you covered][onnx].

[usearch]: https://github.com/unum-cloud/usearch
[onnx]: https://huggingface.co/unum-cloud/uform-coreml-onnx

## Models

### Embedding Models

| Model                                    | Parameters | Languages |                                 Architecture |
| :--------------------------------------- | ---------: | --------: | -------------------------------------------: |
| [`uform-vl-english`][model-e]            |       143M |         1 | 2 text layers, ViT-B/16, 2 multimodal layers |
| [`uform-vl-multilingual-v2`][model-m-v2] |       206M |        21 | 8 text layers, ViT-B/16, 4 multimodal layers |
| [`uform-vl-multilingual`][model-m]       |       206M |        12 | 8 text layers, ViT-B/16, 4 multimodal layers |

[model-e]: https://huggingface.co/unum-cloud/uform-vl-english/
[model-m]: https://huggingface.co/unum-cloud/uform-vl-multilingual/
[model-m-v2]: https://huggingface.co/unum-cloud/uform-vl-multilingual-v2/

### Generative Models

| Model                        | Parameters |               Purpose |         Architecture |
| :--------------------------- | ---------: | --------------------: | -------------------: |
| [`uform-gen`][model-g]       |       1.5B | Image Captioning, VQA | llama-1.3B, ViT-B/16 |
| [`uform-gen-chat`][model-gc] |       1.5B |       Multimodal Chat | llama-1.3B, ViT-B/16 |

[model-g]: https://huggingface.co/unum-cloud/uform-gen/
[model-gc]: https://huggingface.co/unum-cloud/uform-gen-chat/


## Quick Start

Once you `pip install uform`, fetching the models is as easy as:

```python
import uform

model = uform.get_model('unum-cloud/uform-vl-english') # Just English
model = uform.get_model('unum-cloud/uform-vl-multilingual-v2') # 21 Languages
```

### Producing Embeddings

```python
from PIL import Image
import torch.nn.functional as F

text = 'a small red panda in a zoo'
image = Image.open('red_panda.jpg')

image_data = model.preprocess_image(image)
text_data = model.preprocess_text(text)

image_features, image_embedding = model.encode_image(image_data, return_features=True)
text_features, text_embedding = model.encode_text(text_data, return_features=True)

similarity = F.cosine_similarity(image_embedding, text_embedding)
```

To search for similar items, the embeddings can be compared using cosine similarity.
The resulting value will fall within the range of `-1` to `1`, where `1` indicates a high likelihood of a match. 
Once the list of nearest neighbors (best matches) is obtained, the joint multimodal embeddings, created from both text and image features, can be used to better rerank (reorder) the list.
The model can calculate a "matching score" that falls within the range of `[0, 1]`, where `1` indicates a high likelihood of a match.

```python
joint_embedding = model.encode_multimodal(
    image_features=image_features,
    text_features=text_features,
    attention_mask=text_data['attention_mask']
)
score = model.get_matching_scores(joint_embedding)
```

### Image Captioning and Question Answering

The generative model can be used to caption images, summarize their content, or answer questions about them.
The exact behavior is controlled by prompts.

```python
from uform.gen_model import VLMForCausalLM, VLMProcessor

model = VLMForCausalLM.from_pretrained("unum-cloud/uform-gen")
processor = VLMProcessor.from_pretrained("unum-cloud/uform-gen")

# [cap] Narrate the contents of the image with precision.
# [cap] Summarize the visual content of the image.
# [vqa] What is the main subject of the image?
prompt = "[cap] Summarize the visual content of the image."
image = Image.open("zebra.jpg")

inputs = processor(texts=[prompt], images=[image], return_tensors="pt")
with torch.inference_mode():
     output = model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        max_new_tokens=128,
        eos_token_id=32001,
        pad_token_id=processor.tokenizer.pad_token_id
    )

prompt_len = inputs["input_ids"].shape[1]
decoded_text = processor.batch_decode(output[:, prompt_len:])[0]
```

### Multimodal Chat

The generative models can be used for chat-like experiences, where the user can provide both text and images as input.
To use that feature, you can start with the following CLI command:

```bash
uform-chat --model unum-cloud/uform-gen-chat --image_path=zebra.jpg
uform-chat --model unum-cloud/uform-gen-chat --image_path=zebra.jpg --device="cuda:0" --fp16
```

### Multi-GPU

To achieve higher throughput, you can launch UForm on multiple GPUs.
For that pick the encoder of the model you want to run in parallel (`text_encoder` or `image_encoder`), and wrap it in `nn.DataParallel` (or `nn.DistributedDataParallel`).

```python
import uform

model = uform.get_model('unum-cloud/uform-vl-english')
model_image = nn.DataParallel(model.image_encoder)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

On RTX 3090, the following performance is expected on text encoding.

| Model                                     | Multilingual |                  Speed |    Speedup |
| :---------------------------------------- | -----------: | ---------------------: | ---------: |
| `bert-base-uncased`                       |           No | 1'612 sequences/second |            |
| `distilbert-base-uncased`                 |           No | 3'174 sequences/second |     x 1.96 |
| `sentence-transformers/all-MiniLM-L12-v2` |      __Yes__ | 3'604 sequences/second |     x 2.24 |
| `unum-cloud/uform-vl-multilingual-v2`     |      __Yes__ | 6'809 sequences/second | __x 4.22__ |

On RTX 3090, the following performance is expected on text token generation using `float16`, equivalent PyTorch settings, and greedy decoding.

| Model                               | Size |               Speed |   Speedup |
| :---------------------------------- | ---: | ------------------: | --------: |
| `llava-hf/llava-1.5-7b-hf`          |   7B |  ~ 40 tokens/second |           |
| `Salesforce/instructblip-vicuna-7b` |   7B |  ~ 40 tokens/second |           |
| `unum-cloud/uform-gen`              | 1.5B | ~ 140 tokens/second | __x 3.5__ |

## License

All models come under the same license as the code - Apache 2.0.
