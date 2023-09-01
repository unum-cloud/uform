<h1 align="center">UForm</h1>
<h3 align="center">
Pocket-Sized Multi-Modal Transformers Library<br/>
For Semantic Search & Recommendation Systems<br/>
</h3>
<br/>

<p align="center">
<a href="https://discord.gg/jsMURnSFM2"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/discord.svg" alt="Discord"></a>
&nbsp;&nbsp;&nbsp;
<a href="https://www.linkedin.com/company/unum-cloud/"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/linkedin.svg" alt="LinkedIn"></a>
&nbsp;&nbsp;&nbsp;
<a href="https://twitter.com/unum_cloud"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/twitter.svg" alt="Twitter"></a>
&nbsp;&nbsp;&nbsp;
<a href="https://unum.cloud/post"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/blog.svg" alt="Blog"></a>
&nbsp;&nbsp;&nbsp;
<a href="https://github.com/unum-cloud/uform"><img height="25" src="https://github.com/unum-cloud/.github/raw/main/assets/github.svg" alt="GitHub"></a>
</p>

---

![UForm + USearch + UCall Demo](https://github.com/ashvardanian/usearch-images/raw/main/assets/usearch-images-slow.gif)

Welcome to UForm, a multi-modal AI library that's as versatile as it is efficient.
Imagine encoding text, images, and soon, audio, video, and documents into a shared digital playground.
All of this, with models so compact, they can run anywhere—from your server farm down to your smartphone. 📱💻 
[Check them out on HuggingFace!](https://huggingface.co/unum-cloud) 🤗

## 🌟 Key Features

### 🚀 Speed & Efficiency

- __Tiny Embeddings__: With just 256 dimensions, our embeddings are lean and fast, making your search operations 1.5-3x quicker compared to other CLIP-like models with 512-1024 dimensions.
  
- __Quantization Magic__: Our models are trained to be quantization-aware, letting you downcast embeddings from `f32` to `i8` without losing much accuracy. The result? Smaller search indexes and blazing-fast performance, especially on IoT devices.

### 🌍 Global Reach

- __Balanced Training__: Our models are cosmopolitan, trained on a balanced diet of English and other languages. This gives us an edge in languages often overlooked by other models, from Hebrew and Armenian to Hindi and Arabic.

### 🛠 Versatility

- __Mid-Fusion Tech__: Our models use mid-fusion to align multiple transformer towers, enabling database-like operations on multi-modal data. 

- __Mixed Features__: Thanks to mid-fusion, our models can produce mixed vision+language features, perfect for recommendation systems.

- __Hardware Friendly__: Whether it's CoreML, ONNX, or specialized AI hardware like Graphcore IPUs, we've got you covered.

## 🎓 Architectural Innovation

Inspired by the ALBEF paper by Salesforce, we've pushed the boundaries of pre-training objectives to squeeze more language-vision understanding into smaller models.
Some UForm models were trained on just 4 million samples across 10 GPUs — a __100x reduction in both dataset size and compute budget compared to OpenAI's CLIP__.
While they may not be suited for zero-shot classification tasks, they are your __go-to choice for processing large image datasets or even petabytes of video frame-by-frame__.

### 🤝 Mid-Fusion: The Best of Both Worlds

![Fusion Models](https://raw.githubusercontent.com/unum-cloud/uform/main/assets/model_types_bg.png)

- __Late-Fusion Models__: Great for capturing the big picture but might miss the details. Ideal for large-scale retrieval. OpenAI CLIP is one of those.
- __Early-Fusion Models__: These are your detail-oriented models, capturing fine-grained features. They're usually employed for re-ranking smaller retrieval results.
- __Mid-Fusion Models__: The balanced diet of models. They offer both a unimodal and a multimodal part, capturing both the forest and the trees. The multimodal part enhances the unimodal features with a cross-attention mechanism.

So, if you're looking to navigate the complex world of multi-modal data, UForm is the tiny but mighty companion you've been searching for! 🌈🔍

### New Training Objectives

_Coming soon_

## Installation

Install UForm via pip:

```bash
pip install uform
```

> Note: For versions below 0.3.0, dependencies include transformers and timm.
> Newer versions only require PyTorch and utility libraries.
> For optimal performance, use PyTorch v2.0.0 or above.

## Quick Start

### Loading a Model

```python
import uform

model = uform.get_model('unum-cloud/uform-vl-english') # Just English
model = uform.get_model('unum-cloud/uform-vl-multilingual-v2') # 21 Languages
```

The Multi-Lingual model is much heavier due to 10x larger vocabulary.
So if you only expect English data, take the former for efficiency.
You can also load your own Mid-fusion model.
Just upload it on HuggingFace and pass the model name to `get_model`.

### Encoding Data

```python
from PIL import Image

text = 'a small red panda in a zoo'
image = Image.open('red_panda.jpg')

image_data = model.preprocess_image(image)
text_data = model.preprocess_text(text)

image_embedding = model.encode_image(image_data)
text_embedding = model.encode_text(text_data)
joint_embedding = model.encode_multimodal(image=image_data, text=text_data)
```

### Retrieving Features

```python
image_features, image_embedding = model.encode_image(image_data, return_features=True)
text_features, text_embedding = model.encode_text(text_data, return_features=True)
```

These features can later be used to produce joint multimodal encodings faster, as the first layers of the transformer can be skipped.
Those might be useful for both re-ranking of search results, and recommendation systems.

```python
joint_embedding = model.encode_multimodal(
    image_features=image_features,
    text_features=text_features,
    attention_mask=text_data['attention_mask']
)
```

### Graphcore IPU Inference

First, you will need to setup PopTorch for Graphcore IPUs.
Follow the user [guide](https://docs.graphcore.ai/projects/poptorch-user-guide/en/latest/intro.html).

```python
import poptorch
from PIL import Image

options = poptorch.Options()
options.replicationFactor(1)
options.deviceIterations(4)

model = get_model_ipu('unum-cloud/uform-vl-english').parallelize()
model = poptorch.inferenceModel(model, options=options)

text = 'a small red panda in a zoo'
image = Image.open('red_panda.jpg')
image_data = model.preprocess_image(image)
text_data = model.preprocess_text(text)

image_data = image_data.repeat(4, 1, 1, 1)
text_data = {k: v.repeat(4, 1) for k,v in text_data.items()}

image_features, text_features = model(image_data, text_data)
```

### Remote Procedure Calls for Cloud Deployments

You can also use our larger, faster, better proprietary models deployed in optimized cloud environments.
For that, please, choose the cloud of liking, search the marketplace for "Unum UForm" and reinstall UForm with optional dependencies:

```bash
pip install uform[remote]
```

The only thing that changes after that is calling `get_client` with the IP address of your instance instead of using `get_model` for local usage.

```python
model = uform.get_client('0.0.0.0:7000')
```

__[Please, join our Discord for early access!](https://discord.gg/jsMURnSFM2)__

## Models

### Architecture

| Model                                 | Language Tower | Image Tower | Multimodal Part | Languages |                        URL |
| :------------------------------------ | :------------: | :---------: | :-------------: | :-------: | -------------------------: |
| `unum-cloud/uform-vl-english`         | BERT, 2 layers |  ViT-B/16   |    2 layers     |     1     |    [weights.pt][weights-e] |
| `unum-cloud/uform-vl-multilingual`    | BERT, 8 layers |  ViT-B/16   |    4 layers     |    12     |    [weights.pt][weights-m] |
| `unum-cloud/uform-vl-multilingual-v2` | BERT, 8 layers |  ViT-B/16   |    4 layers     |    21     | [weights.pt][weights-m-v2] |

The multilingual were trained on a language-balanced dataset.
For pre-training, we translated captions with [NLLB](https://github.com/facebookresearch/fairseq/tree/nllb).

[weights-e]: https://huggingface.co/unum-cloud/uform-vl-english/resolve/main/torch_weight.pt
[weights-m]: https://huggingface.co/unum-cloud/uform-vl-multilingual/resolve/main/torch_weight.pt
[weights-m-v2]: https://huggingface.co/unum-cloud/uform-vl-multilingual-v2/resolve/main/torch_weight.pt

### Evaluation

Evaluating the `unum-cloud/uform-vl-multilingual-v2` model, one can expect the following metrics for text-to-image search, compared against `xlm-roberta-base-ViT-B-32` [OpenCLIP](https://github.com/mlfoundations/open_clip) model.
Check out the [`unum-cloud/coco-sm`](https://github.com/unum-cloud/coco-sm) for details.

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

### Performance

On RTX 3090, the following performance is expected from `uform` on text encoding.

| Model                              | Multilingual | Sequences per Second |    Speedup |
| :--------------------------------- | -----------: | -------------------: | ---------: |
| `bert-base-uncased`                |           No |                1'612 |            |
| `distilbert-base-uncased`          |           No |                3'174 |     x 1.96 |
| `sentence-transformers/MiniLM-L12` |          Yes |                3'604 |     x 2.24 |
| `sentence-transformers/MiniLM-L6`  |           No |                6'107 |     x 3.79 |
|                                    |              |                      |            |
| `unum-cloud/uform-vl-multilingual` |          Yes |                6'809 | __x 4.22__ |

## Additional Tooling

There are two options to calculate semantic compatibility between an image and a text: [Cosine Similarity](#cosine-similarity) and [Matching Score](#matching-score).

### Cosine Similarity

```python
import torch.nn.functional as F

similarity = F.cosine_similarity(image_embedding, text_embedding)
```

The `similarity` will belong to the `[-1, 1]` range, `1` meaning the absolute match.

__Pros__:

- Computationally cheap.
- Only unimodal embeddings are required. Unimodal encoding is faster than joint encoding.
- Suitable for retrieval in large collections.

__Cons__:

- Takes into account only coarse-grained features.

### Matching Score 

Unlike cosine similarity, unimodal embedding is not enough.
Joint embedding will be needed, and the resulting `score` will belong to the `[0, 1]` range, `1` meaning the absolute match.

```python
score = model.get_matching_scores(joint_embedding)
```

__Pros__:

- Joint embedding captures fine-grained features.
- Suitable for re-ranking - sorting retrieval result.

__Cons__:

- Resource-intensive.
- Not suitable for retrieval in large collections.

