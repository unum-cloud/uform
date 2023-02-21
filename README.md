<h1 align="center">UForm</h1>
<h3 align="center">
Multi-Modal Inference Library<br/>
For Semantic Search Applications<br/>
</h3>
<br/>

<p align="center">
<a href="https://discord.gg/jsMURnSFM2"><img height="25" src="https://github.com/unum-cloud/ukv/raw/main/assets/icons/discord.svg" alt="Discord"></a>
&nbsp;&nbsp;&nbsp;
<a href="https://www.linkedin.com/company/unum-cloud/"><img height="25" src="https://github.com/unum-cloud/ukv/raw/main/assets/icons/linkedin.svg" alt="LinkedIn"></a>
&nbsp;&nbsp;&nbsp;
<a href="https://twitter.com/unum_cloud"><img height="25" src="https://github.com/unum-cloud/ukv/raw/main/assets/icons/twitter.svg" alt="Twitter"></a>
&nbsp;&nbsp;&nbsp;
<a href="https://unum.cloud/post"><img height="25" src="https://github.com/unum-cloud/ukv/raw/main/assets/icons/blog.svg" alt="Blog"></a>
&nbsp;&nbsp;&nbsp;
<a href="https://github.com/unum-cloud/uform"><img height="25" src="https://github.com/unum-cloud/ukv/raw/main/assets/icons/github.svg" alt="GitHub"></a>
</p>

---

UForm is Multi-Modal Modal Inference package, designed to encode Multi-Lingual Texts, Images, and, soon, Audio, Video, and Documents, into a shared vector space!
It extends HuggingFace `transfromers` to ...

## Installation

```bash
pip install uform
```

## Usage

To load model use following code:

```python
import uform

model = uform.get_model('eng') # for the monolingual model
model = uform.get_model('multilingual') # for the multilingual model
```

To encode data:

```python
from PIL import Image

# data preprocessing
img = Image.open('red_panda.jpg')
img = model.img_transform(img).unsqueeze(0)
text = 'a small red panda in a zoo'
text = model.text_transform(text)

# image encoding
image_embedding = model.encode_image(img)

# text encoding
text_embedding = model.encode_text(text)

# joint multimodal encoding
img_text_embedding = model.encode_multimodal(img=img, text_data=text)
```

To calculate the compatibility of the image and the text:

```python
# vectors normalization
image_embedding /= image_embedding.square().sum(dim=1, keepdim=True).pow(0.5)
text_embedding /= text_embedding.square().sum(dim=1, keepdim=True).pow(0.5)

# unimodal similarity (range [-1, 1])
scores = (image_embedding * text_embedding).sum(dim=1)

# multimodal similarity (range [0, 1])
scores = model.get_matching_scores(img_text_embedding)
```

## Models

The Multilingual model supports 11 language, trained on a balanced dataset, containing the following languages.

|      |      |      |      |
| :--- | :--- | :--- | :--- |
| en   | de   | es   | fr   |
| it   | jp   | ko   | pl   |
| ru   | tr   | zh   |      |

### Architecture

| Model        | Language Tower | Image Tower |  Shared  |  URL |
| :----------- | :------------: | :---------: | :------: | ---: |
| Monolingual  | BERT, 2 layers |  ViT-B/16   | 2 layers |      |
| Multilingual | BERT, 8 layers |  ViT-B/16   | 4 layers |      |

### Performance

