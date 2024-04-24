# UForm Model Benchmarks

## Accuracy

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

### Embedding Models

UForm comes pre-packaged with speed benchmarks for the models.
    
```bash
$ python python/scripts/bench_encoders.py --help
usage: bench_encoders.py [-h] [--filter-out FILTER_OUT] [--batch-size BATCH_SIZE]

options:
  -h, --help            show this help message and exit
  --filter-out FILTER_OUT
                        Filter out models, backends, or devices with a Regular Expression.
  --batch-size BATCH_SIZE
                        Batch size for the benchmark. Batch size 1 measures latency. Large batch sizes may not fit on every GPU.
```

Running that script for a fairly small batch size of 50 on an Nvidia H100 GPU and 

| Model Name                                     | Device | Backend | Images Preprocessed/s | Images Encoded/s | Texts Preprocessed/s | Texts Encoded/s |
| :--------------------------------------------- | :----- | :------ | --------------------: | :--------------- | :------------------- | :-------------- |
| unum-cloud/uform3-image-text-english-base      | cpu    | torch   |                 23.03 | 76.57            | 15,978.03            | 562.28          |
| unum-cloud/uform3-image-text-english-base      | cpu    | onnx    |                 23.11 | 77.75            | 13,880.27            | 1,067.40        |
| unum-cloud/uform3-image-text-english-base      | cuda   | torch   |                 22.87 | 1,060.40         | 12,348.94            | 13,242.83       |
| unum-cloud/uform3-image-text-english-large     | cpu    | torch   |                 22.41 | 10.84            | 13,350.45            | 145.12          |
| unum-cloud/uform3-image-text-english-large     | cpu    | onnx    |                 23.13 | 19.60            | 18,031.85            | 960.09          |
| unum-cloud/uform3-image-text-english-large     | cuda   | torch   |                 22.78 | 244.86           | 13,226.40            | 10,204.04       |
| unum-cloud/uform3-image-text-english-small     | cpu    | torch   |                 20.08 | 71.68            | 12,147.05            | 249.63          |
| unum-cloud/uform3-image-text-english-small     | cpu    | onnx    |                 22.84 | 195.27           | 13,636.99            | 1,385.25        |
| unum-cloud/uform3-image-text-english-small     | cuda   | torch   |                 22.63 | 2,662.16         | 14,731.18            | 14,694.87       |
| unum-cloud/uform3-image-text-multilingual-base | cpu    | torch   |                 22.98 | 64.28            | 10,129.27            | 209.76          |
| unum-cloud/uform3-image-text-multilingual-base | cpu    | onnx    |                 23.06 | 66.81            | 8,963.13             | 1,104.32        |
| unum-cloud/uform3-image-text-multilingual-base | cuda   | torch   |                 22.88 | 1,051.95         | 15,639.72            | 12,416.12       |

If you are interested in performance numbers on consumer grade hardware, compared to third-party models, here are some rough estimates.
On Nvidia RTX 3090:

| Model                                            | Multilingual |                  Speed |    Speedup |
| :----------------------------------------------- | -----------: | ---------------------: | ---------: |
| `bert-base-uncased`                              |           No | 1'612 sequences/second |            |
| `distilbert-base-uncased`                        |           No | 3'174 sequences/second |     x 1.96 |
| `sentence-transformers/all-MiniLM-L12-v2`        |      __Yes__ | 3'604 sequences/second |     x 2.24 |
| `unum-cloud/uform3-image-text-multilingual-base` |      __Yes__ | 6'809 sequences/second | __x 4.22__ |

Given the small size of the model it also work well on mobile devices.
On Apple M2 Arm chips the energy efficiency of inference can exceed that of the RTX 3090 GPU and other Ampere-generation cards.

| Device                 |               Speed | Device TDP |        Efficiency |
| :--------------------- | ------------------: | ---------: | ----------------: |
| Nvidia RTX 3090        | ~ 140 tokens/second |     < 350W | 0.40 tokens/joule |
| Apple M2 Pro unplugged |  ~ 19 tokens/second |      < 20W | 0.95 tokens/joule |
| Apple M2 Max unplugged |  ~ 38 tokens/second |      < 36W | 1.06 tokens/joule |
| Apple M2 Max plugged   |  ~ 56 tokens/second |      < 89W | 0.63 tokens/joule |

### Generative Models

```bash
$ python python/scripts/bench_decoders.py --help
usage: bench_decoders.py [-h] [--filter-out FILTER_OUT] [--batch-size BATCH_SIZE]

options:
  -h, --help            show this help message and exit
  --filter-out FILTER_OUT
                        Filter out models, backends, or devices with a Regular Expression.
  --batch-size BATCH_SIZE
                        Batch size for the benchmark. Batch size 1 measures latency. Large batch sizes may not fit on every GPU.
```

On Nvidia RTX 3090, the following performance is expected on text token generation using `float16`, equivalent PyTorch settings, and greedy decoding.

| Model                               | Size |               Speed |   Speedup |
| :---------------------------------- | ---: | ------------------: | --------: |
| `llava-hf/llava-1.5-7b-hf`          |   7B |  ~ 40 tokens/second |           |
| `Salesforce/instructblip-vicuna-7b` |   7B |  ~ 40 tokens/second |           |
| `unum-cloud/uform-gen`              | 1.5B | ~ 140 tokens/second | __x 3.5__ |

