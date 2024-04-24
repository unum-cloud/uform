#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script provides the throughput of UForm multimodal embedding models.

The output of the script will cover:
    - Time to preprocess an image, and throughput in images/s.
    - Time to tokenize the text, and throughput in queries/s.
    - Time to encode the image, and throughput in images/s.
    - Time to encode the text, and throughput in queries/s.
    - Share of time spent on each part of the pipeline.
    
Those numbers are presented for every model, device (cpu or gpu), backend (torch or onnx), 
and precision (float32 or bfloat16), producing a pretty comprehensive benchmark.

Before running the script - install all available packages via `pip install -e ".[torch,onnx,onnx-gpu]"`.
Before printing the numbers, a warm-up is performed to ensure the model is loaded and the cache is filled.
"""

from functools import partial
from time import perf_counter
from dataclasses import dataclass
from typing import List, Tuple, Literal, Callable, Generator
import re
import argparse

import requests
from PIL import Image
import pandas as pd

from uform import get_model, Modality, ExecutionProviderError

# Define global constants for the hardware availability
torch_available = False
try:
    import torch

    torch_available = True
except ImportError:
    pass
onnx_available = False
try:
    import onnx

    onnx_available = True
except ImportError:
    pass
cuda_available = False
try:
    if torch_available:
        cuda_available = torch.cuda.is_available()
    elif onnx_available:
        import onnxruntime

        cuda_available = onnxruntime.get_device() == "GPU"
except ImportError:
    pass


@dataclass
class BenchmarkResult:
    model_name: str
    device_name: Literal["cpu", "cuda"] = "cpu"
    backend_name: Literal["torch", "onnx"] = "torch"
    duration_image_preprocessing: float = 0
    duration_image_embedding: float = 0
    duration_text_preprocessing: float = 0
    duration_text_embedding: float = 0


def duration(callable, synchronize=False):
    """Profile the duration of a callable and return the duration and the result."""
    if synchronize and torch_available and cuda_available:
        torch.cuda.synchronize()  # Wait for CUDA operations to complete
    start = perf_counter()
    result = callable()
    if synchronize and torch_available and cuda_available:
        torch.cuda.synchronize()  # Ensure all CUDA kernels have finished
    stop = perf_counter()
    return stop - start, result


def get_captioned_images() -> List[Tuple[Image.Image, str]]:
    """Get a list of pre-downloaded and decoded images and their captions."""
    image_urls = [
        "https://images.unsplash.com/photo-1697665666330-7acf230fa830?q=80&w=2787&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://images.unsplash.com/photo-1695653422543-7da6d6744364?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDF8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://images.unsplash.com/photo-1703244551371-ecffad9cc3b6?q=80&w=2859&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://plus.unsplash.com/premium_photo-1702910931866-2642eee270b1?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
        "https://plus.unsplash.com/premium_photo-1700583712241-893aded49e69?q=80&w=2942&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
    ]
    images = [Image.open(requests.get(url, stream=True).raw) for url in image_urls]
    captions = [
        "lonely house in a beautiful valley. house is made of white wood and black bricks. its surrounded by a green field",
        "grab last-mile delivery driver on a scooter grabbing a delivery in Jakarta",
        "monochrome picture of new york in the late 2th century on a sunny day, showing a few canonical brick buildings and the citizens bank",
        "asian girl sleeping in a bed. top down view",
        "a few food containers, with past, corn, olives, and sliced red & green peppers, with a man pouring sous on top of it",
    ]
    return list(zip(images, captions))


def yield_benchmarks(batch_size: int) -> Generator[Tuple[BenchmarkResult, Callable], None, None]:
    """Yields callable benchmarks for all supported backends of the given model."""

    # Pull the content and artificially grow the batch size
    images, captions = zip(*get_captioned_images())

    if len(images) < batch_size:
        import math

        multiplier = int(math.ceil(batch_size / len(images)))
        images *= multiplier
        captions *= multiplier
    images = images[:batch_size]
    captions = captions[:batch_size]

    def run(model_name: str, device: str, backend_name: str):
        result = BenchmarkResult(
            model_name=model_name,
            backend_name=backend_name,
            device_name=device,
            duration_image_preprocessing=0,
            duration_image_embedding=0,
            duration_text_preprocessing=0,
            duration_text_embedding=0,
        )

        sync = backend_name == "torch"
        processors, models = get_model(
            model_name,
            device=device,
            modalities=[Modality.IMAGE_ENCODER, Modality.TEXT_ENCODER],
            backend=backend_name,
        )

        model_text = models[Modality.TEXT_ENCODER]
        model_image = models[Modality.IMAGE_ENCODER]
        processor_text = processors[Modality.TEXT_ENCODER]
        processor_image = processors[Modality.IMAGE_ENCODER]

        # Image preprocessing
        total_duration = 0
        total_iterations = 0
        while total_duration < 10 and total_iterations < 100:
            seconds, _ = duration(lambda: processor_image(images))
            total_duration += seconds
            total_iterations += len(images)
        duration_per_iteration = total_duration / total_iterations
        result.duration_image_preprocessing = duration_per_iteration

        # Image embedding
        total_duration = 0
        total_iterations = 0
        while total_duration < 10 and total_iterations < 100:
            images_data = processor_image(images)
            seconds, _ = duration(lambda: model_image.encode(images_data), synchronize=sync)
            total_duration += seconds
            total_iterations += len(images)
        duration_per_iteration = total_duration / total_iterations
        result.duration_image_embedding = duration_per_iteration

        # Text preprocessing
        total_duration = 0
        total_iterations = 0
        while total_duration < 10 and total_iterations < 100:
            seconds, _ = duration(lambda: processor_text(captions))
            total_duration += seconds
            total_iterations += len(captions)
        duration_per_iteration = total_duration / total_iterations
        result.duration_text_preprocessing = duration_per_iteration

        # Text embedding
        total_duration = 0
        total_iterations = 0
        while total_duration < 10 and total_iterations < 100:
            texts_data = processor_text(captions)
            seconds, _ = duration(lambda: model_text.encode(texts_data), synchronize=sync)
            total_duration += seconds
            total_iterations += len(captions)
        duration_per_iteration = total_duration / total_iterations
        result.duration_text_embedding = duration_per_iteration

        return result

    devices = ["cpu"]
    if cuda_available:
        devices.append("cuda")
    backends = []
    if torch_available:
        backends.append("torch")
    if onnx_available:
        backends.append("onnx")

    for device in devices:
        for backend_name in backends:
            for model_name in [
                "unum-cloud/uform3-image-text-english-small",
                "unum-cloud/uform3-image-text-english-base",
                "unum-cloud/uform3-image-text-english-large",
                "unum-cloud/uform3-image-text-multilingual-base",
            ]:
                yield BenchmarkResult(
                    model_name=model_name,
                    device_name=device,
                    backend_name=backend_name,
                ), partial(run, model_name, device, backend_name)


def main(filter_out: str = None, batch_size: int = 10):
    results = []
    filter_pattern = re.compile(filter_out) if filter_out else None
    for specs, func in yield_benchmarks(batch_size=batch_size):
        if filter_pattern and (
            filter_pattern.search(specs.model_name)
            or filter_pattern.search(specs.backend_name)
            or filter_pattern.search(specs.device_name)
        ):
            continue

        try:
            print(f"Running `{specs.model_name}` on `{specs.device_name}` using `{specs.backend_name}` backend")
            result = func()
            results.append(result)
        except ExecutionProviderError as e:
            print(f"- skipping missing backend")
            print(e)

    results = sorted(results, key=lambda x: x.model_name)
    results = [x.__dict__ for x in results]

    df = pd.DataFrame(results)
    df.columns = [
        "Model Name",
        "Device",
        "Backend",
        "Images Preprocessed/s",
        "Images Encoded/s",
        "Texts Preprocessed/s",
        "Texts Encoded/s",
    ]

    def inverse(x):
        return 1 / x if x != 0 else 0

    # Apply number formatting directly in the DataFrame
    formatted_df = df.copy()
    formatted_df["Images Preprocessed/s"] = df["Images Preprocessed/s"].map(inverse).map("{:,.2f}".format)
    formatted_df["Images Encoded/s"] = df["Images Encoded/s"].map(inverse).map("{:,.2f}".format)
    formatted_df["Texts Preprocessed/s"] = df["Texts Preprocessed/s"].map(inverse).map("{:,.2f}".format)
    formatted_df["Texts Encoded/s"] = df["Texts Encoded/s"].map(inverse).map("{:,.2f}".format)

    # Convert formatted DataFrame to Markdown
    print(formatted_df.to_markdown())


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filter-out",
        type=str,
        default=None,
        help="Filter out models, backends, or devices with a Regular Expression.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for the benchmark. Batch size 1 measures latency. Large batch sizes may not fit on every GPU.",
    )
    args = parser.parse_args()

    main(filter_out=args.filter_out, batch_size=args.batch_size)
