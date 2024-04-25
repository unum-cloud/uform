from functools import partial
from time import perf_counter
from dataclasses import dataclass
from typing import List
import argparse

import requests
import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    InstructBlipForConditionalGeneration,
    InstructBlipProcessor,
    LlavaForConditionalGeneration,
    AutoModel,
    AutoProcessor,
)

from uform.torch_decoders import VLMForCausalLM, VLMProcessor

dtype = torch.bfloat16
low_cpu_mem_usage = False
device = "cuda:0"


@dataclass
class BenchmarkResult:
    model_name: str
    device_name: str
    backend_name: str
    duration_image_preprocessing: float
    duration_image_embedding: float
    duration_text_preprocessing: float
    duration_text_embedding: float


def caption(model, processor, prompt: str, image: Image.Image, max_length: int, batch_size: int) -> List[str]:
    # BLIP models require the prompt to be the first argument
    prompt = [prompt] * batch_size
    image = [image] * batch_size
    try:
        inputs = processor(prompt, image, return_tensors="pt")
    except ValueError:
        inputs = processor(image, prompt, return_tensors="pt")

    # Downcast and move to device
    for possible_key in ["images", "pixel_values"]:
        if possible_key not in inputs:
            continue
        inputs[possible_key] = inputs[possible_key].to(dtype)  # Downcast floats
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to the right device

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            # use_cache=True,
            max_new_tokens=max_length,
            eos_token_id=32001,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    decoded_texts = processor.batch_decode(
        output[:, prompt_len:],
        skip_special_tokens=True,
    )
    return decoded_texts


def duration(callable):
    """Profile the duration of a callable and return the duration and the result."""
    start = perf_counter()
    result = callable()
    stop = perf_counter()
    return stop - start, result


def bench_captions(
    model,
    processor,
    prompt: str,
    images: List[Image.Image],
    max_length: int = 256,
    batch_size: int = 10,
) -> List[str]:
    total_duration = 0
    total_length = 0
    model = torch.compile(model)

    def caption_image(image):
        return caption(
            model=model,
            processor=processor,
            prompt=prompt,
            image=image,
            max_length=max_length,
            batch_size=batch_size,
        )

    for image in images:
        seconds, captions = duration(partial(caption_image, image=image))
        total_duration += seconds
        total_length += len(captions.strip()) if isinstance(captions, str) else sum(len(t.strip()) for t in captions)

    del model
    del processor
    print(f"Throughput: {total_length/total_duration:.2f} tokens/s")


def main(batch_size: int = 10, max_length: int = 256):

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

    print("UForm-Gen2")
    bench_captions(
        model=AutoModel.from_pretrained(
            "unum-cloud/uform-gen2-dpo",
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            ignore_mismatched_sizes=True,
        ).to(device),
        processor=AutoProcessor.from_pretrained(
            "unum-cloud/uform-gen2-dpo",
            trust_remote_code=True,
        ),
        prompt="Describe the picture in great detail",
        images=images,
        batch_size=batch_size,
        max_length=max_length,
    )

    print("UForm-Gen")
    bench_captions(
        model=VLMForCausalLM.from_pretrained(
            "unum-cloud/uform-gen",
            torch_dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
            ignore_mismatched_sizes=True,
        ).to(device),
        processor=VLMProcessor.from_pretrained(
            "unum-cloud/uform-gen",
        ),
        prompt="[cap] Summarize the visual content of the image.",
        images=images,
        batch_size=batch_size,
        max_length=max_length,
    )

    print("LLaVA")
    bench_captions(
        model=LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            torch_dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        ).to(device),
        processor=AutoProcessor.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
        ),
        prompt="USER: <image>\nWhat are these?\nASSISTANT:",
        images=images,
        batch_size=batch_size,
        max_length=max_length,
    )

    print("InstructBLIP")
    bench_captions(
        model=InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-vicuna-7b",
            torch_dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        ).to(device),
        processor=InstructBlipProcessor.from_pretrained(
            "Salesforce/instructblip-vicuna-7b",
        ),
        prompt="Summarize the visual content of the image.",
        images=images,
        batch_size=batch_size,
        max_length=max_length,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for the benchmark. Batch size 1 measures latency. Large batch sizes may not fit on every GPU.",
    )
    parser.add_argument(
        "--max-length",
        type=str,
        default=256,
        help="Maximum length of the generated text in tokens.",
    )
    args = parser.parse_args()

    main(batch_size=args.batch_size, max_length=args.max_length)
