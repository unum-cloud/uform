from functools import partial
from time import perf_counter
from typing import List

import requests
import torch
from PIL import Image
from transformers import (AutoProcessor, InstructBlipForConditionalGeneration,
                          InstructBlipProcessor, LlavaForConditionalGeneration)

from uform import get_model
from uform.gen_model import VLMForCausalLM, VLMProcessor

dtype = torch.bfloat16
low_cpu_mem_usage = False
device = "cuda:0"


def caption(model, processor, prompt: str, image: Image.Image) -> str:
    inputs = processor(prompt, image, return_tensors="pt")
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
            max_new_tokens=128,
            eos_token_id=32001,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    prompt_len = inputs["input_ids"].shape[1]
    decoded_text = processor.batch_decode(
        output[:, prompt_len:],
        skip_special_tokens=True,
    )[0].strip()
    return decoded_text


def duration(callable):
    start = perf_counter()
    result = callable()
    stop = perf_counter()
    return stop - start, result


def bench_captions(
    model,
    processor,
    prompt: str,
    images: List[Image.Image],
) -> List[str]:
    total_duration = 0
    total_length = 0
    model = torch.compile(model)

    def caption_image(image, model=model, processor=processor, prompt=prompt):
        return caption(model=model, processor=processor, prompt=prompt, image=image)

    for image in images:
        seconds, text = duration(partial(caption_image, image=image))
        total_duration += seconds
        total_length += len(text)

    del model
    del processor
    print(f"Throughput: {total_length/total_duration:.2f} tokens/s")


def bench_image_embeddings(model, images):
    total_duration = 0
    total_embeddings = 0
    images *= 10
    while total_duration < 10:
        seconds, embeddings = duration(
            lambda: model.encode_image(model.preprocess_image(images))
        )
        total_duration += seconds
        total_embeddings += len(embeddings)

    print(f"Throughput: {total_embeddings/total_duration:.2f} images/s")


def bench_text_embeddings(model, texts):
    total_duration = 0
    total_embeddings = 0
    texts *= 10
    while total_duration < 10:
        seconds, embeddings = duration(
            lambda: model.encode_text(model.preprocess_text(texts))
        )
        total_duration += seconds
        total_embeddings += len(embeddings)

    print(f"Throughput: {total_embeddings/total_duration:.2f} queries/s")


if __name__ == "__main__":
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

    print("UForm-Gen")
    bench_captions(
        model=VLMForCausalLM.from_pretrained(
            "unum-cloud/uform-gen",
            torch_dtype=dtype,
            low_cpu_mem_usage=low_cpu_mem_usage,
        ).to(device),
        processor=VLMProcessor.from_pretrained(
            "unum-cloud/uform-gen",
        ),
        prompt="[cap] Summarize the visual content of the image.",
        images=images,
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
    )

    print("UForm-English")
    bench_image_embeddings(get_model("unum-cloud/uform-vl-english"), images)
    bench_text_embeddings(get_model("unum-cloud/uform-vl-english"), captions)

    print("UForm-Multilingual")
    bench_image_embeddings(get_model("unum-cloud/uform-vl-multilingual-v2"), images)
    bench_text_embeddings(get_model("unum-cloud/uform-vl-multilingual-v2"), captions)
