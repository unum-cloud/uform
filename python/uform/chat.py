from argparse import ArgumentParser

import requests
import torch
from PIL import Image
from transformers import TextStreamer, AutoModel, AutoProcessor


def parse_args():
    parser = ArgumentParser(description="Chat with UForm generative model")

    parser.add_argument("--model", type=str, default="unum-cloud/uform-gen-chat", help="Model name or path")
    parser.add_argument("--image", type=str, required=True, help="Path to image or URL")
    parser.add_argument("--device", type=str, required=True, help="Device to run on, like `cpu` or `cuda:0`")
    parser.add_argument("--fp16", action="store_true", help="Use half-precision math for faster inference")

    return parser.parse_args()


def run_chat(opts, model, processor):
    streamer = TextStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    is_first_message = True

    if opts.image.startswith("http"):
        image = Image.open(requests.get(opts.image, stream=True).raw)
    else:
        image = Image.open(opts.image)

    image = (
        processor.feature_extractor(image)  #
        .unsqueeze(0)
        .to(torch.bfloat16 if opts.fp16 else torch.float32)
        .to(opts.device)
    )

    while True:
        if messages[-1]["role"] in ("system", "assistant"):
            message = input("User: ")
            if is_first_message:
                message = f" <image> {message}"
                is_first_message = False
            messages.append({"role": "user", "content": message})

            print()

        else:
            input_ids = processor.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(opts.device)

            attention_mask = torch.ones(
                1,
                input_ids.shape[1] + processor.num_image_latents - 1,
            ).to(opts.device)
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "images": image,
            }

            print("Assistant: ", end="")
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=1024,
                    eos_token_id=151645,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    streamer=streamer,
                )
            print()

            prompt_len = inputs["input_ids"].shape[1]
            message = processor.batch_decode(output[:, prompt_len:-1])[0]

            messages.append({"role": "assistant", "content": message})


def main():
    try:
        opts = parse_args()
        processor = AutoProcessor.from_pretrained(opts.model, trust_remote_code=True)
        model = (
            AutoModel.from_pretrained(
                opts.model,
                torch_dtype=torch.bfloat16 if opts.fp16 else torch.float32,
                ignore_mismatched_sizes=True,
                trust_remote_code=True,
            )
            .eval()
            .to(opts.device)
        )

        run_chat(opts, model, processor)

    except KeyboardInterrupt:
        print("Bye!")
        pass


if __name__ == "__main__":
    main()
