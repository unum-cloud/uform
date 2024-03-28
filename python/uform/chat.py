from argparse import ArgumentParser

import requests
import torch
from PIL import Image
from transformers import TextStreamer

from uform.gen_model import VLMForCausalLM, VLMProcessor

EOS_TOKEN = 32001


def parse_args():
    parser = ArgumentParser(description="Chat with UForm generative model")

    parser.add_argument("--model", type=str, default="unum-cloud/uform-gen-chat")
    parser.add_argument("--image", type=str, help="", required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--fp16", action="store_true")

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
        image = (
            processor.image_processor(
                Image.open(requests.get(opts.image, stream=True).raw),
            )
            .unsqueeze(0)
            .to(torch.bfloat16 if opts.fp16 else torch.float32)
            .to(opts.device)
        )
    else:
        image = (
            processor.image_processor(Image.open(opts.image))
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
            x = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "images": image,
            }

            print("Assistant: ", end="")
            with torch.inference_mode():
                y = model.generate(
                    **x,
                    do_sample=False,
                    use_cache=True,
                    max_new_tokens=1024,
                    eos_token_id=EOS_TOKEN,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    streamer=streamer,
                )
            print()

            message = processor.batch_decode(y[:, x["input_ids"].shape[1] : -1])[0]

            messages.append({"role": "assistant", "content": message})


def main():
    try:
        opts = parse_args()

        model = (
            VLMForCausalLM.from_pretrained(
                opts.model,
                torch_dtype=torch.bfloat16 if opts.fp16 else torch.float32,
            )
            .eval()
            .to(opts.device)
        )
        processor = VLMProcessor.from_pretrained(opts.model)

        run_chat(opts, model, processor)

    except KeyboardInterrupt:
        print("Bye!")
        pass


if __name__ == "__main__":
    main()
