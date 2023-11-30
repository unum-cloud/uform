import torch

from transformers import TextStreamer
from .src.gen_model import VLMForCausalLM, VLMProcessor
from PIL import Image

EOS_TOKEN = 32001


if __name__ == "__main__":
    print(
        "1) For setting an image: [img] path/to/the/image",
        "2) For captioning: [cap] describe the image / give a detailed description etc",
        "3) For VQA: [vqa] question",
        "4) For only-text prompts: [txt] prompt",
        sep="\n",
    )
    image = None

    print("\nLoading model")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = VLMForCausalLM.from_pretrained("unum-cloud/uform-gen").eval().to(device)
    processor = VLMProcessor.from_pretrained("unum-cloud/uform-gen")
    streamer = TextStreamer(
        processor.tokenizer, skip_prompt=True, skip_special_tokens=True
    )

    while True:
        print("> ", end="")
        prompt = input()

        if prompt.startswith("[img]"):
            image_path = prompt.split("[img]")[-1].strip()
            image = Image.open(image_path)
            print("Image is set!")
            continue

        is_text_only = prompt.startswith("[txt]")

        input_data = processor(text=prompt, images=image, return_tensors="pt").to(
            device
        )

        with torch.inference_mode():
            response = model.generate(
                input_ids=input_data["input_ids"],
                attention_mask=None if is_text_only else input_data["attention_mask"],
                images=None if is_text_only else input_data["images"],
                use_cache=True,
                do_sample=False,
                max_new_tokens=1024,
                eos_token_id=EOS_TOKEN,
                pad_token_id=processor.tokenizer.pad_token_id,
                streamer=streamer,
            )
