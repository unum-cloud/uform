from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch import nn

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import AutoModelForCausalLM, AutoModel
from transformers import AutoConfig, AutoTokenizer
from transformers.utils import logging
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding
from transformers.configuration_utils import PretrainedConfig
from torchvision.transforms import *

from uform import VisualEncoder
from PIL import Image

logger = logging.get_logger(__name__)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.weight = nn.Parameter(init_values * torch.ones(dim))
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(self.weight) if self.inplace else x * self.weight


class ImageFeaturesPooler(nn.Module):
    def __init__(self, config, text_config):
        super().__init__()
        self.pooler = nn.TransformerDecoderLayer(
            text_config.hidden_size,
            config.image_pooler_num_attn_heads,
            config.image_pooler_intermediate_size,
            activation=nn.functional.silu,
            batch_first=True,
            norm_first=True,
        )
        self.projection = nn.Linear(config.in_sizes, text_config.hidden_size)
        self.image_latents = nn.Parameter(
            torch.randn(1, config.num_image_latents, text_config.hidden_size)
            * config.initializer_range**0.5
        )

    def forward(self, features):
        features = self.projection(features)
        return self.pooler(
            self.image_latents.expand(features.size(0), -1, -1), features
        )


class VLMConfig(PretrainedConfig):
    model_type = "vlm"

    def __init__(
        self,
        text_decoder_name_or_path: str = "",
        in_sizes: List = [768],
        image_encoder_hidden_size: int = 768,
        image_encoder_patch_size: int = 16,
        image_size: int = 224,
        image_encoder_num_layers: int = 12,
        image_encoder_num_heads: int = 12,
        image_encoder_embedding_dim: int = 256,
        image_encoder_pooling: str = "cls",
        image_pooler_num_attn_heads: int = 16,
        image_pooler_intermediate_size: int = 5504,
        image_token_id: int = 32002,
        num_image_latents: int = 196,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        center_crop: bool = True,
        **kwargs,
    ):
        self.text_decoder_name_or_path = text_decoder_name_or_path
        self.in_sizes = in_sizes

        self.image_encoder_hidden_size = image_encoder_hidden_size
        self.image_encoder_patch_size = image_encoder_patch_size
        self.image_size = image_size
        self.image_encoder_num_layers = image_encoder_num_layers
        self.image_encoder_num_heads = image_encoder_num_heads
        self.image_encoder_embedding_dim = image_encoder_embedding_dim
        self.image_encoder_pooling = image_encoder_pooling

        self.image_pooler_num_attn_heads = image_pooler_num_attn_heads
        self.image_pooler_intermediate_size = image_pooler_intermediate_size
        self.image_token_id = image_token_id
        self.num_image_latents = num_image_latents

        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.center_crop = center_crop

        super().__init__(**kwargs)


class VLMPreTrainedModel(PreTrainedModel):
    config_class = VLMConfig
    base_model_prefix = "vlm"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        pass

    def _initialize_weights(self, module):
        pass


class VLMForCausalLM(VLMPreTrainedModel):
    def __init__(self, config: VLMConfig):
        super().__init__(config)

        self.config = config
        self.text_config = AutoConfig.from_pretrained(config.text_decoder_name_or_path)
        self.text_decoder = AutoModelForCausalLM.from_config(self.text_config)

        vocab_size, dim = self.text_decoder.model.embed_tokens.weight.shape
        r = 8
        self.embs_lora_a = nn.Parameter(torch.zeros(vocab_size, r))
        self.embs_lora_b = nn.Parameter(torch.randn(r, dim))

        self.register_buffer(
            "trainable_tokens_ids", torch.arange(32003, 32109)[None, None]
        )

        self.image_encoder = VisualEncoder(
            self.config.image_encoder_hidden_size,
            self.config.image_encoder_patch_size,
            self.config.image_size,
            self.config.image_encoder_num_layers,
            self.config.image_encoder_num_heads,
            self.config.image_encoder_embedding_dim,
            self.config.image_encoder_pooling,
        )

        # replace models' layerscales because transformers automatically rename keys in state_dict
        for i in range(len(self.image_encoder.blocks)):
            self.image_encoder.blocks[i].ls1 = LayerScale(
                self.image_encoder.blocks[i].ls1.dim
            )
            self.image_encoder.blocks[i].ls2 = LayerScale(
                self.image_encoder.blocks[i].ls2.dim
            )

        self.image_pooler = ImageFeaturesPooler(config, self.text_config)

    def get_input_embeddings(self):
        return self.text_decoder.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.text_decoder.set_input_embeddings(value)

    def get_images_embeddings(self, images):
        features = self.image_encoder.forward_features(images)
        return self.image_pooler(features)

    def gather_continuous_embeddings(
        self,
        input_ids: torch.Tensor,
        word_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        start_indices = (input_ids == self.config.image_token_id).nonzero()[:, 1]
        embeddings = []

        for sample_idx, start_idx in enumerate(start_indices.tolist()):
            embeddings.append(
                torch.cat(
                    (
                        word_embeddings[sample_idx, :start_idx],
                        image_embeddings[sample_idx],
                        word_embeddings[sample_idx, start_idx + 1 :],
                    ),
                    dim=0,
                )
            )

        return torch.stack(embeddings, dim=0)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        images: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[dict, Tuple, CausalLMOutputWithPast]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_is or inputs_embeds")

        if inputs_embeds is None and past_key_values is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            trainable_add = F.embedding(input_ids, self.embs_lora_a) @ self.embs_lora_b
            mask = (input_ids[:, :, None] == self.trainable_tokens_ids).int().sum(dim=2)
            mask = (1 - mask).bool().unsqueeze(2)

            inputs_embeds += trainable_add.masked_fill_(mask, 0)

            if images is not None:
                image_embeds = self.get_images_embeddings(images)
                inputs_embeds = self.gather_continuous_embeddings(
                    input_ids, inputs_embeds, image_embeds
                )

        if position_ids is None:
            seq_length = (
                inputs_embeds.shape[1]
                if inputs_embeds is not None
                else input_ids.shape[1]
            )
            past_key_values_length = 0

            if past_key_values is not None:
                past_key_values_length = past_key_values[0][0].shape[2]

            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0)

        outputs = self.text_decoder(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids if past_key_values is not None else None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            return_dict=return_dict,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        images=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        if images is not None:
            model_inputs["images"] = images

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": images if past_key_values is None else None,
            }
        )
        return model_inputs

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls._from_config(config, **kwargs)


class VLMProcessor(ProcessorMixin):
    def __init__(self, config, **kwargs):
        """self.image_processor = AutoImageProcessor.from_pretrained(
            config.image_encoder_name_or_path
        )"""

        if config.center_crop:
            self.image_processor = Compose(
                [
                    # Resize(256, interpolation=InterpolationMode.BICUBIC),
                    # convert_to_rgb,
                    # CenterCrop(224),
                    ToTensor(),
                    Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )
        else:
            self.image_processor = Compose(
                [
                    # RandomResizedCrop(
                    #     224, scale=(0.75, 1), interpolation=InterpolationMode.BICUBIC
                    # ),
                    # convert_to_rgb,
                    ToTensor(),
                    Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711),
                    ),
                ]
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.text_decoder_name_or_path, additional_special_tokens=["<|im_end|>"]
        )
        self.num_image_latents = config.num_image_latents

    def __call__(
        self, text=None, images=None, prompt=None, return_tensors=None, **kwargs
    ):
        if prompt is not None and text is not None and images is not None:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f" <image> {prompt}"},
            ]
            tokenized_prompt = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )

            answer = self.tokenizer(f"\n{text} <|im_end|>\n")["input_ids"][3:]

            labels = torch.full(
                (
                    len(tokenized_prompt) + len(answer) - 1,
                ),  # -1 because <image> and task tokens will be deleted
                fill_value=-100,
                dtype=torch.int64,
            )
            labels[len(tokenized_prompt) - 2 : -1] = torch.LongTensor(answer)

            if isinstance(images, list):
                batch_images = torch.empty(
                    (len(images), 3, 224, 224),
                    dtype=torch.float32,
                )

                for i, image in enumerate(images):
                    image = self.letterbox_resize(image, (224, 224))
                    batch_images[i] = self.image_processor(image)

            else:
                images = self.letterbox_resize(images, (224, 224))
                batch_images = self.image_processor(images).unsqueeze(0)

            encoding = {
                "input_ids": torch.LongTensor(tokenized_prompt + answer),
                "labels": labels,
                # "pixel_values": self.image_processor(
                #     images, return_tensors=return_tensors, **kwargs
                # ).pixel_values,
                "pixel_values": batch_images,
            }
            return encoding

        if text is not None:
            if isinstance(text, str):
                text = [text]

            tokenized_texts = []
            for t in text:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f" <image> {t}"},
                ]
                tokenized_prompt = self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, return_tensors=return_tensors
                )

                tokenized_texts.append(tokenized_prompt)

            max_len = max(len(t[0]) for t in tokenized_texts)
            input_ids = torch.full(
                (len(tokenized_texts), max_len),
                fill_value=self.tokenizer.pad_token_id,
                dtype=torch.int64,
            )
            attention_mask = torch.full(
                (len(tokenized_texts), max_len), fill_value=0, dtype=torch.int64
            )

            for i, tokens in enumerate(tokenized_texts):
                input_ids[i, -len(tokens[0]) :] = tokens[0]
                attention_mask[i, -len(tokens[0]) :] = 1

            attention_mask = F.pad(
                attention_mask, pad=(0, self.num_image_latents - 1), value=1
            )

            encoding = BatchEncoding(
                data={"input_ids": input_ids, "attention_mask": attention_mask}
            )

        if images is not None:
            if isinstance(images, list):
                image_features = torch.empty(
                    (len(images), 3, 224, 224),
                    dtype=torch.float32,
                )

                for i, image in enumerate(images):
                    image = self.letterbox_resize(image, (224, 224))
                    image_features[i] = self.image_processor(image)

            else:
                images = self.letterbox_resize(images, (224, 224))
                image_features = self.image_processor(images).unsqueeze(0)

        if text is not None and images is not None:
            encoding["images"] = image_features
            return encoding

        elif text is not None:
            return encoding

        else:
            return BatchEncoding(
                data={
                    "images": image_features,
                },
                tensor_type=return_tensors,
            )

    def letterbox_resize(self, image, target_size):
        # Calculate the aspect ratio of the original image
        original_width, original_height = image.size
        original_aspect_ratio = original_width / original_height

        # Calculate the aspect ratio of the target size
        target_width, target_height = target_size
        target_aspect_ratio = target_width / target_height

        # Calculate the new size to fit into the target size while maintaining the aspect ratio
        if original_aspect_ratio > target_aspect_ratio:
            new_width = target_width
            new_height = int(target_width / original_aspect_ratio)
        else:
            new_width = int(target_height * original_aspect_ratio)
            new_height = target_height

        # Resize the image while maintaining the aspect ratio
        resized_image = image.resize((new_width, new_height)).convert("RGB")

        # Create a new image with the target size and paste the resized image onto it (letterboxing)
        letterboxed_image = Image.new("RGB", target_size, (122, 116, 104))
        letterboxed_image.paste(resized_image, (0, 0))

        return letterboxed_image

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to CLIPTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        cache_dir=None,
        force_download: bool = False,
        local_files_only: bool = False,
        token=None,
        revision: str = "main",
        **kwargs,
    ):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        return cls(config)


AutoConfig.register("vlm", VLMConfig)
AutoModel.register(VLMConfig, VLMForCausalLM)
