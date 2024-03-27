from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import (CenterCrop, Compose, InterpolationMode,
                                    Normalize, RandomResizedCrop, Resize,
                                    ToTensor)
from transformers import AutoConfig, AutoTokenizer
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.auto.modeling_auto import (AutoModel,
                                                    AutoModelForCausalLM)
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import BatchEncoding

from uform.models import VisualEncoder

IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)


def convert_to_rgb(image):
    return image.convert("RGB")


class LayerScale(nn.Module):
    def __init__(self, dim, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.weight = nn.Parameter(init_values * torch.ones(dim))
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(self.weight) if self.inplace else x * self.weight


class ImageFeaturesPooler(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_attn_heads,
        intermediate_size,
        num_latents,
        initializer_range,
    ):
        super().__init__()
        self.projection = nn.Linear(input_size, hidden_size)

        self.pooler = nn.TransformerDecoderLayer(
            hidden_size,
            num_attn_heads,
            intermediate_size,
            activation=nn.functional.silu,
            batch_first=True,
            norm_first=True,
        )
        self.image_latents = nn.Parameter(
            torch.randn(1, num_latents, hidden_size) * initializer_range**0.5,
        )

    def forward(self, features):
        features = self.projection(features)
        return self.pooler(
            self.image_latents.expand(features.shape[0], -1, -1),
            features,
        )


class VLMConfig(PretrainedConfig):
    model_type = "vlm"

    def __init__(
        self,
        text_decoder_name_or_path: str = "",
        tokenizer_name_or_path: str = "",
        image_size: int = 224,
        image_encoder_hidden_size: int = 768,
        image_encoder_patch_size: int = 16,
        image_encoder_num_layers: int = 12,
        image_encoder_num_heads: int = 12,
        image_encoder_embedding_dim: int = 256,
        image_encoder_pooling: str = "cls",
        image_pooler_num_attn_heads: int = 16,
        image_pooler_intermediate_size: int = 5504,
        image_pooler_num_latents: int = 196,
        image_token_id: int = 32002,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        center_crop: bool = True,
        **kwargs,
    ):
        self.text_decoder_name_or_path = text_decoder_name_or_path
        self.tokenizer_name_or_path = tokenizer_name_or_path

        self.image_size = image_size
        self.image_encoder_hidden_size = image_encoder_hidden_size
        self.image_encoder_patch_size = image_encoder_patch_size
        self.image_encoder_num_layers = image_encoder_num_layers
        self.image_encoder_num_heads = image_encoder_num_heads
        self.image_encoder_embedding_dim = image_encoder_embedding_dim
        self.image_encoder_pooling = image_encoder_pooling

        self.image_pooler_num_attn_heads = image_pooler_num_attn_heads
        self.image_pooler_intermediate_size = image_pooler_intermediate_size
        self.image_pooler_num_latents = image_pooler_num_latents

        self.image_token_id = image_token_id

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
        self.text_config.vocab_size += 3
        self.text_decoder = AutoModelForCausalLM.from_config(self.text_config)

        self.image_encoder = VisualEncoder(
            self.config.image_encoder_hidden_size,
            self.config.image_encoder_patch_size,
            self.config.image_size,
            self.config.image_encoder_num_layers,
            self.config.image_encoder_num_heads,
            self.config.image_encoder_embedding_dim,
            self.config.image_encoder_pooling,
        )

        # replace models' layerscales because `transformers` automatically renames keys in state_dict
        for i in range(len(self.image_encoder.blocks)):
            self.image_encoder.blocks[i].ls1 = LayerScale(
                self.image_encoder.blocks[i].ls1.dim,
            )
            self.image_encoder.blocks[i].ls2 = LayerScale(
                self.image_encoder.blocks[i].ls2.dim,
            )

        self.image_pooler = ImageFeaturesPooler(
            self.config.image_encoder_hidden_size,
            self.text_config.hidden_size,
            self.config.image_pooler_num_attn_heads,
            self.config.image_pooler_intermediate_size,
            self.config.image_pooler_num_latents,
            self.config.initializer_range,
        )

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
                ),
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
                "You cannot specify both input_ids and inputs_embeds at the same time",
            )
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_is or inputs_embeds")

        if inputs_embeds is None and past_key_values is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

            if images is not None:
                image_embeds = self.get_images_embeddings(images)
                inputs_embeds = self.gather_continuous_embeddings(
                    input_ids,
                    inputs_embeds,
                    image_embeds,
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
            labels=labels,
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
            },
        )
        return model_inputs

    @classmethod
    def from_config(cls, config, **kwargs):
        return cls._from_config(config, **kwargs)


class VLMProcessor(ProcessorMixin):
    def __init__(self, config, **kwargs):
        self.feature_extractor = None
        self.config = config

        if config.center_crop:
            self.image_processor = Compose(
                [
                    Resize(256, interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(config.image_size),
                    convert_to_rgb,
                    ToTensor(),
                    Normalize(
                        mean=IMAGENET_MEAN,
                        std=IMAGENET_STD,
                    ),
                ],
            )
        else:
            self.image_processor = Compose(
                [
                    RandomResizedCrop(
                        config.image_size,
                        scale=(0.8, 1),
                        interpolation=InterpolationMode.BICUBIC,
                    ),
                    convert_to_rgb,
                    ToTensor(),
                    Normalize(
                        mean=IMAGENET_MEAN,
                        std=IMAGENET_STD,
                    ),
                ],
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_name_or_path,
            additional_special_tokens=["<|im_end|>"],
        )
        self.num_image_latents = config.image_pooler_num_latents

    def __call__(self, texts=None, images=None, return_tensors="pt", **kwargs):
        if texts is not None:
            if isinstance(texts, str):
                texts = [texts]

            tokenized_texts = []
            for text in texts:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f" <image> {text}"},
                ]
                tokenized_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors=return_tensors,
                )

                tokenized_texts.append(tokenized_prompt)

            max_len = max(len(t[0]) for t in tokenized_texts)
            input_ids = torch.full(
                (len(tokenized_texts), max_len),
                fill_value=self.tokenizer.pad_token_id,
                dtype=torch.int64,
            )
            attention_mask = torch.full(
                (len(tokenized_texts), max_len),
                fill_value=0,
                dtype=torch.int64,
            )

            for i, tokens in enumerate(tokenized_texts):
                input_ids[i, -len(tokens[0]) :] = tokens[0]
                attention_mask[i, -len(tokens[0]) :] = 1

            attention_mask = F.pad(
                attention_mask,
                pad=(0, self.num_image_latents - 1),
                value=1,
            )

            encoding = BatchEncoding(
                data={"input_ids": input_ids, "attention_mask": attention_mask},
            )

        if images is not None:
            if isinstance(images, (list, tuple)):
                image_features = torch.empty(
                    (len(images), 3, self.config.image_size, self.config.image_size),
                    dtype=torch.float32,
                )

                for i, image in enumerate(images):
                    image_features[i] = self.image_processor(image)
            else:
                image_features = self.image_processor(images).unsqueeze(0)

        if texts is not None and images is not None:
            encoding["images"] = image_features
            return encoding

        if texts is not None:
            return encoding

        return BatchEncoding(
            data={
                "images": image_features,
            },
            tensor_type=return_tensors,
        )

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
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
