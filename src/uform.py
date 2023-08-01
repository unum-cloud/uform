from json import load
from typing import Optional, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from transformers import AutoTokenizer
from transformers import BertConfig, BertModel, RobertaConfig, RobertaModel, XLMRobertaConfig, XLMRobertaModel
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.roberta.modeling_roberta import RobertaAttention
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaAttention
from huggingface_hub import snapshot_download, hf_hub_download


class VisualEncoder(nn.Module):
    def __init__(
        self,
        backbone,
        backbone_type: str,
        dim: int,
        output_dim: int,
        pooling='cls',
    ):

        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=False,
            num_classes=0,
        )
        self.backbone_type = backbone_type
        self.pooling = pooling

        if self.pooling == 'attention':
            self.attention_pooling = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=1,
                batch_first=True,
                dropout=0.1,
            )
            self.queries = nn.Parameter(torch.randn(1, 197, dim))

        self.proj = nn.Linear(dim, output_dim, bias=False)
        self.dim = dim

        if hasattr(self.encoder, 'fc_norm'):
            self.encoder.fc_norm = nn.Identity()

        if hasattr(self.encoder, 'head'):
            self.encoder.head = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        features = self.forward_features(x)
        return features, self.get_embedding(features)

    def forward_features(self, x: Tensor) -> Tensor:
        if self.backbone_type == 'vit':
            features = self.forward_features_vit(x)
        else:
            features = self.forward_features_conv(x)

        if self.pooling == 'attention':
            return self.attention_pooling(
                self.queries.expand(x.shape[0], -1, -1),
                features,
                features,
            )[0]

        return features

    def forward_features_conv(self, x: Tensor) -> Tensor:
        return self.encoder.forward_features(x).flatten(2).permute(0, 2, 1)

    def forward_features_vit(self, x: Tensor) -> Tensor:
        x = self.encoder.patch_embed(x)
        x = self.encoder._pos_embed(x)
        x = self.encoder.norm_pre(x)

        for block in self.encoder.blocks:
            x = block(x)

        x = self.encoder.norm(x)

        return x

    def get_embedding(self, x: Tensor, project: bool = True) -> Tensor:
        if isinstance(x, list):
            x = x[-1]

        if self.pooling == 'cls' or self.pooling == 'attention':
            x = x[:, 0]
        elif self.pooling == 'mean':
            x = x.mean(dim=1)

        if project:
            return self.proj(x)

        return x


class TextEncoder(nn.Module):
    def __init__(
        self,
        backbone,
        backbone_type: str,
        unimodal_n_layers: int,
        context_dim: int,
        dim: int,
        output_dim: int,
        pooling: str = 'cls',
        head_one_neuron: bool = False,
    ):

        super().__init__()
        self.backbone = TextEncoderBackbone(
            backbone, backbone_type, unimodal_n_layers)
        if context_dim != dim:
            self.context_proj = nn.Linear(context_dim, dim, bias=False)
        else:
            self.context_proj = nn.Identity()

        self.pooling = pooling
        self.proj = nn.Linear(dim, output_dim, bias=False)
        self.clf_head = nn.Linear(dim, 1 if head_one_neuron else 2)
        self.head_one_neuron = head_one_neuron
        self.dim = dim

    def forward(self, x: Tensor, attention_mask: Tensor, causal: bool = False) -> Tensor:
        features = self.forward_unimodal(x, attention_mask, causal)
        return features, self.get_embedding(features, attention_mask)

    def forward_unimodal(self, x: Tensor, attention_mask: Tensor, causal: bool = False) -> Tensor:
        prep_attention_mask = self.prepare_attention_mask(
            attention_mask, causal)
        x = self.backbone.embeddings(x)

        for layer in self.backbone.unimodal_encoder:
            x = layer(x, prep_attention_mask)[0]

        return x

    def forward_multimodal(
        self,
        x: Tensor,
        attention_mask: Tensor,
        context: Tensor,
        causal: bool = False,
    ) -> Tensor:
        prep_attention_mask = self.prepare_attention_mask(
            attention_mask,
            causal,
        )
        context = self.context_proj(context)
        for layer in self.backbone.multimodal_encoder:
            x, _, _ = layer(x, prep_attention_mask, context)

        return self.get_embedding(x, attention_mask, project=False)

    def get_matching_scores(self, x: Tensor, attention_mask: Tensor, context) -> Tensor:
        embeddings = self.forward_multimodal(
            x, attention_mask, context, False)
        return self._logit_and_norm(embeddings)

    def _logit_and_norm(self, embeddings: Tensor) -> Tensor:
        logits = self.clf_head(embeddings)
        if self.head_one_neuron:
            return torch.sigmoid(logits)[:, 0]

        return F.softmax(logits, dim=1)[:, 1]

    def get_embedding(self, x: Tensor, attention_mask: Tensor, project: bool = True) -> Tensor:
        if self.pooling == 'mean':
            mask_expanded = attention_mask.unsqueeze(2)
            vec_sum = (x * mask_expanded).sum(dim=1)
            x = vec_sum / mask_expanded.sum(dim=1)

        elif self.pooling == 'cls':
            x = x[:, 0]

        if project:
            return self.proj(x)

        return x

    def prepare_attention_mask(self, mask: Tensor, causal: bool = False) -> Tensor:
        if causal:
            causal_mask = torch.ones(
                mask.size(1), mask.size(1), device=mask.device).tril()
            # bs x seq_len x seq_len
            mask = mask[:, None, :] * causal_mask[None, :, :]
            mask = (1 - mask) * -10e9
            return mask[:, None]

        mask = (1 - mask) * -10e9
        return mask[:, None, None, :]


class TextEncoderBackbone(nn.Module):
    type2classes = {
        'bert': (BertConfig, BertModel, BertAttention),
        'roberta': (RobertaConfig, RobertaModel, RobertaAttention),
        'xlm_roberta': (XLMRobertaConfig, XLMRobertaModel, XLMRobertaAttention)
    }

    def __init__(self, pretrained: str, backbone_type: str, unimodal_n_layers: int):
        super().__init__()
        self.unimodal_n_layers = unimodal_n_layers

        config_file = hf_hub_download(
            repo_id=pretrained,
            filename='config.json',
        )
        config_cls, model_cls, attention_layer_cls = self.type2classes[backbone_type]
        config = config_cls.from_json_file(config_file)
        model = model_cls(config)

        self.construct_model(model, attention_layer_cls, config)

    def construct_model(self, backbone, attention_layer_cls: type, config: dict):

        self.unimodal_encoder = backbone.encoder.layer[:self.unimodal_n_layers]
        self.embeddings = backbone.embeddings
        self.multimodal_encoder = []

        for layer in backbone.encoder.layer[self.unimodal_n_layers:]:
            self.multimodal_encoder.append(
                FusedTransformerLayer(
                    config,
                    attention_layer_cls,
                    layer,
                )
            )

        self.multimodal_encoder = nn.ModuleList(self.multimodal_encoder)


class FusedTransformerLayer(nn.Module):

    def __init__(self, config, attention_layer_cls, base_layer):
        super().__init__()

        self.self_attention = base_layer.attention
        self.intermediate = base_layer.intermediate
        self.output = base_layer.output
        self.cross_attention = attention_layer_cls(config)

    def forward(self, x: Tensor, attention_mask: Tensor, context: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        attention_output, self_attention_probs = self.self_attention(
            x,
            attention_mask,
            output_attentions=True,
        )
        attention_output, cross_attention_probs = self.cross_attention(
            attention_output,
            encoder_hidden_states=context,
            output_attentions=True,
        )  # [0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, self_attention_probs, cross_attention_probs


class VLM(nn.Module):
    """
    Vision-Language Model for multi-modal embeddings.

    ...

    Attributes
    ----------
    text_encoder : TextEncoder
        Textual Encoder. Generally a BERT-like transformer.
    image_encoder : VisualEncoder
        Visual Encoder. Generally a ViT or a ConvNet.
    dim : int
        Dimensionality of the multi-modal embedding.
        It is generally larger than `text_encoder.dim` and `image_encoder.dim`.

    Methods
    -------
    preprocess_text(text)
        Transforms one or more strings into Torch Tensors of tokens.
    encode_text(text_data)
        Passes the pre-processed texts through `text_encoder` to produce embeddings.
    preprocess_image(image)
        Transforms one or more Pillow images into Torch Tensors.
    encode_image(image_data)
        Passes the pre-processed images through `image_encoder` to produce embeddings.
    encode_multimodal(image=image, text=text)
        Passes pre-processed texts and images through both towers, mixing the signal.
    get_matching_scores(multimodal_embedding)
        Computes the probability that there is a match between image and text based on their multimodal embedding
    """

    def __init__(self, config: dict):
        super().__init__()
        self.text_encoder = TextEncoder(**config['text_encoder'])
        self.image_encoder = VisualEncoder(**config['img_encoder'])
        self._tokenizer = AutoTokenizer.from_pretrained(
            config['text_encoder']['backbone'])

        self._embedding_dim = config['text_encoder']['output_dim']

    def encode_image(self, x: Tensor, return_features: bool = False) -> Tensor:
        features, embs = self.image_encoder(x)

        if return_features:
            return features, embs

        return embs

    def encode_text(self, x: dict, return_features: bool = False) -> Tensor:
        features, embs = self.text_encoder(x['input_ids'], x['attention_mask'])

        if return_features:
            return features, embs

        return embs

    def encode_multimodal(
        self,
        *,
        image: Optional[Tensor] = None,
        text: Optional[dict] = None,
        image_features: Optional[Tensor] = None,
        text_features: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:

        assert image is not None or image_features is not None, 'Either `image` or `image_features` should be non None'
        assert text is not None or text_features is not None, 'Either `text_data` or `text_features` should be non None'

        if text_features is not None:
            assert attention_mask is not None, 'if `text_features` is not None, then you should pass `attention_mask`'

        if image_features is None:
            image_features = self.image_encoder.forward_features(image)

        if text_features is None:
            text_features = self.text_encoder.forward_unimodal(
                text['input_ids'],
                text['attention_mask']
            )

        return self.text_encoder.forward_multimodal(
            text_features,
            attention_mask if attention_mask is not None else text['attention_mask'],
            image_features
        )

    def get_matching_scores(self, x: Tensor):
        return self.text_encoder._logit_and_norm(x)

    def preprocess_text(self, x) -> dict:
        x = self._tokenizer(
            x,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            pad_to_max_length=True,
            max_length=77,
        )
        if 'token_type_ids' in x:
            del x['token_type_ids']

        return x

    def preprocess_image(self, x) -> Tensor:
        preprocessor = Compose([
            Resize(224, interpolation=InterpolationMode.BICUBIC),
            lambda x: x.convert('RGB'),
            CenterCrop(224),
            ToTensor(),
            Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])
        if isinstance(x, list):
            images = []
            for image in x:
                images.append(preprocessor(image))

            batch_images = torch.stack(images, dim=0)
            return batch_images
        else:
            return preprocessor(x).unsqueeze(0)

    @property
    def text_features_dim(self) -> int:
        return self.text_encoder.dim

    @property
    def image_features_dim(self) -> int:
        return self.image_encoder.dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim
    
    @property
    def multimodal_embedding_dim(self) -> int:
        return self.text_encoder.dim


class TritonClient(VLM):
    """
    Nvidia Triton client to connect to the remote VLM inference server.
    """

    def __init__(self, url: str = 'localhost:7001'):
        import tritonclient.http as httpclient
        self._client = httpclient
        self._triton_client = self._client.InferenceServerClient(
            url=url
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            'google/bert_uncased_L-4_H-768_A-12'
        )

    def encode_image(
        self,
        imgs,
    ):
        """
        Returns the image embedding.
            Parameters:
                imgs (numpy.ndarray): Pre-processed image

            Returns:
                output_data (numpy.ndarray): Image embedding
        """
        # images prep
        inputs = []
        outputs = []
        imgs = imgs.cpu().detach().numpy()
        inputs.append(
            self._client.InferInput('inputs', imgs.shape, 'FP32')
        )
        inputs[0].set_data_from_numpy(imgs)
        outputs.append(self._client.InferRequestedOutput('output'))

        # Querying the server
        results = self._triton_client.infer(
            model_name='vit',
            inputs=inputs,
            outputs=outputs
        )
        output_data = torch.from_numpy(results.as_numpy('output'))
        return output_data

    def encode_text(
        self,
        text,
    ):
        """
        Returns the image embedding.
            Parameters:
                text (dict): Tokenized Text

            Returns:
                output_vec (numpy.ndarray): Text embedding
        """
        # texts prep
        inputs = []
        input_ids, attention_mask = text['input_ids'], text['attention_mask']
        input_ids = input_ids.type(dtype=torch.int32).cpu().detach().numpy()
        attention_mask = attention_mask.type(
            dtype=torch.int32).cpu().detach().numpy()
        inputs.append(self._client.InferInput(
            'attention_mask', attention_mask.shape, 'INT32'))
        inputs.append(self._client.InferInput(
            'input_ids', input_ids.shape, 'INT32'))
        inputs[0].set_data_from_numpy(attention_mask)
        inputs[1].set_data_from_numpy(input_ids)
        test_output = self._client.InferRequestedOutput('output')

        # Querying the server
        results = self._triton_client.infer(
            model_name='albef',
            inputs=inputs,
            outputs=[test_output]
        )
        output_vec = torch.from_numpy(results.as_numpy('output'))
        return output_vec

    def encode_multimodal(self, *args, **kwargs):
        raise NotImplementedError('Multimodal encodings coming soon!')


def get_model(model_name: str, token: Optional[Union[bool, str]] = None) -> VLM:
    model_path = snapshot_download(
        repo_id=model_name,
        token=token,
    )
    config_path = f'{model_path}/config.json'
    state = torch.load(f'{model_path}/weight.pt')

    with open(config_path, 'r') as f:
        model = VLM(load(f))

    model.image_encoder.load_state_dict(state['img_encoder'])
    model.text_encoder.load_state_dict(state['text_encoder'], strict=False)

    return model.eval()


def get_client(url: str) -> TritonClient:
    return TritonClient(url)
