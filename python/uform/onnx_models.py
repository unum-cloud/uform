from os.path import join
from typing import Dict, Optional, Tuple, Union

import onnxruntime as ort
from numpy import ndarray


class ExecutionProviderError(Exception):
    """Exception raised when a requested execution provider is not available."""


def available_providers(device: str) -> Tuple[str, ...]:
    gpu_providers = ("CUDAExecutionProvider", "TensorrtExecutionProvider")
    cpu_providers = ("OpenVINOExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider")
    available = ort.get_available_providers()
    if device == "gpu":
        if all(provider not in available for provider in gpu_providers):
            raise ExecutionProviderError(
                f"GPU providers are not available, consider installing `onnxruntime-gpu` and make sure the CUDA is available on your system. Currently installed: {available}"
            )
        return gpu_providers

    return cpu_providers


class VisualEncoderONNX:
    def __init__(self, model_path: str, device: str):
        """
        :param model_path: Path to onnx model
        :param device: Device name, either cpu or gpu
        """

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=available_providers(device),
        )

    def __call__(self, images: ndarray) -> Tuple[ndarray, ndarray]:
        return self.session.run(None, {"images": images})


class TextEncoderONNX:
    def __init__(self, text_encoder_path: str, reranker_path: str, device: str):
        """
        :param text_encoder_path: Path to onnx of text encoder
        :param reranker_path: Path to onnx of reranker
        :param device: Device name, either cpu or gpu
        """

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.text_encoder_session = ort.InferenceSession(
            text_encoder_path,
            sess_options=session_options,
            providers=available_providers(device),
        )

        self.reranker_session = ort.InferenceSession(
            reranker_path,
            sess_options=session_options,
            providers=available_providers(device),
        )

    def __call__(self, input_ids: ndarray, attention_mask: ndarray) -> Tuple[ndarray, ndarray]:
        return self.text_encoder_session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

    def forward_multimodal(
        self, text_features: ndarray, attention_mask: ndarray, image_features: ndarray
    ) -> Tuple[ndarray, ndarray]:
        return self.reranker_session.run(
            None,
            {
                "text_features": text_features,
                "attention_mask": attention_mask,
                "image_features": image_features,
            },
        )


class VLM_ONNX:
    def __init__(self, checkpoint_path: str, config: Dict, device: str, dtype: str):
        assert device in (
            "cpu",
            "gpu",
        ), f"Invalid `device`: {device}. Must be either `cpu` or `gpu`"
        assert dtype in (
            "fp32",
            "fp16",
        ), f"Invalid `dtype`: {dtype}. Must be either `fp32` or `fp16` (only for gpu)"
        assert (
            device == "cpu" and dtype == "fp32"
        ) or device == "gpu", "Combination `device`=`cpu` & `dtype=fp16` is not supported"

        self.device = device
        self.dtype = dtype

        self._embedding_dim = config["text_encoder"]["embedding_dim"]
        self._text_encoder_dim = config["text_encoder"]["dim"]
        self._image_encoder_dim = config["image_encoder"]["dim"]

        self.text_encoder = TextEncoderONNX(
            join(checkpoint_path, f"text_encoder.onnx"),
            join(checkpoint_path, f"reranker.onnx"),
            device,
        )

        self.image_encoder = VisualEncoderONNX(join(checkpoint_path, f"image_encoder.onnx"), device)

    def encode_image(
        self,
        images: ndarray,
        return_features: bool = False,
    ) -> Union[ndarray, Tuple[ndarray, ndarray]]:
        """Passes the pre-processed images through `image_encoder` to produce images features (optional) and embeddings.

        :param images: Preprocessed image
        :param return_features: Whether to return images features or return only embeddings
        """

        features, embeddings = self.image_encoder(images)

        if return_features:
            return features, embeddings

        return embeddings

    def encode_text(
        self,
        texts: Dict[str, ndarray],
        return_features: bool = False,
    ) -> Union[ndarray, Tuple[ndarray, ndarray]]:
        """Passes the pre-processed texts through `text_encoder` to produce texts features (optional) and embeddings.

        :param texts: Dictionary with tokenized texts and attention masks
        :param return_features: Whether to return texts features or return only embeddings
        """

        features, embeddings = self.text_encoder(**texts)

        if return_features:
            return features, embeddings

        return embeddings

    def encode_multimodal(
        self,
        image: Optional[ndarray] = None,
        text: Dict[str, ndarray] = None,
        image_features: Optional[ndarray] = None,
        text_features: Optional[ndarray] = None,
        attention_mask: Optional[ndarray] = None,
        return_scores: bool = False,
    ) -> Union[ndarray, Tuple[ndarray, ndarray]]:
        """Passes preprocessed texts (or precomputed texts features) and
            preprocessed images (or precomputed images features) through multimodal encoded to produce matching scores and optionally multimodal joint embeddings.

        :param image: Preprocessed images
        :param text: Preprocessed texts
        :param image_features: Precomputed images features
        :param text_features: Precomputed text features
        :param attention_mask: Attention masks, not required if pass `text` instead of text_features
        """

        assert image is not None or image_features is not None, "Either `image` or `image_features` should be non None"
        assert text is not None or text_features is not None, "Either `text_data` or `text_features` should be non None"

        if text_features is not None:
            assert attention_mask is not None, "if `text_features` is not None, then you should pass `attention_mask`"

        if image_features is None:
            image_features = self.image_encoder(image)

        if text_features is None:
            text_features = self.text_encoder(
                text["input_ids"],
                text["attention_mask"],
            )

        matching_scores, embeddings = self.text_encoder.forward_multimodal(
            text_features,
            attention_mask if attention_mask is not None else text["attention_mask"],
            image_features,
        )

        if return_scores:
            return matching_scores, embeddings

        return embeddings

    def forward(
        self,
        images: ndarray,
        texts: Dict[str, ndarray],
    ) -> Union[ndarray, ndarray]:
        """Inference forward method

        :param images: Preprocessed images
        :param texts: Preprocessed texts
        :return: embeddings for images and texts
        """
        _, image_embeddings = self.image_encoder(images)
        _, text_embeddings = self.text_encoder(texts)
        return image_embeddings, text_embeddings

    @property
    def text_features_dim(self) -> int:
        """Dimensionality of the text encoder features."""

        return self._text_encoder_dim

    @property
    def image_features_dim(self) -> int:
        """Dimensionality of the image encoder features."""

        return self._image_encoder_dim

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of shared space embedding."""

        return self._embedding_dim

    @property
    def multimodal_embedding_dim(self) -> int:
        """Dimensionality of multimodal joint embedding."""
        return self._text_encoder_dim
