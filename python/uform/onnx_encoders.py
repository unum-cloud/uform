from os import PathLike
from typing import Dict, Optional, Tuple, Union, Literal
import json

import onnxruntime as ort
from numpy import ndarray


class ExecutionProviderError(Exception):
    """Exception raised when a requested execution provider is not available."""


def available_providers(device: Optional[str]) -> Tuple[str, ...]:
    """Returns a tuple of available execution providers based on the requested device.
    https://onnxruntime.ai/docs/execution-providers/

    :param device: Device name, either `cpu` or `gpu`, or a specific execution provider name.
    :return: Tuple of available execution providers.
    :raises ExecutionProviderError: If the requested device is not available.
    """

    gpu_providers = ("CUDAExecutionProvider", "TensorrtExecutionProvider")
    cpu_providers = ("OpenVINOExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider")
    available = ort.get_available_providers()

    # If no target device is specified, let's sort all the available ones with respect to our preference
    if device is None:
        preferences = gpu_providers + cpu_providers
        filtered_preferences = tuple(provider for provider in preferences if provider in available)
        if len(filtered_preferences):
            return filtered_preferences
        if len(available):
            return available
        raise ExecutionProviderError("No execution providers are available")

    # If a GPU is requested, but no GPU providers are available, raise an error
    if device == "gpu" or device == "cuda":
        if all(provider not in available for provider in gpu_providers):
            raise ExecutionProviderError(
                f"GPU providers are not available, consider installing `onnxruntime-gpu` and make sure the CUDA is available on your system. Currently installed: {available}"
            )
        return gpu_providers

    # If a CPU is requested, but no CPU providers are available, raise an error
    if device == "cpu":
        if all(provider not in available for provider in cpu_providers):
            raise ExecutionProviderError(
                f"CPU providers are not available, consider installing `onnxruntime` and make sure the OpenVINO and CoreML are available on your system. Currently installed: {available}"
            )
        return cpu_providers

    if device not in available:
        available_providers = ", ".join(available)
        raise ExecutionProviderError(
            f"Execution provider {device} is not available. Currently installed: {available_providers}"
        )

    return (device,)


class VisualEncoder:
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
        return self.session.run(None, {"input": images})


class TextEncoder:
    def __init__(self, text_encoder_path: str, device: str):
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

    def __call__(self, input_ids: ndarray, attention_mask: ndarray) -> Tuple[ndarray, ndarray]:
        return self.text_encoder_session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})


class TextVisualEncoder:
    def __init__(
        self,
        config_path: PathLike,
        modality_paths: Union[Dict[str, PathLike], PathLike] = None,
        *,
        device: Literal["cpu", "cuda"] = "cpu",
    ):
        """Initializes the model with the configuration and pre-trained weights.

        :param config_path: Path to the JSON model configuration file
        :param modality_paths:  Dictionary with paths to different modalities,
                                or a single path to the model checkpoint
        """
        self.device = device

        config = json.load(open(config_path, "r"))
        self._embedding_dim = config["text_encoder"]["embedding_dim"]
        self._text_encoder_dim = config["text_encoder"]["dim"]
        self._image_encoder_dim = config["image_encoder"]["dim"]

        text_encoder_path = modality_paths.get("text_encoder", None)
        image_encoder_path = modality_paths.get("image_encoder", None)
        self.text_encoder = TextEncoder(text_encoder_path, device) if text_encoder_path else None
        self.image_encoder = VisualEncoder(image_encoder_path, device) if image_encoder_path else None

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


VLM_ONNX = TextVisualEncoder  # legacy
