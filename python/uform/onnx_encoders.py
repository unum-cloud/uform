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


class ImageEncoder:
    def __init__(
        self,
        model_path: str,
        *,
        device: Literal["cpu", "cuda"] = "cpu",
        return_features: bool = True,
    ):
        """
        :param model_path: Path to onnx model
        :param device: Device name, either cpu or gpu
        """

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.return_features = return_features
        self.session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=available_providers(device),
        )

    def __call__(
        self, images: ndarray, return_features: Optional[bool] = None
    ) -> Union[ndarray, Tuple[ndarray, ndarray]]:
        features, embeddings = self.session.run(None, {"images": images})
        return_features = return_features if return_features is not None else self.return_features
        if return_features:
            return features, embeddings
        return embeddings


class TextEncoder:
    def __init__(
        self,
        model_path: str,
        *,
        device: Literal["cpu", "cuda"] = "cpu",
        return_features: bool = True,
    ):
        """
        :param text_encoder_path: Path to onnx of text encoder
        :param device: Device name, either cpu or gpu
        """

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.return_features = return_features
        self.text_encoder_session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=available_providers(device),
        )

    def __call__(
        self,
        x: Union[ndarray, dict],
        attention_mask: Optional[ndarray] = None,
        return_features: Optional[bool] = None,
    ) -> Union[ndarray, Tuple[ndarray, ndarray]]:
        if isinstance(x, dict):
            assert attention_mask is None, "If `x` is a dictionary, then `attention_mask` should be None"
            attention_mask = x["attention_mask"]
            input_ids = x["input_ids"]
        else:
            input_ids = x

        features, embeddings = self.text_encoder_session.run(
            None, {"input_ids": input_ids, "attention_mask": attention_mask}
        )

        return_features = return_features if return_features is not None else self.return_features
        if return_features:
            return features, embeddings
        return embeddings
