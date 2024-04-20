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
    ):
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


class TextEncoder:
    def __init__(
        self,
        model_path: str,
        *,
        device: Literal["cpu", "cuda"] = "cpu",
    ):
        """
        :param text_encoder_path: Path to onnx of text encoder
        :param device: Device name, either cpu or gpu
        """

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.text_encoder_session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=available_providers(device),
        )

    def __call__(self, input_ids: ndarray, attention_mask: ndarray) -> Tuple[ndarray, ndarray]:
        return self.text_encoder_session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})
