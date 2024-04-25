from enum import Enum
from typing import Union
from os import PathLike
import json


class Modality(Enum):
    TEXT_ENCODER = "text_encoder"
    IMAGE_ENCODER = "image_encoder"
    VIDEO_ENCODER = "video_encoder"
    TEXT_DECODER = "text_decoder"


class ExecutionProviderError(Exception):
    """Exception raised when a requested execution provider is not available."""


ConfigOrPath = Union[PathLike, str, object]


def read_config(path_or_object: ConfigOrPath) -> object:
    if isinstance(path_or_object, (PathLike, str)):
        with open(path_or_object, "r") as f:
            return json.load(f)
    else:
        return path_or_object
