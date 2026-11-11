"""Image preprocessing operations used by the OCR application.

Images are represented as RGB arrays when they have three channels. OpenCV
operations are applied internally using grayscale or binary uint8 arrays as
appropriate, but the public color conversion API never silently swaps RGB and
BGR channels.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping
from uuid import uuid4

import cv2
import numpy as np


COLOR_RGB = 0
COLOR_GRAYSCALE = 1
COLOR_BINARY = 2

_BINARY_OPERATIONS = {"erode", "dilate", "open", "close"}
_GRAYSCALE_OPERATIONS = _BINARY_OPERATIONS | {"white_tophat", "black_tophat"}
_KERNEL_SHAPES = {"rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE}
_RECT_ORIENTATIONS = {"square", "horizontal", "vertical"}


def _validate_image(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a NumPy array")
    if image.ndim not in (2, 3):
        raise ValueError("image must be a 2D grayscale or 3D color array")
    if image.ndim == 3 and image.shape[2] not in (3, 4):
        raise ValueError("color images must have 3 RGB or 4 RGBA channels")
    if image.size == 0:
        raise ValueError("image must not be empty")
    return image


def _as_uint8(image: np.ndarray) -> np.ndarray:
    image = _validate_image(image)
    if image.dtype == np.uint8:
        return image.copy()

    if np.issubdtype(image.dtype, np.floating) and np.nanmax(image) <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    image = _as_uint8(image)
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def convert_color(image: np.ndarray, mode: int = COLOR_RGB) -> np.ndarray:
    """Convert an image to RGB, grayscale, or Otsu-binarized output.

    Args:
        image: Input grayscale, RGB, or RGBA image.
        mode: ``0`` for RGB, ``1`` for grayscale, or ``2`` for binary output
            using Otsu thresholding.

    Returns:
        A new uint8 image in the requested mode. RGB is the canonical color
        channel order; mode 0 does not perform an RGB/BGR swap.
    """

    if mode not in (COLOR_RGB, COLOR_GRAYSCALE, COLOR_BINARY):
        raise ValueError("mode must be 0 (RGB), 1 (grayscale), or 2 (binary)")

    image = _as_uint8(image)
    grayscale = _to_grayscale(image)

    if mode == COLOR_GRAYSCALE:
        return grayscale
    if mode == COLOR_BINARY:
        _, binary = cv2.threshold(
            grayscale,
            0,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
        return binary

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def reverse_color(image: np.ndarray) -> np.ndarray:
    """Invert every image channel, exchanging black and white values."""

    return cv2.bitwise_not(_as_uint8(image))


def _validate_morphology_parameters(
    operation: str,
    kernel_size: int,
    iterations: int,
    allowed_operations: set[str],
) -> tuple[str, int, int]:
    operation = operation.lower().replace("-", "_").replace(" ", "_")
    if operation not in allowed_operations:
        allowed = ", ".join(sorted(allowed_operations))
        raise ValueError(f"unsupported morphology operation; choose from {allowed}")
    if not isinstance(kernel_size, (int, np.integer)) or kernel_size < 1:
        raise ValueError("kernel_size must be a positive integer")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    if not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    return operation, int(kernel_size), int(iterations)


def _kernel(
    kernel_size: int,
    kernel_shape: str,
    kernel_orientation: str = "square",
) -> np.ndarray:
    kernel_shape = kernel_shape.lower().replace("-", "_")
    if kernel_shape not in _KERNEL_SHAPES:
        raise ValueError("kernel_shape must be 'rect' or 'ellipse'")

    if kernel_shape == "rect":
        kernel_orientation = kernel_orientation.lower().replace("-", "_")
        if kernel_orientation not in _RECT_ORIENTATIONS:
            raise ValueError(
                "kernel_orientation must be 'square', 'horizontal', or 'vertical'"
            )
        kernel_dimensions = {
            "square": (kernel_size, kernel_size),
            "horizontal": (kernel_size, 1),
            "vertical": (1, kernel_size),
        }[kernel_orientation]
    else:
        kernel_dimensions = (kernel_size, kernel_size)

    return cv2.getStructuringElement(
        _KERNEL_SHAPES[kernel_shape],
        kernel_dimensions,
    )


def _morphology(
    image: np.ndarray,
    operation: str,
    kernel_size: int,
    iterations: int,
    kernel_shape: str,
    kernel_orientation: str,
    allowed_operations: set[str],
) -> np.ndarray:
    operation, kernel_size, iterations = _validate_morphology_parameters(
        operation,
        kernel_size,
        iterations,
        allowed_operations,
    )
    structuring_element = _kernel(kernel_size, kernel_shape, kernel_orientation)

    if operation == "erode":
        return cv2.erode(image, structuring_element, iterations=iterations)
    if operation == "dilate":
        return cv2.dilate(image, structuring_element, iterations=iterations)

    operation_code = {
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "white_tophat": cv2.MORPH_TOPHAT,
        "black_tophat": cv2.MORPH_BLACKHAT,
    }[operation]
    return cv2.morphologyEx(
        image,
        operation_code,
        structuring_element,
        iterations=iterations,
    )


def binary_morphology(
    image: np.ndarray,
    operation: str = "open",
    kernel_size: int = 3,
    iterations: int = 1,
    kernel_shape: str = "rect",
    kernel_orientation: str = "square",
) -> np.ndarray:
    """Apply morphology to a binary image.

    Non-binary inputs are converted with Otsu thresholding using white as the
    foreground. The output always contains only 0 and 255 values.
    """

    grayscale = _to_grayscale(image)
    if np.all(np.isin(np.unique(grayscale), (0, 255))):
        binary = grayscale
    else:
        _, binary = cv2.threshold(
            grayscale,
            0,
            255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
    result = _morphology(
        binary,
        operation,
        kernel_size,
        iterations,
        kernel_shape,
        kernel_orientation,
        _BINARY_OPERATIONS,
    )
    return np.where(result > 0, 255, 0).astype(np.uint8)


def grayscale_morphology(
    image: np.ndarray,
    operation: str = "open",
    kernel_size: int = 3,
    iterations: int = 1,
    kernel_shape: str = "rect",
    kernel_orientation: str = "square",
) -> np.ndarray:
    """Apply grayscale morphology, including white/black top-hat filters."""

    grayscale = _to_grayscale(image)
    return _morphology(
        grayscale,
        operation,
        kernel_size,
        iterations,
        kernel_shape,
        kernel_orientation,
        _GRAYSCALE_OPERATIONS,
    )


def remove_artifact(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Remove small salt-and-pepper artifacts with a median filter."""

    if not isinstance(kernel_size, (int, np.integer)) or kernel_size < 3:
        raise ValueError("kernel_size must be an integer greater than or equal to 3")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    image = _as_uint8(image)
    return cv2.medianBlur(image, int(kernel_size))


@dataclass
class PreprocessPipeline:
    """An editable, serializable sequence of preprocessing transformations."""

    steps: list[dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.steps = deepcopy(self.steps or [])
        for step in self.steps:
            step.setdefault("id", uuid4().hex)
            step.setdefault("params", {})

    @staticmethod
    def default_step(step_type: str) -> dict[str, Any]:
        defaults: dict[str, dict[str, Any]] = {
            "convert_color": {"mode": COLOR_GRAYSCALE},
            "reverse_color": {},
            "filter": {"kernel_size": 3},
            "binary_morphology": {
                "operation": "open",
                "kernel_size": 3,
                "iterations": 1,
                "kernel_shape": "rect",
                "kernel_orientation": "square",
            },
            "grayscale_morphology": {
                "operation": "open",
                "kernel_size": 3,
                "iterations": 1,
                "kernel_shape": "rect",
                "kernel_orientation": "square",
            },
        }
        if step_type not in defaults:
            raise ValueError(f"unsupported pipeline step: {step_type}")
        return {
            "id": uuid4().hex,
            "type": step_type,
            "params": deepcopy(defaults[step_type]),
        }

    def add_step(
        self,
        step_type: str,
        params: Mapping[str, Any] | None = None,
    ) -> int:
        step = self.default_step(step_type)
        if params:
            step["params"].update(dict(params))
        self.steps.append(step)
        return len(self.steps) - 1

    def remove_step(self, index: int) -> None:
        self._check_index(index)
        self.steps.pop(index)

    def move_step(self, index: int, new_index: int) -> None:
        self._check_index(index)
        if not 0 <= new_index < len(self.steps):
            raise IndexError("new_index is outside the pipeline")
        step = self.steps.pop(index)
        self.steps.insert(new_index, step)

    def update_step(self, index: int, params: Mapping[str, Any]) -> None:
        self._check_index(index)
        self.steps[index].setdefault("params", {}).update(dict(params))

    def apply(self, image: np.ndarray) -> np.ndarray:
        result = _as_uint8(image)
        for step in self.steps:
            step_type = step.get("type")
            params = dict(step.get("params", {}))
            if step_type == "convert_color":
                result = convert_color(result, **params)
            elif step_type == "reverse_color":
                result = reverse_color(result, **params)
            elif step_type == "filter":
                result = remove_artifact(result, **params)
            elif step_type == "binary_morphology":
                result = binary_morphology(result, **params)
            elif step_type == "grayscale_morphology":
                result = grayscale_morphology(result, **params)
            else:
                raise ValueError(f"unsupported pipeline step: {step_type}")
        return result

    def to_dict(self) -> list[dict[str, Any]]:
        return deepcopy(self.steps)

    def _check_index(self, index: int) -> None:
        if not 0 <= index < len(self.steps):
            raise IndexError("pipeline step index is outside the pipeline")
