"""Image preprocessing operations used by the OCR application.

Images are represented as RGB arrays when they have three channels. OpenCV
operations are applied internally using grayscale or binary uint8 arrays as
appropriate, but the public color conversion API never silently swaps RGB and
BGR channels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
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


def _otsu_threshold(grayscale: np.ndarray) -> np.ndarray:
    """Apply Otsu thresholding to a grayscale uint8 image."""
    _, binary = cv2.threshold(
        grayscale,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
    return binary


# ---------------------------------------------------------------------------
# Standalone image transformation functions
# ---------------------------------------------------------------------------


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
        return _otsu_threshold(grayscale)

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def reverse_color(image: np.ndarray) -> np.ndarray:
    """Invert every image channel, exchanging black and white values."""
    return cv2.bitwise_not(_as_uint8(image))


def remove_artifact(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Remove small salt-and-pepper artifacts with a median filter."""
    if not isinstance(kernel_size, (int, np.integer)) or kernel_size < 3:
        raise ValueError("kernel_size must be an integer greater than or equal to 3")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    image = _as_uint8(image)
    return cv2.medianBlur(image, int(kernel_size))


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
        binary = _otsu_threshold(grayscale)

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


# ---------------------------------------------------------------------------
# Transformation parameter schema and plugin classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    """Descriptor for a configurable transformation parameter."""

    name: str
    label: str
    type: str  # "select", "slider", "number"
    default: Any
    options: list[Any] | None = None
    option_labels: dict[Any, str] | None = None
    min_value: int | float | None = None
    max_value: int | float | None = None
    step: int | float | None = None
    help: str | None = None
    visible_when: dict[str, Any] | None = None


class BaseTransformation(ABC):
    """Abstract base class for all preprocessing transformations."""

    name: str = ""
    label: str = ""
    description: str = ""
    param_specs: tuple[ParamSpec, ...] = ()

    def __init__(
        self,
        step_id: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> None:
        self.id: str = step_id or uuid4().hex
        self.params: dict[str, Any] = self.default_params()
        if params:
            self.params.update(dict(params))

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {spec.name: spec.default for spec in cls.param_specs}

    @abstractmethod
    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply the transformation to an input image."""
        ...

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.name,
            "params": deepcopy(self.params),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BaseTransformation:
        step_id = data.get("id") or uuid4().hex
        params = data.get("params", {})
        return cls(step_id=step_id, params=params)

    # Dictionary-like access for backwards compatibility
    def __getitem__(self, key: str) -> Any:
        if key == "id":
            return self.id
        if key == "type":
            return self.name
        if key == "params":
            return self.params
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key == "id":
            self.id = value
        elif key == "params":
            self.params = value
        else:
            raise KeyError(f"Cannot set key '{key}' on {self.__class__.__name__}")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key: str, default: Any = None) -> Any:
        if key in ("id", "type", "params"):
            return self[key]
        return default


class ConvertColorTransform(BaseTransformation):
    """Convert image color mode (RGB, Grayscale, or Otsu Binary)."""

    name = "convert_color"
    label = "Convert color"
    description = "Convert color channel mode between RGB, Grayscale, and Otsu Binary."
    param_specs = (
        ParamSpec(
            name="mode",
            label="Output mode",
            type="select",
            default=COLOR_GRAYSCALE,
            options=[COLOR_RGB, COLOR_GRAYSCALE, COLOR_BINARY],
            option_labels={
                COLOR_RGB: "RGB",
                COLOR_GRAYSCALE: "Grayscale",
                COLOR_BINARY: "Binary (Otsu)",
            },
        ),
    )

    def apply(self, image: np.ndarray) -> np.ndarray:
        return convert_color(image, **self.params)


class ReverseColorTransform(BaseTransformation):
    """Invert image color channels."""

    name = "reverse_color"
    label = "Reverse color"
    description = "Invert all channel values, exchanging black and white."
    param_specs = ()

    def apply(self, image: np.ndarray) -> np.ndarray:
        return reverse_color(image)


class FilterTransform(BaseTransformation):
    """Remove small salt-and-pepper artifacts with a median filter."""

    name = "filter"
    label = "Filter (median)"
    description = "Remove small salt-and-pepper artifacts with a median filter."
    param_specs = (
        ParamSpec(
            name="kernel_size",
            label="Median kernel size",
            type="slider",
            default=3,
            min_value=3,
            max_value=31,
            step=2,
            help="Larger values remove more salt-and-pepper artifacts.",
        ),
    )

    def apply(self, image: np.ndarray) -> np.ndarray:
        return remove_artifact(image, **self.params)


class BinaryMorphologyTransform(BaseTransformation):
    """Morphological operations on binary images."""

    name = "binary_morphology"
    label = "Binary morphology"
    description = "Apply morphological operations (erode, dilate, open, close) on binary images."
    param_specs = (
        ParamSpec(
            name="operation",
            label="Operation",
            type="select",
            default="open",
            options=["erode", "dilate", "open", "close"],
            option_labels={
                "erode": "Erode",
                "dilate": "Dilate",
                "open": "Open",
                "close": "Close",
            },
        ),
        ParamSpec(
            name="kernel_size",
            label="Kernel size",
            type="slider",
            default=3,
            min_value=1,
            max_value=31,
            step=2,
        ),
        ParamSpec(
            name="iterations",
            label="Iterations",
            type="number",
            default=1,
            min_value=1,
            max_value=20,
            step=1,
        ),
        ParamSpec(
            name="kernel_shape",
            label="Kernel shape",
            type="select",
            default="rect",
            options=["rect", "ellipse"],
            option_labels={"rect": "Rectangular", "ellipse": "Elliptical"},
        ),
        ParamSpec(
            name="kernel_orientation",
            label="Rectangle orientation",
            type="select",
            default="square",
            options=["square", "horizontal", "vertical"],
            option_labels={
                "square": "Square",
                "horizontal": "Horizontal",
                "vertical": "Vertical",
            },
            visible_when={"kernel_shape": "rect"},
        ),
    )

    def apply(self, image: np.ndarray) -> np.ndarray:
        return binary_morphology(image, **self.params)


class GrayscaleMorphologyTransform(BaseTransformation):
    """Morphological operations on grayscale images."""

    name = "grayscale_morphology"
    label = "Grayscale morphology"
    description = "Apply morphological operations on grayscale images including top-hat filters."
    param_specs = (
        ParamSpec(
            name="operation",
            label="Operation",
            type="select",
            default="open",
            options=[
                "erode",
                "dilate",
                "open",
                "close",
                "white_tophat",
                "black_tophat",
            ],
            option_labels={
                "erode": "Erode",
                "dilate": "Dilate",
                "open": "Open",
                "close": "Close",
                "white_tophat": "White tophat",
                "black_tophat": "Black tophat",
            },
        ),
        ParamSpec(
            name="kernel_size",
            label="Kernel size",
            type="slider",
            default=3,
            min_value=1,
            max_value=31,
            step=2,
        ),
        ParamSpec(
            name="iterations",
            label="Iterations",
            type="number",
            default=1,
            min_value=1,
            max_value=20,
            step=1,
        ),
        ParamSpec(
            name="kernel_shape",
            label="Kernel shape",
            type="select",
            default="rect",
            options=["rect", "ellipse"],
            option_labels={"rect": "Rectangular", "ellipse": "Elliptical"},
        ),
        ParamSpec(
            name="kernel_orientation",
            label="Rectangle orientation",
            type="select",
            default="square",
            options=["square", "horizontal", "vertical"],
            option_labels={
                "square": "Square",
                "horizontal": "Horizontal",
                "vertical": "Vertical",
            },
            visible_when={"kernel_shape": "rect"},
        ),
    )

    def apply(self, image: np.ndarray) -> np.ndarray:
        return grayscale_morphology(image, **self.params)


# ---------------------------------------------------------------------------
# PreprocessPipeline holding all available transformation methods
# ---------------------------------------------------------------------------


class PreprocessPipeline:
    """An editable, serializable sequence of preprocessing transformations."""

    _REGISTRY: dict[str, type[BaseTransformation]] = {}

    @classmethod
    def register_transformation(cls, transform_cls: type[BaseTransformation]) -> None:
        """Register a transformation class into the pipeline registry."""
        if not transform_cls.name:
            raise ValueError("Transformation class must define a non-empty 'name'.")
        cls._REGISTRY[transform_cls.name] = transform_cls

    @classmethod
    def get_available_transformations(cls) -> dict[str, type[BaseTransformation]]:
        """Return all registered transformation classes keyed by transformation name."""
        return dict(cls._REGISTRY)

    @classmethod
    def get_transformation_class(cls, step_type: str) -> type[BaseTransformation]:
        """Look up a transformation class by name, with alias support."""
        if step_type in cls._REGISTRY:
            return cls._REGISTRY[step_type]
        # Alias support
        if step_type == "remove_artifact" and "filter" in cls._REGISTRY:
            return cls._REGISTRY["filter"]
        raise ValueError(f"unsupported pipeline step: {step_type}")

    @classmethod
    def default_step(cls, step_type: str) -> dict[str, Any]:
        """Return a default step dictionary for backward compatibility."""
        transform_cls = cls.get_transformation_class(step_type)
        return {
            "id": uuid4().hex,
            "type": transform_cls.name,
            "params": transform_cls.default_params(),
        }

    def __init__(
        self,
        steps: list[BaseTransformation | Mapping[str, Any]] | None = None,
    ) -> None:
        self.steps: list[BaseTransformation] = []
        if steps:
            for step in steps:
                if isinstance(step, BaseTransformation):
                    self.steps.append(step)
                elif isinstance(step, Mapping):
                    step_type = step.get("type")
                    if not step_type:
                        raise ValueError("Step dictionary missing 'type' key")
                    transform_cls = self.get_transformation_class(str(step_type))
                    self.steps.append(transform_cls.from_dict(step))
                else:
                    raise TypeError(f"Invalid step type: {type(step)}")

    def add_step(
        self,
        step_type: str,
        params: Mapping[str, Any] | None = None,
    ) -> int:
        """Instantiate and append a transformation step to the pipeline."""
        transform_cls = self.get_transformation_class(step_type)
        step = transform_cls(params=params)
        self.steps.append(step)
        return len(self.steps) - 1

    def remove_step(self, index: int) -> None:
        """Remove a transformation step by index."""
        self._check_index(index)
        self.steps.pop(index)

    def move_step(self, index: int, new_index: int) -> None:
        """Move a transformation step from index to new_index."""
        self._check_index(index)
        if not 0 <= new_index < len(self.steps):
            raise IndexError("new_index is outside the pipeline")
        step = self.steps.pop(index)
        self.steps.insert(new_index, step)

    def update_step(self, index: int, params: Mapping[str, Any]) -> None:
        """Update parameters for the transformation step at index."""
        self._check_index(index)
        self.steps[index].params.update(dict(params))

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply all pipeline transformations in order.

        If no transformations are configured, the input image is returned
        directly without any modification or type conversion.
        """
        if not self.steps:
            return image

        result = _as_uint8(image)
        for step in self.steps:
            result = step.apply(result)
        return result

    def to_dict(self) -> list[dict[str, Any]]:
        """Serialize pipeline steps to a list of dictionaries."""
        return [step.to_dict() for step in self.steps]

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | list[dict[str, Any]],
    ) -> PreprocessPipeline:
        """Create a PreprocessPipeline from saved settings or a steps list."""
        if isinstance(data, dict):
            if data.get("version") not in (1, 2):
                raise ValueError(f"unsupported settings version: {data.get('version')}")
            steps = data.get("steps")
            if not isinstance(steps, list):
                raise ValueError("settings must contain a list of pipeline steps")
            return cls(steps)
        if isinstance(data, list):
            return cls(data)
        raise TypeError("data must be a dictionary or list")

    def _check_index(self, index: int) -> None:
        if not 0 <= index < len(self.steps):
            raise IndexError("pipeline step index is outside the pipeline")


# Register built-in transformations into PreprocessPipeline
PreprocessPipeline.register_transformation(ConvertColorTransform)
PreprocessPipeline.register_transformation(ReverseColorTransform)
PreprocessPipeline.register_transformation(FilterTransform)
PreprocessPipeline.register_transformation(BinaryMorphologyTransform)
PreprocessPipeline.register_transformation(GrayscaleMorphologyTransform)
