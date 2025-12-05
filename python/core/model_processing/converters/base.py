from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import onnx
    import torch


class ModelType(Enum):
    ONNX = "ONNX"


ONNXLayerDict = dict[
    str,
    int | str | list[str] | dict[str, list[int]] | list | None | dict,
]

CircuitParamsDict = dict[str, int | dict[str, bool]]


class ModelConverter(ABC):
    """
    Abstract base class for AI model conversion, quantization, and I/O operations.

    This class defines the required interface for implementing a model converter
    that can handle:
    - Saving/loading models in various formats
    - Quantizing models
    - Extracting model weights
    - Generating model outputs

    Concrete subclasses must implement all abstract methods to provide
    model-specific conversion logic.
    """

    @abstractmethod
    def save_model(self: ModelConverter, file_path: str) -> None:
        """Save the current model to the specified file path.

        Args:
            file_path (str): Path to save the model file.
        """

    @abstractmethod
    def load_model(
        self: ModelConverter,
        file_path: str,
        model_type: ModelType | None = None,
    ) -> onnx.ModelProto:
        """
        Load a model from a file.

        Args:
            file_path (str): Path to the model file.
            model_type (Optional[ModelType]):
                Optional identifier for the model format/type.
                Useful if multiple formats are supported.

        Returns:
            onnx.ModelProto: The loaded model.
        """

    @abstractmethod
    def save_quantized_model(self: ModelConverter, file_path: str) -> None:
        """Save the quantized version of the model to the specified file path.

        Args:
            file_path (str): Path to save the quantized model file.
        """

    @abstractmethod
    def load_quantized_model(self: ModelConverter, file_path: str) -> None:
        """Load a quantized model from a file.

        Args:
            file_path (str): Path to the quantized model file.
        """

    @abstractmethod
    def quantize_model(
        self: ModelConverter,
        model: onnx.ModelProto,
        scale_base: int,
        scale_exponent: int,
        rescale_config: dict | None = None,
    ) -> onnx.ModelProto:
        """Quantize a model with a given scale and optional rescaling configuration.

        Args:
            model (onnx.ModelProto): The model instance to quantize.
            scale_base (int): Base for fixed-point scaling (e.g., 2).
            scale_exponent (int): Quantization scale factor.
            rescale_config (Optional[dict], optional):
                Configuration for rescaling layers or weights during quantization.
                Defaults to None.

        Returns:
            onnx.ModelProto: The quantized model.
        """

    @abstractmethod
    def get_weights(
        self: ModelConverter,
    ) -> tuple[
        dict[str, list[ONNXLayerDict]],
        dict[str, list[ONNXLayerDict]],
        CircuitParamsDict,
    ]:
        """Retrieve the model's weights.

        Returns:
            tuple[dict[str, list[ONNXLayerDict]],
            dict[str, list[ONNXLayerDict]], CircuitParamsDict]:
                A tuple ``(architecture, weights, circuit_params)``:
                - ``architecture``: dict with serialized ``architecture`` layers.
                - ``weights``: dict containing ``w_and_b`` (serialized tensors).
                - ``circuit_params``: dict containing scaling parameters and
                  ``rescale_config``.
        """

    @abstractmethod
    def get_model_and_quantize(self: ModelConverter) -> None:
        """Retrieve the model and quantize it in a single operation."""

    @abstractmethod
    def get_outputs(
        self: ModelConverter,
        inputs: np.ndarray | torch.Tensor,
    ) -> list[np.ndarray]:
        """
        Run inference on the given inputs and return model outputs.

        Args:
            inputs (np.ndarray | torch.Tensor):
                Input data in the format expected by the model.

        Returns:
            list[np.ndarray]: Model outputs after processing the inputs.
        """
