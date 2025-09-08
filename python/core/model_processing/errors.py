from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.core.model_processing.converters.base import ModelType


class ModelConversionError(Exception):
    """Base class for all model conversion errors."""

    def __init__(
        self: ModelConversionError,
        message: str,
        model_type: ModelType,
        context: dict | None = None,
    ) -> None:
        self.message = message
        self.model_type = model_type
        self.context = context or {}
        super().__init__(self.__str__())

    def __str__(self: ModelConversionError) -> str:
        msg = f"Error converting {self.model_type.value} model: {self.message}"
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            msg += f"(Context: {ctx_str})"
        return msg


class ModelLoadError(ModelConversionError):
    """Raised when an ONNX model cannot be loaded."""

    def __init__(
        self: ModelLoadError,
        file_path: str,
        model_type: ModelType,
        reason: str = "",
    ) -> None:
        message = f"Failed to load {model_type.value} model from '{file_path}'."
        if reason:
            message += f" Reason: {reason}"
        super().__init__(message, model_type, context={"file_path": file_path})


class ModelSaveError(ModelConversionError):
    """Raised when saving a model fails."""

    def __init__(
        self: ModelSaveError,
        file_path: str,
        model_type: ModelType,
        reason: str = "",
    ) -> None:
        message = f"Failed to save {model_type.value} model to '{file_path}'."
        if reason:
            message += f" Reason: {reason}"
        super().__init__(message, model_type, context={"file_path": file_path})


class InferenceError(ModelConversionError):
    """Raised when inference via ONNX Runtime fails."""

    def __init__(
        self: InferenceError,
        model_type: ModelType,
        model_path: str | None = None,
        reason: str = "",
    ) -> None:
        message = f" {model_type.value} inference failed."
        if model_path:
            message += f" Model: '{model_path}'."
        if reason:
            message += f" Reason: {reason}"
        super().__init__(message, model_type, context={"model_path": model_path})


class LayerAnalysisError(ModelConversionError):
    """Raised when analyzing model layers fails."""

    def __init__(
        self: LayerAnalysisError,
        model_type: ModelType,
        layer_name: str | None = None,
        reason: str = "",
    ) -> None:
        message = "Layer analysis failed."
        if layer_name:
            message += f" Problematic layer: '{layer_name}'."
        if reason:
            message += f" Reason: {reason}"
        super().__init__(message, model_type, context={"layer_name": layer_name})


class IOInfoExtractionError(ModelConversionError):
    """Raised when extracting input/output info fails."""

    def __init__(
        self: IOInfoExtractionError,
        model_type: ModelType,
        model_path: str | None = None,
        reason: str = "",
    ) -> None:
        message = "Failed to extract input/output info from model."
        if model_path:
            message += f" Model: '{model_path}'."
        if reason:
            message += f" Reason: {reason}"
        super().__init__(message, model_type, context={"model_path": model_path})


class InvalidModelError(ModelConversionError):
    """Raised when an ONNX model fails validation checks (onnx.checker)."""

    def __init__(
        self: InvalidModelError,
        model_type: ModelType,
        model_path: str | None = None,
        reason: str = "",
    ) -> None:
        msg = f"The {model_type.value} model is invalid."
        if model_path:
            msg += f" Model: '{model_path}'."
        if reason:
            msg += f" Reason: {reason}"
        super().__init__(
            message=msg,
            model_type=model_type,
            context={"model_path": model_path},
        )


class SerializationError(ModelConversionError):
    """Raised when model data cannot be serialized to the required format."""

    def __init__(
        self: SerializationError,
        model_type: ModelType,
        tensor_name: str | None = None,
        reason: str = "",
    ) -> None:
        message = "Failed to serialize model data."
        if tensor_name:
            message += f" Tensor: '{tensor_name}'."
        if reason:
            message += f" Reason: {reason}"
        super().__init__(message, model_type, context={"tensor_name": tensor_name})
