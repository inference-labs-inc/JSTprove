from __future__ import annotations

import math
from pathlib import Path
from typing import TypeVar

import numpy as np
import torch

from python.core.circuits.zk_model_base import ZKModelBase
from python.core.model_processing.converters.onnx_converter import (
    ONNXConverter,
    ONNXOpQuantizer,
)
from python.core.model_processing.onnx_quantizer.layers.base import BaseOpQuantizer

T = TypeVar("T", bound="ONNXConverter")


class GenericModelONNX(ONNXConverter, ZKModelBase):
    """
    A generic ONNX-based Zero-Knowledge (ZK) circuit model wrapper.

    This class provides:
        - Integration for ONNX model loading, quantization (in `ONNXConverter`)
          and ZK circuit infrastructure (in `ZKModelBase`).
        - Support for model quantization via `ONNXOpQuantizer`.
        - Input/output scaling and formatting utilities for ZK compatibility.

    Attributes
    ----------
    name : str
        Internal identifier for the binary to be run in rust backend.
    op_quantizer : ONNXOpQuantizer
        Operator quantizer for applying custom ONNX quantization rules.
    rescale_config : dict
        Per-node override for rescaling during quantization.
        Keys are node names, values are booleans.
        If not specified, assumption is to rescale each layer
    model_file_name : str
        Path to the ONNX model file used for the circuit.
    scale_base : int
        Base multiplier for scaling (default: 2).
    scale_exponent : int
        Exponent applied to `scale_base` for final scaling factor.

    Parameters
    ----------
    model_name : str
        Name of the model to load (with or without `.onnx` extension).

    Notes
    -----
    - The scaling factor (`scale_base ** scale_exponent`) determines how floating point
      inputs/outputs are represented as integers inside the ZK circuit.
    - By default, scaling is fixed; dynamic scaling based on model analysis
      is planned for future implementation.
    - The quantization logic assumes operators are registered with
      `ONNXOpQuantizer`.
    """

    def __init__(self: T, model_name: str) -> None:
        self.name = "onnx_generic_circuit"
        self.op_quantizer = ONNXOpQuantizer()
        self.rescale_config = {}
        self.model_file_name = self.find_model(model_name)
        self.scale_base = 2
        self.scale_exponent = 18
        ONNXConverter.__init__(self)

    def find_model(self: T, model_name: str) -> str:
        """Resolve the ONNX model file path.

        Args:
            model_name (str): Name of the model (with or without `.onnx` extension).

        Returns:
            str: Full path to the model file.
        """
        if ".onnx" not in model_name:
            model_name = model_name + ".onnx"
        if Path(model_name).exists():
            return model_name
        if "models_onnx" in model_name:
            return model_name
        return f"models_onnx/{model_name}"

    def adjust_inputs(self: T, input_file: str) -> str:
        """Preprocess and flatten model inputs for the circuit.

        Args:
            input_file (str): Input data file or array compatible with the model.

        Returns:
            str: Adjusted input file after reshaping and scaling.
        """
        input_shape = self.input_shape.copy()
        shape = self.adjust_shape(input_shape)
        self.input_shape = [math.prod(shape)]
        x = super().adjust_inputs(input_file)
        self.input_shape = input_shape.copy()
        return x

    def get_outputs(
        self: T,
        inputs: np.ndarray | list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Run inference and flatten outputs.

        Args:
            inputs (List[int]): Preprocessed model inputs.

        Returns:
            torch.Tensor: Flattened model outputs as a tensor.
        """
        return torch.as_tensor(np.array(super().get_outputs(inputs))).flatten()

    def format_inputs(
        self: T,
        inputs: np.ndarray | list[int] | torch.Tensor,
    ) -> dict[str, list[int]]:
        """Format raw inputs into scaled integer tensors for the circuit
        and transformed into json to be sent to rust backend.
        Inputs are scaled by `scale_base ** scale_exponent`
        and converted to long to ensure compatibility with ZK circuits

        Args:
            inputs (Any): Raw model inputs.

        Returns:
            Dict[str, List[int]]: Dictionary mapping `input` to scaled integer values.
        """
        x = {"input": inputs}
        scaling = BaseOpQuantizer.get_scaling(
            scale_base=self.scale_base,
            scale_exponent=self.scale_exponent,
        )
        for key in x:
            x[key] = torch.as_tensor(x[key]).flatten().tolist()
            x[key] = (torch.as_tensor(x[key]) * scaling).long().tolist()
        return x


if __name__ == "__main__":
    pass
