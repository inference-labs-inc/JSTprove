from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from onnx import numpy_helper

from python.core import RUST_BINARY_NAME
from python.core.circuits.errors import (
    CircuitFileError,
    CircuitProcessingError,
    CircuitRunError,
)
from python.core.circuits.zk_model_base import ZKModelBase

if TYPE_CHECKING:
    import torch

    from python.core.model_processing.converters.onnx_converter import (
        CircuitParamsDict,
        ONNXLayerDict,
    )
from python.core.model_processing.converters.onnx_converter import (
    ONNXConverter,
    ONNXLayer,
    ONNXOpQuantizer,
)
from python.core.model_processing.onnx_quantizer.layers.base import BaseOpQuantizer


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

    def __init__(
        self: GenericModelONNX,
        model_name: str,
        *,
        use_find_model: bool = False,
    ) -> None:
        try:
            self.name = RUST_BINARY_NAME
            self.op_quantizer = ONNXOpQuantizer()
            self.rescale_config = {}
            if use_find_model:
                self.model_file_name = self.find_model(model_name)
            else:
                self.model_file_name = model_name

            self.scale_base = 2
            self.scale_exponent = 18
            self.weights_as_inputs = False
            ONNXConverter.__init__(self)
        except Exception as e:
            msg = f"Failed to initialize GenericModelONNX with model '{model_name}'"
            raise CircuitFileError(
                msg,
                file_path=model_name,
            ) from e

    def find_model(self: GenericModelONNX, model_name: str) -> str:
        """Resolve the ONNX model file path.

        Args:
            model_name (str): Name of the model (with or without `.onnx` extension).

        Returns:
            str: Full path to the model file.
        """
        if ".onnx" not in model_name:
            model_name = model_name + ".onnx"

        # Check direct path first
        if Path(model_name).exists():
            return model_name

        # Check models_onnx directory
        if "models_onnx" in model_name:
            if Path(model_name).exists():
                return model_name
            models_onnx_path = model_name
        else:
            models_onnx_path = f"models_onnx/{model_name}"

        if not Path(models_onnx_path).exists():
            msg = f"Model file not found: '{model_name}'"
            raise CircuitFileError(
                msg,
                file_path=models_onnx_path,
            )
        return models_onnx_path

    def adjust_inputs(
        self: GenericModelONNX,
        inputs: dict[str, np.ndarray],
        input_file: str,
    ) -> str:
        """Preprocess and flatten model inputs for the circuit.

        Args:
            inputs (str): inputs, read from json file
            input_file (str): path to input_file

        Returns:
            str: Adjusted input file after reshaping and scaling.
        """
        try:
            input_shape = self.input_shape.copy()
            shape = self.adjust_shape(input_shape)
            self.input_shape = [math.prod(shape)]
            x = super().adjust_inputs(inputs, input_file)
            self.input_shape = input_shape.copy()
        except Exception as e:
            msg = f"Failed to adjust inputs for GenericModelONNX: {e}"
            raise ValueError(msg) from e
        else:
            return x

    def get_outputs(
        self: GenericModelONNX,
        inputs: np.ndarray | list[int] | torch.Tensor,
    ) -> torch.Tensor:
        """Run inference and flatten outputs.

        Args:
            inputs (List[int]): Preprocessed model inputs.

        Returns:
            torch.Tensor: Flattened model outputs as a tensor.
        """
        import torch  # noqa: PLC0415

        try:
            raw_outputs = super().get_outputs(inputs)
        except Exception as e:
            msg = "Failed to get outputs for GenericModelONNX"
            raise CircuitRunError(
                msg,
                operation="get_outputs",
            ) from e
        else:
            flat_outputs = [o.flatten() for o in raw_outputs]
            combined = np.concatenate(flat_outputs, axis=0)
            return torch.as_tensor(combined)

    def format_inputs(
        self: GenericModelONNX,
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
        import torch  # noqa: PLC0415

        def _raise_type_error(inputs: np.ndarray | list[int] | torch.Tensor) -> None:
            msg = (
                "Expected np.ndarray, torch.Tensor, "
                f"list, or dict for inputs, got {type(inputs)}"
            )

            raise TypeError(msg)

        try:
            if isinstance(inputs, (np.ndarray, torch.Tensor, list)):
                inputs = {"input": inputs}
            elif not isinstance(inputs, dict):
                _raise_type_error(inputs)
            scaling = BaseOpQuantizer.get_scaling(
                scale_base=self.scale_base,
                scale_exponent=self.scale_exponent,
            )

            input_shapes: dict[str, tuple[int, ...]] = {}
            flattened_tensors: list[torch.Tensor] = []

            # Flatten, scale, and collect each input
            for name, value in inputs.items():
                tensor = torch.as_tensor(value)
                input_shapes[name] = tuple(tensor.shape)

                scaled = (tensor * scaling).long().flatten()
                flattened_tensors.append(scaled)

            # Concatenate all inputs into one long tensor
            concatenated = torch.cat(flattened_tensors, dim=0)
            flattened_list = concatenated.tolist()

            # Wrap it into a dict under "input" key to read into rust
            formatted_inputs = {"input": flattened_list}
        except Exception as e:
            msg = f"Failed to format inputs for GenericModelONNX: {e}"
            raise CircuitProcessingError(
                msg,
                operation="format_inputs",
                data_type=type(inputs).__name__,
            ) from e
        else:
            return formatted_inputs

    def get_weights(
        self: GenericModelONNX,
    ) -> dict[str, list[ONNXLayerDict]]:
        _, w_and_b, _ = super().get_weights()
        # Currently want to read these in separately
        return w_and_b

    def get_architecture(
        self: GenericModelONNX,
    ) -> dict[str, list[ONNXLayerDict]]:
        architecture, _, _ = super().get_weights()
        # Currently want to read these in separately
        return architecture

    def _get_input_scales(self: GenericModelONNX) -> dict[str, float]:
        """Read per-input scale factors from the quantized model's initializers.

        Each input ``X`` may have a corresponding ``X_scale`` initializer
        created during quantization (reflecting SCALE_PLAN). Only inputs
        with an explicit scale initializer are included; callers fall back
        to ``scale_base ** scale_exponent`` for unlisted inputs.

        Results are cached on the instance since the quantized model is
        immutable after loading.
        """
        cached = getattr(self, "_cached_input_scales", None)
        if cached is not None:
            return cached

        if not hasattr(self, "quantized_model") or self.quantized_model is None:
            return {}
        init_map = {i.name: i for i in self.quantized_model.graph.initializer}
        scales: dict[str, float] = {}
        graph_input_names = {gi.name for gi in self.quantized_model.graph.input}
        for name in graph_input_names:
            scale_name = f"{name}_scale"
            if scale_name not in init_map:
                continue
            arr = numpy_helper.to_array(init_map[scale_name])
            # Scale initializers are scalar (shape [1]) from insert_scale_node;
            # take flat[0] to stay safe if a future quantizer produces 1-D
            # per-channel scales.
            scales[name] = float(arr.flat[0])
        self._cached_input_scales = scales
        return scales

    def process_for_witness(
        self: GenericModelONNX,
        raw_inputs: dict[str, list],
    ) -> tuple[dict[str, list[int]], dict[str, list]]:
        """Scale inputs and run inference in a single pass.

        Parameters
        ----------
        raw_inputs : dict[str, list]
            User-provided float inputs (e.g. ``{"tile_in": [...]}``).\
            Missing WAI inputs (W, B) are filled from the quantized
            model's initializers automatically.

        Returns
        -------
        tuple[dict[str, list[int]], dict[str, list]]
            ``(circuit_inputs, formatted_outputs)`` where
            ``circuit_inputs`` is ``{"input": [flat ints]}`` and
            ``formatted_outputs`` has ``"output"`` (int) and
            ``"rescaled_output"`` (float) keys.
        """
        import torch  # noqa: PLC0415

        ordered_inputs = {}

        if (
            hasattr(self, "input_shape")
            and isinstance(self.input_shape, dict)
            and hasattr(self, "quantized_model")
            and self.quantized_model is not None
        ):
            provided = set(raw_inputs.keys())
            qm_init = {i.name: i for i in self.quantized_model.graph.initializer}
            for name in self.input_shape:
                if name in provided:
                    ordered_inputs[name] = raw_inputs[name]
                elif name in qm_init:
                    ordered_inputs[name] = numpy_helper.to_array(qm_init[name]).tolist()
                else:
                    src = "raw_inputs" if name not in provided else "initializer"
                    msg = f"Input '{name}' missing from {src}"
                    raise CircuitProcessingError(msg)
        else:
            ordered_inputs = dict(raw_inputs)

        input_scales = self._get_input_scales()
        default_scale = float(self.scale_base**self.scale_exponent)

        scaled = {}
        for name, value in ordered_inputs.items():
            scale = input_scales.get(name, default_scale)
            tensor = torch.as_tensor(value, dtype=torch.float64)
            scaled[name] = (tensor * scale).long().tolist()

        circuit_inputs = self.reshape_inputs_for_circuit(scaled)

        inference_inputs = dict(ordered_inputs)
        inference_inputs = self.reshape_inputs_for_inference(inference_inputs)
        raw_outputs = self.get_outputs(inference_inputs)
        flat = raw_outputs.flatten()

        formatted = {
            "output": flat.long().tolist(),
            "rescaled_output": (flat.double() / default_scale).tolist(),
        }
        return circuit_inputs, formatted

    def get_metadata(
        self: GenericModelONNX,
    ) -> CircuitParamsDict:
        architecture, _, circuit_params = super().get_weights()
        architecture_layers = [
            self._dict_to_onnx_layer(d) for d in architecture.get("architecture", [])
        ]
        n_bits_config = self.compute_n_bits_from_bounds(
            architecture_layers=architecture_layers,
            rescale_config=circuit_params.get("rescale_config", {}),
            scale_base=circuit_params.get("scale_base", 2),
            scale_exponent=circuit_params.get("scale_exponent", 18),
            input_bounds=getattr(self, "input_bounds", (0.0, 1.0)),
        )
        circuit_params["n_bits_config"] = n_bits_config
        circuit_params["weights_as_inputs"] = self.weights_as_inputs
        input_scales = self._get_input_scales()
        if input_scales:
            circuit_params["input_scales"] = input_scales
        return circuit_params

    @staticmethod
    def _dict_to_onnx_layer(d: dict) -> ONNXLayer:
        return ONNXLayer(
            id=d.get("id", 0),
            name=d.get("name", ""),
            op_type=d.get("op_type", ""),
            inputs=d.get("inputs", []),
            outputs=d.get("outputs", []),
            shape=d.get("shape", {}),
            tensor=d.get("tensor"),
            params=d.get("params"),
            opset_version_number=d.get("opset_version_number", -1),
        )


if __name__ == "__main__":
    pass
