from __future__ import annotations

import json
from pathlib import Path

import torch

from python.core.utils.errors import (
    CircuitUtilsError,
    InputFileError,
    MissingCircuitAttributeError,
    ShapeMismatchError,
)


class GeneralLayerFunctions:
    """
    A collection of utility functions for reading, generating, scaling, and
    formatting model inputs/outputs. This is primarily intended for
    preparing inputs for ONNX models or similar layer-based models.
    """

    def read_input(self: GeneralLayerFunctions, file_name: str) -> list | dict:
        """Read model input data from a JSON file.

        Args:
            file_name (str): Path to the JSON file containing input data.

        Returns:
            Any: The value of the "input" field from the JSON file.
        """
        try:
            with Path(file_name).open("r") as file:
                data = json.load(file)
        except FileNotFoundError as e:
            raise InputFileError(file_name, "File not found", cause=e) from e
        except json.JSONDecodeError as e:
            raise InputFileError(
                file_name,
                f"Invalid JSON format: {e.msg}",
                cause=e,
            ) from e

        if "input" not in data:
            raise InputFileError(file_name, "Missing required 'input' field in JSON")

        return data["input"]

    def get_inputs_from_file(
        self: GeneralLayerFunctions,
        file_name: str,
        *,
        is_scaled: bool = False,
    ) -> torch.Tensor:
        """Load and optionally scale inputs from a file.

        Args:
            file_name (str): Path to the file containing input data.
            is_scaled (bool, optional):
                If True, returns unscaled values. If False, applies scaling using
                `self.scale_base ** self.scale_exponent`. Defaults to False.

        Returns:
            torch.Tensor: The loaded, reshaped, and potentially rescaled input tensor.
        """
        inputs = self.read_input(file_name)
        try:
            tensor = torch.as_tensor(inputs)
        except Exception as e:
            raise InputFileError(
                file_name,
                f"Invalid input data for tensor conversion: {e}",
            ) from e

        if not is_scaled:
            if not (hasattr(self, "scale_base") and hasattr(self, "scale_exponent")):
                attr_name = "scale_base/scale_exponent"
                msg = "needed for scaling"
                raise MissingCircuitAttributeError(
                    attr_name,
                    msg,
                )
            tensor = torch.mul(tensor, self.scale_base**self.scale_exponent)

        tensor = tensor.long()
        return self.reshape_inputs(tensor)

    def reshape_inputs(self: GeneralLayerFunctions, tensor: torch.Tensor) -> None:
        if hasattr(self, "input_shape"):
            shape = self.input_shape
            if hasattr(self, "adjust_shape") and callable(
                self.adjust_shape,
            ):
                shape = self.adjust_shape(shape)
            try:
                tensor = tensor.reshape(shape)
            except RuntimeError as e:
                raise ShapeMismatchError(shape, list(tensor.shape)) from e

        return tensor

    def get_inputs(
        self: GeneralLayerFunctions,
        file_path: str | None = None,
        *,
        is_scaled: bool = False,
    ) -> torch.Tensor:
        """Retrieve model inputs, either from a file or by generating new inputs.

        Args:
            file_path (str, optional):
                Path to the input file. If None,
                new random inputs are generated. Defaults to None.
            is_scaled (bool, optional):
                Whether to skip scaling of loaded inputs. Defaults to False.

        Raises:
            NotImplementedError: If `self.input_shape` is not defined.

        Returns:
            torch.Tensor: The input tensor shaped according to `self.input_shape`.
        """
        if file_path is None:
            attr_name = "input_shape"
            if not hasattr(self, attr_name):
                msg = "needed to generate random inputs"
                raise MissingCircuitAttributeError(
                    attr_name,
                    msg,
                )
            return self.create_new_inputs()

        return self.get_inputs_from_file(file_path, is_scaled=is_scaled).reshape(
            self.input_shape,
        )

    def create_new_inputs(self: GeneralLayerFunctions) -> torch.Tensor:
        """Generate new random input tensors.

        Returns:
            __type__:
                - If `self.input_shape` is a list/tuple, returns a single tensor.
                - If `self.input_shape` is a dict, returns a dictionary mapping
                  input names to tensors.
        """
        attr_name = "input_shape"
        if not hasattr(self, attr_name):
            context = "needed to generate new inputs"
            raise MissingCircuitAttributeError(
                attr_name,
                context,
            )
        # ONNX inputs will be in this form, and require inputs to not be scaled up
        if isinstance(self.input_shape, dict):
            keys = self.input_shape.keys()
            if len(keys) == 1:
                # If unknown dim in batch spot, assume batch size of 1
                first_key = next(iter(keys))
                input_shape = self.input_shape[first_key]
                input_shape[0] = max(input_shape[0], 1)
                return self.get_rand_inputs(input_shape)
            inputs = {}
            for key in keys:
                # If unknown dim in batch spot, assume batch size of 1
                input_shape = self.input_shape[key]
                if not isinstance(input_shape, (list, tuple)):
                    msg = f"Invalid input shape for key '{key}': {input_shape}"
                    raise CircuitUtilsError(msg)
                input_shape[0] = max(input_shape[0], 1)
                inputs[key] = self.get_rand_inputs(input_shape)
            return inputs

        if not (hasattr(self, "scale_base") and hasattr(self, "scale_exponent")):
            attr_name = "scale_base/scale_exponent"
            context = "needed for scaling random inputs"
            raise MissingCircuitAttributeError(
                attr_name,
                context,
            )

        return torch.mul(
            self.get_rand_inputs(self.input_shape),
            self.scale_base**self.scale_exponent,
        ).long()

    def get_rand_inputs(
        self: GeneralLayerFunctions,
        input_shape: list[int],
    ) -> torch.Tensor:
        """Generate random input values in the range [-1, 1).

        Args:
            input_shape (list[int]): Shape of the tensor to generate.

        Returns:
            torch.Tensor: A tensor of random values in [-1, 1).
        """
        if not isinstance(input_shape, (list, tuple)):
            msg = (
                f"Invalid input_shape type: {type(input_shape)}."
                " Expected list or tuple of ints."
            )
            raise CircuitUtilsError(msg)
        if not all(isinstance(x, int) and x > 0 for x in input_shape):
            raise ShapeMismatchError(
                expected_shape="positive integers",
                actual_shape=input_shape,
            )
        return torch.rand(input_shape) * 2 - 1

    def format_inputs(
        self: GeneralLayerFunctions,
        inputs: torch.Tensor,
    ) -> dict[str, list[int]]:
        """Format input tensors for JSON serialization.

        Args:
            inputs (torch.Tensor): The input tensor.

        Returns:
            dict[str, list[int]]:
                A dictionary with the key "input"
                containing the tensor as a list of integers.
        """
        return {"input": inputs.long().tolist()}

    def format_outputs(
        self: GeneralLayerFunctions,
        outputs: torch.Tensor,
    ) -> dict[str, list[int]]:
        """Format output tensors for JSON serialization,
          including rescaled outputs for readability.

        Args:
            outputs (torch.Tensor): _deThe output tensor.cription_

        Returns:
            dict[str, list[int]]: A dictionary containing:
                  - "output": the raw output tensor as a list of integers.
                  - "rescaled_output": the output divided by the scaling factor.
        """
        if hasattr(self, "scale_exponent") and hasattr(self, "scale_base"):
            try:
                rescaled = torch.div(outputs, self.scale_base**self.scale_exponent)
            except Exception as e:
                msg = (
                    "Failed to rescale outputs using scale_base="
                    f"{getattr(self, 'scale_base', None)} "
                    f"and scale_exponent={getattr(self, 'scale_exponent', None)}: {e}"
                )
                raise CircuitUtilsError(msg) from e
            return {
                "output": outputs.long().tolist(),
                "rescaled_output": rescaled.tolist(),
            }
        return {"output": outputs.long().tolist()}

    def format_inputs_outputs(
        self: GeneralLayerFunctions,
        inputs: torch.Tensor,
        outputs: torch.Tensor,
    ) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
        """Format both inputs and outputs for JSON serialization.

        Args:
            inputs (torch.Tensor): Model inputs.
            outputs (torch.Tensor): Model outputs.

        Returns:
            tuple[dict[str, list[int]], dict[str, list[int]]]:
                A tuple containing the formatted inputs and formatted outputs.
        """
        return self.format_inputs(inputs), self.format_outputs(outputs)
