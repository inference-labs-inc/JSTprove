from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from python.core.utils.witness_utils import compare_witness_to_io, load_witness

if TYPE_CHECKING:
    import numpy as np
import torch

from python.core.circuits.errors import (
    CircuitConfigurationError,
    CircuitFileError,
    CircuitInputError,
    CircuitProcessingError,
    CircuitRunError,
    WitnessMatchError,
)
from python.core.model_processing.onnx_quantizer.layers.base import BaseOpQuantizer
from python.core.utils.helper_functions import (
    CircuitExecutionConfig,
    RunType,
    ZKProofSystems,
    compile_circuit,
    compute_and_store_output,
    generate_proof,
    generate_verification,
    generate_witness,
    prepare_io_files,
    read_from_json,
    run_end_to_end,
    to_json,
)


class Circuit:
    """
    Base class for all ZK circuits.

    This class defines the standard interface and common utilities for
    building, testing, and running ZK circuits.
    Subclasses are expected to implement circuit-specific logic such as
    input preparation, output computation, and model handling.
    """

    def __init__(self: Circuit) -> None:
        # Default folder paths - can be overridden in subclasses
        self.input_folder = "inputs"
        self.proof_folder = "analysis"
        self.temp_folder = "temp"
        self.circuit_folder = ""
        self.weights_folder = "weights"
        self.output_folder = "output"
        self.proof_system = ZKProofSystems.Expander

        # This will be set by prepare_io_files decorator
        self._file_info = None
        self.required_keys = None
        self.logger = logging.getLogger(__name__)

    def check_attributes(self: Circuit) -> None:
        """
        Check if the necessary attributes are defined in subclasses.
        Must be overridden in subclasses

        Raises:
            CircuitConfigurationError: If required attributes are missing.
        """
        missing = [
            attr
            for attr in ("required_keys", "name", "scale_exponent", "scale_base")
            if not hasattr(self, attr)
        ]
        if missing:
            raise CircuitConfigurationError(missing_attributes=missing)

    def parse_inputs(self: Circuit, **kwargs: dict[str, Any]) -> None:
        """Parse and validate required input parameters
        for the circuit into an instance attribute.

        Args:
            **kwargs (dict[str, Any]): Input parameters to parse and validate.

        Raises:
            NotImplementedError: If `required_keys` is not set.
            KeyError: If any required parameter is missing.
            ValueError: If any parameter value is not an integer or list of integers.
        """
        if self.required_keys is None:
            msg = "self.required_keys must"
            " be specified in the circuit definition."
            raise CircuitConfigurationError(
                msg,
            )
        for key in self.required_keys:
            if key not in kwargs:
                msg = f"Missing required parameter: '{key}'"
                raise CircuitInputError(msg)

            value = kwargs[key]

            # # Validate type (ensure integer)
            if not isinstance(value, (int, list)):
                msg = (
                    f"Parameter '{key}' must be an int or list of ints, "
                    f"got {type(value).__name__}."
                )
                raise CircuitInputError(
                    msg,
                )
            setattr(self, key, value)

    @compute_and_store_output
    def get_outputs(self: Circuit) -> None:
        """
        Compute circuit outputs.
        This method should be implemented by subclasses.
        """
        msg = "get_outputs must be implemented"
        raise NotImplementedError(msg)

    def get_inputs(
        self: Circuit,
        file_path: str | None = None,
        *,
        is_scaled: bool | None = False,
    ) -> None:
        """
        Compute and return the circuit's input values.
        This method should be implemented by subclasses.

        Args:
            file_path (str | None): Optional path to input file.
            is_scaled (bool | None): Whether inputs are scaled.
        """
        _ = file_path, is_scaled
        msg = "get_inputs must be implemented"
        raise NotImplementedError(msg)

    @prepare_io_files
    def base_testing(self: Circuit, exec_config: CircuitExecutionConfig) -> None:
        """Run the circuit in a specified mode
        (testing, proving, compiling, etc.).

        File path resolution is handled automatically by the
        `prepare_io_files` decorator.

        Args:
            exec_config (CircuitExecutionConfig): Configuration object containing
                run_type, file paths, and other execution parameters.

        Raises:
            CircuitConfigurationError: If `_file_info` is not set by the decorator.
        """
        if exec_config.circuit_path is None:
            exec_config.circuit_path = f"{exec_config.circuit_name}.txt"

        if not self._file_info:
            msg = (
                "Circuit file information (_file_info)"
                " must be set by the prepare_io_files decorator."
            )
            raise CircuitConfigurationError(
                msg,
                details={"decorator": "prepare_io_files"},
            )
        exec_config.weights_path = self._file_info.get("weights")

        # Run the appropriate proof operation based on run_type
        self.parse_proof_run_type(exec_config)

    def _raise_unknown_run_type(self: Circuit, run_type: RunType) -> None:
        self.logger.error("Unknown run type: %s", run_type)
        msg = f"Unsupported run type: {run_type}"
        raise CircuitRunError(
            msg,
            operation="parse_proof_run_type",
            details={"run_type": run_type},
        )

    def parse_proof_run_type(
        self: Circuit,
        exec_config: CircuitExecutionConfig,
    ) -> None:
        """Dispatch proof-related operations based on the selected run type.

        Args:
            exec_config (CircuitExecutionConfig): Configuration object containing
                file paths, run type, and other parameters.

        Raises:
            CircuitRunError: If `run_type` is unknown or operation fails.
        """
        is_scaled = True

        try:
            if exec_config.run_type == RunType.END_TO_END:
                self._compile_preprocessing(
                    weights_path=exec_config.weights_path,
                    quantized_path=exec_config.quantized_path,
                )
                processed_input_file = self._gen_witness_preprocessing(
                    input_file=exec_config.input_file,
                    output_file=exec_config.output_file,
                    quantized_path=exec_config.quantized_path,
                    write_json=exec_config.write_json,
                    is_scaled=is_scaled,
                )
                run_end_to_end(
                    circuit_name=exec_config.circuit_name,
                    circuit_path=exec_config.circuit_path,
                    input_file=processed_input_file,
                    output_file=exec_config.output_file,
                    proof_system=exec_config.proof_system,
                    dev_mode=exec_config.dev_mode,
                    ecc=exec_config.ecc,
                )
            elif exec_config.run_type == RunType.COMPILE_CIRCUIT:
                self._compile_preprocessing(
                    weights_path=exec_config.weights_path,
                    quantized_path=exec_config.quantized_path,
                )
                compile_circuit(
                    circuit_name=exec_config.circuit_name,
                    circuit_path=exec_config.circuit_path,
                    proof_system=exec_config.proof_system,
                    dev_mode=exec_config.dev_mode,
                    bench=exec_config.bench,
                )
            elif exec_config.run_type == RunType.GEN_WITNESS:
                processed_input_file = self._gen_witness_preprocessing(
                    input_file=exec_config.input_file,
                    output_file=exec_config.output_file,
                    quantized_path=exec_config.quantized_path,
                    write_json=exec_config.write_json,
                    is_scaled=is_scaled,
                )
                generate_witness(
                    circuit_name=exec_config.circuit_name,
                    circuit_path=exec_config.circuit_path,
                    witness_file=exec_config.witness_file,
                    input_file=processed_input_file,
                    output_file=exec_config.output_file,
                    proof_system=exec_config.proof_system,
                    dev_mode=exec_config.dev_mode,
                    bench=exec_config.bench,
                )
            elif exec_config.run_type == RunType.PROVE_WITNESS:
                generate_proof(
                    circuit_name=exec_config.circuit_name,
                    circuit_path=exec_config.circuit_path,
                    witness_file=exec_config.witness_file,
                    proof_file=exec_config.proof_file,
                    proof_system=exec_config.proof_system,
                    dev_mode=exec_config.dev_mode,
                    ecc=exec_config.ecc,
                    bench=exec_config.bench,
                )
            elif exec_config.run_type == RunType.GEN_VERIFY:
                witness_file = exec_config.witness_file
                output_file = exec_config.output_file
                processed_input_file = self.rename_inputs(exec_config.input_file)
                proof_system = exec_config.proof_system
                if not self.load_and_compare_witness_to_io(
                    witness_path=witness_file,
                    input_path=processed_input_file,
                    output_path=output_file,
                    proof_system=proof_system,
                ):
                    raise WitnessMatchError  # noqa: TRY301
                generate_verification(
                    circuit_name=exec_config.circuit_name,
                    circuit_path=exec_config.circuit_path,
                    input_file=processed_input_file,
                    output_file=output_file,
                    witness_file=witness_file,
                    proof_file=exec_config.proof_file,
                    proof_system=proof_system,
                    dev_mode=exec_config.dev_mode,
                    ecc=exec_config.ecc,
                    bench=exec_config.bench,
                )
            else:
                self._raise_unknown_run_type(exec_config.run_type)
        except CircuitRunError:
            self.logger.exception(
                "Operation %s failed",
                exec_config.run_type,
                extra={"run_type": exec_config.run_type},
            )
            raise
        except (
            CircuitProcessingError,
            CircuitInputError,
            CircuitFileError,
            Exception,
        ) as e:
            self.logger.exception(
                "Operation %s failed",
                exec_config.run_type,
                extra={"run_type": exec_config.run_type},
            )
            raise CircuitRunError(
                operation=exec_config.run_type,
            ) from e

    def load_and_compare_witness_to_io(
        self: Circuit,
        witness_path: str,
        input_path: str,
        output_path: str,
        proof_system: ZKProofSystems,
    ) -> bool:
        """
        Load a witness from disk and compare its
        public inputs to expected inputs and outputs.

        Args:
            witness_path (str): Path to the binary witness file.
            input_path (str): Path to a JSON file containing expected inputs.
            output_path (str): Path to a JSON file containing expected outputs.
                            Only the `"outputs"` field is used for comparison.
            proof_system(ZKProofSystems): Proof system to be used.

        Returns:
            bool:
                True if all witness public inputs match the expected inputs and outputs,
                False otherwise.

        Raises:
            WitnessMatchError:
                If the witness file is malformed or missing the modulus field.
        """
        w = load_witness(witness_path, proof_system)
        expected_inputs = self._read_from_json_safely(input_path)
        expected_outputs = self._read_from_json_safely(output_path)
        if "modulus" not in w:
            msg = "Witness not correctly formed. Missing modulus."
            raise WitnessMatchError(msg)
        return compare_witness_to_io(
            w,
            expected_inputs,
            expected_outputs,
            w["modulus"],
            proof_system,
            self.scale_and_round,
        )

    def contains_float(self: Circuit, obj: float | dict | list) -> bool:
        """Recursively check whether an object contains any float values.

        Args:
            obj (float | dict | list): The object to inspect.
                Can be a float, list, dict.

        Returns:
            bool: True if any float is found within the object
                (including nested lists/dicts), False otherwise.
        """
        if isinstance(obj, float):
            return True
        if isinstance(obj, dict):
            return any(self.contains_float(v) for v in obj.values())
        if isinstance(obj, list):
            return any(self.contains_float(i) for i in obj)
        return False

    def adjust_shape(self: Circuit, shape: list[int] | dict[str, int]) -> list[int]:
        """Normalize a shape representation into a valid list of positive integers.

        Args:
            shape (list[int] | dict[str, int]):
                The shape, which can be a list of ints
                or a dict containing one shape list.

        Raises:
            CircuitInputError:
                If `shape` is a dict containing more than one shape definition.

        Returns:
            list[int]:
                The adjusted shape where all non-positive values are replaced with 1.
        """
        if isinstance(shape, dict):
            # Get the first shape from the dict
            # (assuming only one input is relevant here)
            if len(shape.values()) == 1:
                shape = next(iter(shape.values()))
            else:
                msg = (
                    "Shape dictionary contains multiple entries;"
                    " only one input shape is allowed."
                )
                raise CircuitInputError(
                    msg,
                    parameter="shape",
                    expected="dict with exactly one key-value pair",
                    details={"shape_keys": list(shape.keys())},
                )
        return [s if s > 0 else 1 for s in shape]

    def scale_and_round(
        self: Circuit,
        value: list[int] | np.ndarray | torch.Tensor,
        scale_base: int,
        scale_exponent: int,
    ) -> list[int] | np.ndarray | torch.Tensor:
        """Scale and round numeric values to integers based on
        circuit scaling parameters.

        Args:
            value (list[int] | np.ndarray | torch.Tensor): The values to process.

        Returns:
            list[int] | np.ndarray | torch.Tensor: The scaled and rounded values,
                preserving the original structure.
        """
        scaling = BaseOpQuantizer.get_scaling(
            scale_base=scale_base,
            scale_exponent=scale_exponent,
        )
        if self.contains_float(value):
            return (
                torch.round(
                    torch.tensor(value) * scaling,
                )
                .long()
                .tolist()
            )
        return value

    def adjust_inputs(self: Circuit, input_file: str) -> str:
        """
        Load input values from a JSON file, adjust them by scaling
        and reshaping according to circuit parameters,
        and save the adjusted inputs to a new file.

        Args:
            input_file (str):
                Path to the input JSON file containing the original input values.

        Returns:
            str: Path to the new file containing the adjusted input values.

        Raises:
            CircuitFileError: If reading from or writing to JSON files fails.
            CircuitInputError: If input validation fails
                (e.g., multiple 'input' keys when expecting single).
            CircuitConfigurationError: If required shape attributes are missing.
            CircuitProcessingError: If reshaping or scaling operations fail.
        """
        inputs = self._read_from_json_safely(input_file)

        input_variables = getattr(self, "input_variables", ["input"])
        if input_variables == ["input"]:
            new_inputs = self._adjust_single_input(inputs)
        else:
            new_inputs = self._adjust_multiple_inputs(inputs, input_variables)

        # Save reshaped inputs
        path = Path(input_file)
        new_input_file = path.stem + "_reshaped" + path.suffix
        self._to_json_safely(new_inputs, new_input_file, "adjusted input")
        return new_input_file

    def _adjust_single_input(self: Circuit, inputs: dict) -> dict:
        """
        Adjust inputs when there is a single 'input' variable,
        handling special cases like multiple keys containing 'input'
        or fallback from 'output' to 'input'.

        Args:
            inputs (dict): Dictionary of input values loaded from JSON.

        Returns:
            dict: Adjusted inputs with scaled and reshaped values.

        Raises:
            CircuitInputError:
                If multiple keys containing 'input' are found
                or if required shape attributes are missing.
        """
        new_inputs: dict[str, Any] = {}
        has_input_been_found = False

        for key, value in inputs.items():
            value_adjusted = self.scale_and_round(
                value,
                self.scale_base,
                self.scale_exponent,
            )
            if "input" in key:
                if has_input_been_found:
                    msg = (
                        "Multiple inputs found containing 'input'. "
                        "Only one allowed when input_variables = ['input']"
                    )
                    raise CircuitInputError(
                        msg,
                        parameter="input",
                        expected="single input key",
                        details={"input_keys": [k for k in inputs if "input" in k]},
                    )
                has_input_been_found = True
                value_adjusted = self._reshape_input_value(
                    value_adjusted,
                    "input_shape",
                    key,
                )
                new_inputs["input"] = value_adjusted
            else:
                new_inputs[key] = value_adjusted

        # Special case: fallback mapping output → input
        if "input" not in new_inputs and "output" in new_inputs:
            new_inputs["input"] = inputs["output"]
            del inputs["output"]

        return new_inputs

    def _adjust_multiple_inputs(
        self: Circuit,
        inputs: dict,
        input_variables: list[str],
    ) -> dict:
        """
        Adjust inputs when there are multiple named input variables,
        scaling and reshaping each according to their respective shape attributes.

        Args:
            inputs (dict): Dictionary of input values loaded from JSON.
            input_variables (list[str]): List of input variable names to adjust.

        Returns:
            dict: Adjusted inputs with scaled and reshaped values.

        Raises:
            CircuitConfigurationError:
                If required shape attributes are missing for any input variable.
            CircuitProcessingError: If reshaping operations fail.
        """
        new_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            value_adjusted = self.scale_and_round(
                value,
                self.scale_base,
                self.scale_exponent,
            )
            if key in input_variables:
                shape_attr = f"{key}_shape"
                value_adjusted = self._reshape_input_value(
                    value_adjusted,
                    shape_attr,
                    key,
                )
            new_inputs[key] = value_adjusted
        return new_inputs

    def _reshape_input_value(
        self: Circuit,
        value: list[int] | np.ndarray | torch.Tensor,
        shape_attr: str,
        input_key: str,
    ) -> list[int]:
        """
        Reshape an input value to match the
        specified shape attribute of the circuit.

        Args:
            value (list[int] | np.ndarray | torch.Tensor):
                The input value to reshape, typically a list or tensor-like structure.
            shape_attr (str):
                Name of the attribute containing the target shape (e.g., 'input_shape').
            input_key (str):
                Key of the input being reshaped, used for error messages.

        Returns:
            list[int]: The reshaped value as a list.

        Raises:
            CircuitConfigurationError: If the required shape attribute is not defined.
            CircuitProcessingError: If the reshaping operation fails.
        """
        if not hasattr(self, shape_attr):
            msg = f"Required shape attribute '{shape_attr}'"
            f" must be defined to reshape input '{input_key}'."
            raise CircuitConfigurationError(
                msg,
                missing_attributes=[shape_attr],
                details={"input_key": input_key},
            )

        shape = getattr(self, shape_attr)
        shape = self.adjust_shape(shape)

        try:
            return torch.tensor(value).reshape(shape).tolist()
        except Exception as e:
            msg = f"Failed to reshape input data for '{input_key}'."
            raise CircuitProcessingError(
                msg,
                operation="reshape",
                data_type="tensor",
                details={"shape": shape},
            ) from e

    def _to_json_safely(
        self: Circuit,
        inputs: dict,
        input_file: str,
        var_name: str,
    ) -> None:
        """Safely write data to a JSON file, handling exceptions.

        Args:
            inputs (dict): Data to write.
            input_file (str): Path to the output file.
            var_name (str): Name of the variable for error messages.
        """
        try:
            to_json(inputs, input_file)
        except Exception as e:
            msg = f"Failed to write {var_name} file: {input_file}"
            raise CircuitFileError(
                msg,
                file_path=input_file,
            ) from e

    def _read_from_json_safely(
        self: Circuit,
        input_file: str,
    ) -> dict[str, Any]:
        """Safely read data from a JSON file, handling exceptions.

        Args:
            input_file (str): Path to the input file.

        Returns:
            dict[str, Any]: Data read from the file.
        """
        try:
            return read_from_json(input_file)
        except Exception as e:
            msg = f"Failed to read input file: {input_file}"
            raise CircuitFileError(
                msg,
                file_path=input_file,
            ) from e

    def _gen_witness_preprocessing(  # noqa: PLR0913
        self: Circuit,
        input_file: str,
        output_file: str,
        quantized_path: str,
        *,
        write_json: bool,
        is_scaled: bool,
    ) -> str:
        """Preprocess inputs and outputs before witness generation.

        Args:
            input_file (str): Path to the input JSON file.
            output_file (str): Path to save computed outputs.
            quantized_path (str): Path to quantized model file.
            write_json (bool): Whether to compute new inputs and write to JSON.
            is_scaled (bool): Whether the inputs are already scaled.

        Returns:
            str: Path to the final processed input file.
        """
        # Rescale and reshape
        if quantized_path:
            self.load_quantized_model(quantized_path)
        else:
            self.load_quantized_model(self._file_info.get("quantized_model_path"))

        if write_json:
            inputs = self.get_inputs()
            outputs = self.get_outputs(inputs)

            inputs = self.format_inputs(inputs)

            output = self.format_outputs(outputs)

            self._to_json_safely(inputs, input_file, "input")
            self._to_json_safely(output, output_file, "output")

        else:
            input_file = self.adjust_inputs(input_file)
            inputs = self.get_inputs_from_file(input_file, is_scaled=is_scaled)
            # Compute output (with caching via decorator)
            output = self.get_outputs(inputs)
            outputs = self.format_outputs(output)

            self._to_json_safely(outputs, output_file, "output")
        return input_file

    def _compile_preprocessing(
        self: Circuit,
        weights_path: str,
        quantized_path: str,
    ) -> None:
        """Prepare model weights and quantized files for circuit compilation.

        Args:
            weights_path (str): Path to save model weights in JSON format.
            quantized_path (str): Path to save the quantized model.

        Raises:
            CircuitConfigurationError: If model weights type is unsupported.
        """
        func_model_and_quantize = getattr(self, "get_model_and_quantize", None)
        if callable(func_model_and_quantize):
            func_model_and_quantize()
        if hasattr(self, "flatten"):
            weights = self.get_weights(flatten=True)
        else:
            weights = self.get_weights()

        if quantized_path:
            self.save_quantized_model(quantized_path)

        else:
            self.save_quantized_model(self._file_info.get("quantized_model_path"))

        if isinstance(weights, list):
            for i, w in enumerate(weights):
                if i == 0:
                    self._to_json_safely(w, Path(weights_path), "weights")
                else:
                    val = i + 1
                    file_path = (
                        Path(weights_path).parent
                        / f"{Path(weights_path).stem!s}{val}{Path(weights_path).suffix}"
                    )
                    self._to_json_safely(w, file_path, "weights")
        elif isinstance(weights, (dict, tuple)):
            self._to_json_safely(weights, weights_path, "weights")
        else:
            msg = f"Unsupported weights type: {type(weights)}."
            " Expected list, dict, or tuple."
            raise CircuitConfigurationError(
                msg,
                details={"weights_type": str(type(weights))},
            )

    def save_model(self: Circuit, file_path: str) -> None:
        """
        Save the current model to a file. Should be overridden in subclasses

        Args:
            file_path (str): Path to save the model.
        """

    def load_model(self: Circuit, file_path: str) -> None:
        """
        Load the model from a file. Should be overridden in subclasses

        Args:
            file_path (str): Path to load the model.
        """

    def save_quantized_model(self: Circuit, file_path: str) -> None:
        """
        Save the current quantized model to a file. Should be overridden in subclasses

        Args:
            file_path (str): Path to save the model.
        """

    def load_quantized_model(self: Circuit, file_path: str) -> None:
        """
        Load the quantized model from a file. Should be overridden in subclasses

        Args:
            file_path (str): Path to load the model.
        """

    def get_weights(self: Circuit) -> dict:
        """Retrieve model weights. Should be overridden in subclasses

        Returns:
            dict: Model weights.
        """
        return {}

    def get_inputs_from_file(
        self: Circuit,
        input_file: str,
        *,
        is_scaled: bool = True,
    ) -> dict[str, list[int]]:
        """Load input values from a JSON file, scaling if necessary.

        Args:
            input_file (str): Path to the input JSON file.
            is_scaled (bool, optional): If False, scale inputs
            according to circuit settings. Defaults to True.

        Returns:
            dict[str, list[int]]: Mapping from input names to integer lists of inputs.
        """
        if is_scaled:
            return self._read_from_json_safely(input_file)

        out = {}
        read = self._read_from_json_safely(input_file)

        scaling = BaseOpQuantizer.get_scaling(self.scale_base, self.scale_exponent)
        try:
            for k in read:

                out[k] = torch.as_tensor(read[k]) * scaling
                out[k] = out[k].tolist()
        except Exception as e:
            msg = f"Failed to scale input data for key '{k}'"
            raise CircuitProcessingError(
                msg,
                operation="scale",
                data_type="tensor",
                details={"key": k},
            ) from e
        return out

    def scale_inputs_only(self: Circuit, input_file: str) -> str:
        """
        Load input values from a JSON file, scale them according to circuit parameters,
        without reshaping, and save the scaled inputs to a new file.

        Args:
            input_file (str):
                Path to the input JSON file containing the original input values.

        Returns:
            str: Path to the new file containing the scaled input values.

        Raises:
            CircuitFileError: If reading from or writing to JSON files fails.
        """
        inputs = self._read_from_json_safely(input_file)

        new_inputs = {}
        for key, value in inputs.items():
            new_inputs[key] = self.scale_and_round(
                value,
                self.scale_base,
                self.scale_exponent,
            )

        # Save scaled inputs
        path = Path(input_file)
        new_input_file = path.stem + "_scaled" + path.suffix
        self._to_json_safely(new_inputs, new_input_file, "scaled input")
        return new_input_file

    def rename_inputs(self: Circuit, input_file: str) -> str:
        """
        Load input values from a JSON file, rename keys according to circuit logic
        (similar to adjust_inputs but without scaling or reshaping),
        and save the renamed inputs to a new file.

        Args:
            input_file (str):
                Path to the input JSON file containing the original input values.

        Returns:
            str: Path to the new file containing the renamed input values.

        Raises:
            CircuitFileError: If reading from or writing to JSON files fails.
            CircuitInputError: If input validation fails.
        """
        inputs = self._read_from_json_safely(input_file)

        input_variables = getattr(self, "input_variables", ["input"])
        if input_variables == ["input"]:
            new_inputs = self._rename_single_input(inputs)
        else:
            new_inputs = self._rename_multiple_inputs(inputs, input_variables)

        # Save renamed inputs
        path = Path(input_file)
        new_input_file = path.stem + "_renamed" + path.suffix
        self._to_json_safely(new_inputs, new_input_file, "renamed input")
        return new_input_file

    def _rename_single_input(self: Circuit, inputs: dict) -> dict:
        """
        Rename inputs when there is a single 'input' variable,
        handling special cases like multiple keys containing 'input'
        or fallback from 'output' to 'input'. No scaling or reshaping.

        Args:
            inputs (dict): Dictionary of input values loaded from JSON.

        Returns:
            dict: Renamed inputs with appropriate key mappings.

        Raises:
            CircuitInputError:
                If multiple keys containing 'input' are found.
        """
        new_inputs: dict[str, Any] = {}
        has_input_been_found = False

        for key, value in inputs.items():
            if "input" in key:
                if has_input_been_found:
                    msg = (
                        "Multiple inputs found containing 'input'. "
                        "Only one allowed when input_variables = ['input']"
                    )
                    raise CircuitInputError(
                        msg,
                        parameter="input",
                        expected="single input key",
                        details={"input_keys": [k for k in inputs if "input" in k]},
                    )
                has_input_been_found = True
                new_inputs["input"] = value
            else:
                new_inputs[key] = value

        # Special case: fallback mapping output → input
        if "input" not in new_inputs and "output" in new_inputs:
            new_inputs["input"] = inputs["output"]
            del inputs["output"]

        return new_inputs

    def _rename_multiple_inputs(
        self: Circuit,
        inputs: dict,
    ) -> dict:
        """
        Rename inputs when there are multiple named input variables.
        Keeps keys as is, since no reshaping is done.

        Args:
            inputs (dict): Dictionary of input values loaded from JSON.
            input_variables (list[str]): List of input variable names.

        Returns:
            dict: Renamed inputs (keys unchanged for multiple inputs).
        """
        new_inputs: dict[str, Any] = {}
        for key, value in inputs.items():
            new_inputs[key] = value
        return new_inputs

    def format_outputs(self: Circuit, output: list) -> dict:
        """Format raw model outputs into a standard dictionary format.
        Can be overridden in subclasses

        Args:
            output (list): Raw model output.

        Returns:
            dict: dictionary containing the formatted output under the key 'output'.
        """
        return {"output": output}
