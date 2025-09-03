from __future__ import annotations

from random import randint

from python.core.circuits.base import Circuit, RunType
from python.core.utils.helper_functions import CircuitExecutionConfig, ZKProofSystems


class SimpleCircuit(Circuit):
    def __init__(self: SimpleCircuit) -> None:
        # Initialize the base class
        super().__init__()

        # Circuit-specific parameters
        self.name = "simple_circuit"  # Use exact name that matches the binary
        self.scale_exponent = 1
        self.scale_base = 1

        self.input_a = 100
        self.input_b = 200
        #############################################################
        ### NOTE This is not a prg suitable for use in-production ###
        #############################################################
        self.nonce = randint(0, 10000000)  # noqa: S311

        self.required_keys = ["value_a", "value_b", "nonce"]

        self.input_shape = [1]

    def get_inputs(self: SimpleCircuit) -> dict[str, int]:
        """Retrieve the current input values for the circuit.

        Returns:
            dict[str, int]: A dictionary containing `value_a`, `value_b`, and `nonce`.
        """
        return {"value_a": self.input_a, "value_b": self.input_b, "nonce": self.nonce}

    def get_outputs(self: SimpleCircuit, inputs: dict[str, int] | None = None) -> int:
        """Compute the output of the circuit.

        Args:
            inputs (dict[str, int], optional):
                A dictionary containing `value_a`, `value_b`, and `nonce`.
                If None, uses the instance's default inputs. Defaults to None.

        Returns:
            int: output of function
        """
        if inputs is None:
            inputs = {
                "value_a": self.input_a,
                "value_b": self.input_b,
                "nonce": self.nonce,
            }
        print(  # noqa: T201
            f"Performing addition operation: {inputs['value_a']} + {inputs['value_b']}",
        )
        return inputs["value_a"] + inputs["value_b"]

    def format_inputs(self: SimpleCircuit, inputs: dict[str, int]) -> dict[str, int]:
        """Format the inputs for the circuit.

        Args:
            inputs (dict[str, int]): A dictionary containing circuit input values.

        Returns:
            dict[str, int]: A dictionary containing circuit input values.
        """
        return inputs


# Example code demonstrating circuit operations
if __name__ == "__main__":
    # Create a single circuit instance
    print("\n--- Creating circuit instance ---")  # noqa: T201
    circuit = SimpleCircuit()

    print("\n--- Testing different operations ---")  # noqa: T201

    print("\nGetting output again (should use cached value):")  # noqa: T201
    output_again = circuit.get_outputs()
    print(f"Circuit output: {output_again}")  # noqa: T201

    # Run another operation
    print("\nRunning compilation:")  # noqa: T201
    circuit.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.COMPILE_CIRCUIT,
            dev_mode=True,
            circuit_path="simple_circuit.txt",
            input_file="inputs/simple_circuit_input.json",
            output_file="output/simple_circuit_output.txt",
            proof_system=ZKProofSystems.Expander,
        ),
    )

    # Read the input and output files to verify
    print("\n--- Verifying input and output files ---")  # noqa: T201
    print(f"Input file: {circuit._file_info['input_file']}")  # noqa: SLF001, T201
    print(f"Output file: {circuit._file_info['output_file']}")  # noqa: SLF001, T201

    circuit.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_WITNESS,
            circuit_path="simple_circuit.txt",
            input_file="inputs/simple_circuit_input.json",
            output_file="output/simple_circuit_output.json",
            write_json=True,
            proof_system=ZKProofSystems.Expander,
        ),
    )

    circuit = SimpleCircuit()
    circuit.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.PROVE_WITNESS,
            circuit_path="simple_circuit.txt",
            input_file="inputs/simple_circuit_input.json",
            output_file="output/simple_circuit_output.json",
            proof_system=ZKProofSystems.Expander,
        ),
    )

    circuit = SimpleCircuit()
    circuit.base_testing(
        CircuitExecutionConfig(
            run_type=RunType.GEN_VERIFY,
            circuit_path="simple_circuit.txt",
            input_file="inputs/simple_circuit_input.json",
            output_file="output/simple_circuit_output.json",
            proof_system=ZKProofSystems.Expander,
        ),
    )
