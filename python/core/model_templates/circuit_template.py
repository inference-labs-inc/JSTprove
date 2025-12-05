from __future__ import annotations

from secrets import randbelow

from python.core.circuits.base import Circuit


class SimpleCircuit(Circuit):
    """
    Note: This template is irrelevant if using the ONNX circuit builder.
    The template only helps developers if they choose to incorporate other circuit
    builders into the framework.

    To begin, we need to specify some basic attributes surrounding the circuit we will
    be using.

    - `required_keys`: the variables in the input dictionary (and input file).
    - `name`: name of the Rust bin to be run by the circuit.
    - `scale_base`: base of the scaling applied to each value.
    - `scale_exponent`: exponent applied to the base to get the scaling factor.
      Scaling factor will be multiplied by each input.

    Other default inputs can be defined below.
    """

    def __init__(self, file_name: str | None = None) -> None:
        # Initialize the base class
        super().__init__()
        self.file_name = file_name

        # Circuit-specific parameters
        self.required_keys = ["input_a", "input_b", "nonce"]
        self.name = "simple_circuit"  # Use exact name that matches the binary

        self.scale_exponent = 1
        self.scale_base = 1

        self.input_a = 100
        self.input_b = 200
        self.nonce = randbelow(10_000)

    def get_inputs(self) -> dict[str, int]:
        """
        Specify the inputs to the circuit, based on what was specified
        in `__init__`.
        """
        return {
            "input_a": self.input_a,
            "input_b": self.input_b,
            "nonce": self.nonce,
        }

    def get_outputs(self, inputs: dict[str, int] | None = None) -> int:
        """
        Compute the output of the circuit.

        This is overwritten from the base class to ensure computation happens
        only once.
        """
        if inputs is None:
            inputs = {
                "input_a": self.input_a,
                "input_b": self.input_b,
                "nonce": self.nonce,
            }

        return inputs["input_a"] + inputs["input_b"]
