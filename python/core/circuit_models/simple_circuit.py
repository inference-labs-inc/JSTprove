from typing import Dict
import torch.nn as nn
from python.core.circuits.base import Circuit, RunType
from random import randint

class SimpleCircuit(Circuit):
    def __init__(self):
        # Initialize the base class
        super().__init__()
        
        # Circuit-specific parameters
        # self.layers = {}
        self.name = "simple_circuit"  # Use exact name that matches the binary
        self.scale_exponent = 1
        self.scale_base = 1
        # self.cargo_binary_name = "simple_circuit"
        
        self.input_a = 100
        self.input_b = 200
        #Currently a random value, not sure what value should fit with the validator scheme
        self.nonce = randint(0,10000)
        self.required_keys = ["value_a", "value_b", "nonce"]
        # self.input_variables = ["value_a", "value_b", "nonce"]

        self.input_shape = [1]

    def get_inputs(self) -> Dict[str, int]:
        """Retrieve the current input values for the circuit.

        Returns:
            Dict[str, int]: A dictionary containing `value_a`, `value_b`, and `nonce`.
        """        
        return {'value_a': self.input_a, 'value_b': self.input_b, 'nonce': self.nonce}
    
    def get_outputs(self, inputs: Dict[str, int] = None) -> int:
        
        """Compute the output of the circuit.

        Args:
            inputs (Dict[str, int], optional): A dictionary containing `value_a`, `value_b`, and `nonce`.
            If None, uses the instance's default inputs. Defaults to None.

        Returns:
            int: output of function
        """        
        if inputs == None:
            inputs = {'value_a': self.input_a, 'value_b': self.input_b, 'nonce': self.nonce}
        print(f"Performing addition operation: {inputs['value_a']} + {inputs['value_b']}")
        return inputs['value_a'] + inputs['value_b']
    
    def format_inputs(self, inputs: Dict[str, int]) -> Dict[str, int]:
        """Format the inputs for the circuit.

        Args:
            inputs (Dict[str, int]): A dictionary containing circuit input values.

        Returns:
            Dict[str, int]: A dictionary containing circuit input values.
        """        
        return inputs
    
    

# Example code demonstrating circuit operations
if __name__ == "__main__":
    # Create a single circuit instance
    print("\n--- Creating circuit instance ---")
    circuit = SimpleCircuit()
    
    print("\n--- Testing different operations ---")
    
    print("\nGetting output again (should use cached value):")
    output_again = circuit.get_outputs()
    print(f"Circuit output: {output_again}")
    
    # Run another operation
    print("\nRunning compilation:")
    circuit.base_testing(run_type = RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path="simple_circuit.txt", input_file="inputs/simple_circuit_input.json", output_file="output/simple_circuit_output.txt")
    
    # Read the input and output files to verify
    print("\n--- Verifying input and output files ---")
    print(f"Input file: {circuit._file_info['input_file']}")
    print(f"Output file: {circuit._file_info['output_file']}")

    circuit.base_testing(run_type = RunType.GEN_WITNESS, circuit_path="simple_circuit.txt", input_file="inputs/simple_circuit_input.json", output_file="output/simple_circuit_output.json", write_json=True)

    circuit = SimpleCircuit()
    circuit.base_testing(run_type=RunType.PROVE_WITNESS, circuit_path="simple_circuit.txt", input_file="inputs/simple_circuit_input.json", output_file="output/simple_circuit_output.json")

    circuit = SimpleCircuit()
    circuit.base_testing(run_type=RunType.GEN_VERIFY, circuit_path="simple_circuit.txt", input_file="inputs/simple_circuit_input.json", output_file="output/simple_circuit_output.json")
