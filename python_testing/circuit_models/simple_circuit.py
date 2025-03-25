import json
import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify
import os
import torch.nn as nn
import numpy as np
from python_testing.circuit_components.circuit_helpers import Circuit, RunType
from random import randint

class SimpleCircuit(Circuit):
    def __init__(self):
        # Initialize the base class
        super().__init__()
        
        # Circuit-specific parameters
        self.layers = {}
        self.name = "simple_circuit"  # Use exact name that matches the binary
        self.scaling = 1
        self.cargo_binary_name = "simple_circuit"
        
        self.input_a = 100#randint(0,10000)
        self.input_b = 200#randint(0,10000)
        #Currently a random value, not sure what value should fit with the validator scheme
        self.nonce = randint(0,10000)

    def get_model_params(self, output):
        """
        Get model parameters for the circuit.
        """
        inputs = {
            "input_a": self.input_a,
            "input_b": self.input_b,
            "nonce": self.nonce
        }
        outputs = {
            "output": output
        }
        return inputs, {}, outputs
    
    def get_outputs(self):
        """
        Compute the output of the circuit.
        This is decorated in the base class to ensure computation happens only once.
        """
        print(f"Performing addition operation: {self.input_a} + {self.input_b}")
        return self.input_a + self.input_b

# Example code demonstrating circuit operations
if __name__ == "__main__":
    # Create a single circuit instance
    print("\n--- Creating circuit instance ---")
    circuit = SimpleCircuit()
    
    print("\n--- Computing output (will happen only once) ---")
    output = circuit.get_outputs()
    print(f"Circuit output: {output}")
    
    print("\n--- Testing different operations ---")
    
    # Run base testing operation
    print("\nRunning base testing:")
    circuit.base_testing(RunType.BASE_TESTING)
    
    # Get the output again (should use cached value)
    print("\nGetting output again (should use cached value):")
    output_again = circuit.get_outputs()
    print(f"Circuit output: {output_again}")
    
    # Run another operation
    print("\nRunning compilation:")
    circuit.base_testing(RunType.COMPILE_CIRCUIT)
    
    # Read the input and output files to verify
    print("\n--- Verifying input and output files ---")
    print(f"Input file: {circuit._file_info['input_file']}")
    print(f"Output file: {circuit._file_info['output_file']}")
    
    print("\nReading input and output files:")
    with open(circuit._file_info['input_file'], 'r') as f:
        input_data = json.load(f)
    
    with open(circuit._file_info['output_file'], 'r') as f:
        output_data = json.load(f)
    
    print(f"Input from file: {input_data}")
    print(f"Output from file: {output_data}")

    # SimpleCircuit().base_testing()
    # SimpleCircuit().base_testing(run_type=RunType.END_TO_END)

    
    # SimpleCircuit().base_testing(run_type=RunType.COMPILE_CIRCUIT)

    # outputs = SimpleCircuit.get_outputs()
    # # (inputs, _, outputs) = SimpleCircuit.get_model_params(SimpleCircuit.get_outputs()) # Create a function that calculates and stores model_params to file
#     SimpleCircuit().base_testing(run_type=RunType.GEN_WITNESS) # This should specify the model_params file


#     SimpleCircuit().base_testing(run_type=RunType.PROVE_WITNESS) # This should specify the model_params file
    
#     SimpleCircuit().base_testing(run_type=RunType.GEN_VERIFY)
