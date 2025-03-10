import json
import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify
import os
from python_testing.circuit_components.relu import ReLU, ConversionType

from python_testing.circuit_components.convolution import Convolution, QuantizedConv
# from python_testing.matrix_multiplication import QuantizedMatrixMultiplication
from python_testing.circuit_components.gemm import QuantizedGemm, Gemm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from python_testing.utils.pytorch_helpers import ZKModel
from python_testing.circuit_components.circuit_helpers import Circuit
    
class SimpleCircuit(Circuit):
    def __init__(self):
        self.layers = {}
        self.name = "simple_circuit"

        self.scaling = 1

        self.input_a = 100
        self.input_b = 200

        



    def get_model_params(self, output):
        inputs = {
            "input_a": self.input_a,
            "input_b": self.input_b
        }
        outputs = {
            "output": output
        }
        return inputs,{},outputs
    
    def get_outputs(self):
        return self.input_a + self.input_b

    

if __name__ == "__main__":

    SimpleCircuit().base_testing()
