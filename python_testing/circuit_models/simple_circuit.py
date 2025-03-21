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
from python_testing.circuit_components.circuit_helpers import Circuit, RunType

from random import randint
    
class SimpleCircuit(Circuit):
    def __init__(self):
        self.layers = {}
        self.name = "simple_circuit"

        self.scaling = 1

        self.input_a = 100#randint(0,10000)
        self.input_b = 200#randint(0,10000)
        #Currently a random value, not sure what value should fit with the validator scheme
        self.nonce = randint(0,10000)
        # print(type(self.nonce))

    def get_model_params(self, output):
        inputs = {
            "input_a": self.input_a,
            "input_b": self.input_b,
            "nonce": self.nonce
        }
        outputs = {
            "output": output
        }
        return inputs,{},outputs
    
    def get_outputs(self):
        return self.input_a + self.input_b #+ self.nonce

    

if __name__ == "__main__":

    # SimpleCircuit().base_testing()
    # SimpleCircuit().base_testing(run_type=RunType.END_TO_END)

    
    # SimpleCircuit().base_testing(run_type=RunType.COMPILE_CIRCUIT)

    # outputs = SimpleCircuit.get_outputs()
    # # (inputs, _, outputs) = SimpleCircuit.get_model_params(SimpleCircuit.get_outputs()) # Create a function that calculates and stores model_params to file
    SimpleCircuit().base_testing(run_type=RunType.GEN_WITNESS) # This should specify the model_params file


    # SimpleCircuit().base_testing(run_type=RunType.PROVE_WITNESS) # This should specify the model_params file
    
    SimpleCircuit().base_testing(run_type=RunType.GEN_VERIFY)




