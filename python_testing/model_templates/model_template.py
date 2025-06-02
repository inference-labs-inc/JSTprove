import json
import torch
from python_testing.circuit_models.doom_model import DoomAgent
from python_testing.utils.pytorch_helpers import ZKTorchModel
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify
import os
import torch.nn as nn
import numpy as np
from python_testing.circuit_components.circuit_helpers import Circuit, RunType
from random import randint

class SimpleCircuit(ZKTorchModel):
    '''
    To begin, we need to specify some basic attributes surrounding the model we will be using. 
    required_keys - specify the variables in the input dictionary
    name - name of the rust bin to be run by the circuit
    model_file_name - name of the saved torch model, to be used

    scale_base - specify the base of the scaling applied to each value
    scaling - the exponent applied to the base to get the scaling factor. Scaling factor will be multiplied by each input
    input shape - shape of the input data
    rescale config - by default, each value will be rescaled when an operation is applied to the value that changes the scaling. EG Multiplication. Specify which (if any) values should not be scaled back
    model_type - this is the class of the pytorch model used in inference
    '''
    def __init__(self, file_name):
        self.required_keys = ["input"]
        self.name = "sample_circuit"
        self.model_file_name = file_name


        self.scale_base = 2
        self.scaling = 21
        self.input_shape = [1, 4, 28, 28]
        self.rescale_config = {"fc2": False}
        self.model_type = DoomAgent
    
    '''
    The following are some important functions used by the model. Below are the defaults which are identical to the parent class version (can be deleted here if no changes)
    Changes made would be to specify unique input or output names, or if there are alternative inputs/outputs used. For example a nonce may be used
    '''
    def get_outputs(self, inputs):
        return self.quantized_model(inputs)
    
    def get_inputs(self, file_path:str = None, is_scaled = False):
        if file_path == None:
            return self.create_new_inputs()
        if hasattr(self, "input_shape"):
            return self.get_inputs_from_file(file_path, is_scaled=is_scaled).reshape(self.input_shape)
        else:
            raise NotImplementedError("Must define attribute input_shape")
    
    def format_inputs(self, inputs):
        return {"input": inputs.long().tolist()}
    
    def format_outputs(self, outputs):
        return {"output": outputs.long().tolist()}