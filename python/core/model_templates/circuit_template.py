import torch.nn as nn
from python.core.circuits.base import Circuit
from random import randint

class SimpleCircuit(Circuit):
    '''
    Note: This template is irrelevant if using the onnx circuit builder. 
    The template only helps developers if they choose to incorporate other circuit builders into the framework.

    To begin, we need to specify some basic attributes surrounding the circuit we will be using.
    required_keys - specify the variables in the input dictionary (and input file).
    name - name of the rust bin to be run by the circuit.

    scale_base - specify the base of the scaling applied to each value
    scale_exponent - the exponent applied to the base to get the scaling factor. Scaling factor will be multiplied by each input

    Other default inputs can be defined below
    '''
    def __init__(self, file_name):
        # Initialize the base class
        super().__init__()
        
        # Circuit-specific parameters
        self.required_keys = ["input_a", "input_b", "nonce"]
        self.name = "simple_circuit"  # Use exact name that matches the binary

        self.scale_exponent = 1
        self.scale_base = 1
    
        self.input_a = 100
        self.input_b = 200
        self.nonce = randint(0,10000)
    
    '''
    The following are some important functions used by the model. get inputs should be defined to specify the inputs to the circuit
    '''
    def get_inputs(self):
        '''
        Specify the inputs to the circuit, based on what was specified in the __init__. Can also have inputs to this function for the inputs.
        '''
        return {'input_a': self.input_a, 'input_b': self.input_b, 'nonce': self.nonce}
    
    def get_outputs(self, inputs = None):
        """
        Compute the output of the circuit.
        This is overwritten from the base class to ensure computation happens only once.
        """
        if inputs == None:
            inputs = {'input_a': self.input_a, 'input_b': self.input_b, 'nonce': self.nonce}
        print(f"Performing addition operation: {inputs['input_a']} + {inputs['input_b']}")
        return inputs['input_a'] + inputs['input_b']
    
    # def format_inputs(self, inputs):
    #     return {"input": inputs.long().tolist()}
    
    # def format_outputs(self, outputs):
    #     return {"output": outputs.long().tolist()}
