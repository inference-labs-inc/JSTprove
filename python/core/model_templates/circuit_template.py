import torch.nn as nn
from python.core.circuits.base import Circuit
from random import randint

class SimpleCircuit(Circuit):
    '''
    To begin, we need to specify some basic attributes surrounding the circuit we will be using. If running inference on a pytorch model, should use model_template. If other circuit, this template should suffice
    required_keys - specify the variables in the input dictionary (and input file)
    name - name of the rust bin to be run by the circuit

    scale_base - specify the base of the scaling applied to each value
    scaling - the exponent applied to the base to get the scaling factor. Scaling factor will be multiplied by each input

    Other default inputs can be defined below
    '''
    def __init__(self, file_name):
        # Initialize the base class
        super().__init__()
        
        # Circuit-specific parameters
        self.required_keys = ["input_a", "input_b", "nonce"]
        self.name = "simple_circuit"  # Use exact name that matches the binary

        self.scaling = 1
        self.scale_base = 1
    
        self.input_a = 100
        self.input_b = 200
        self.nonce = randint(0,10000)
    
    '''
    The following are some important functions used by the model. get inputs should be defined to specify the inputs to the circuit
    '''
    def get_inputs(self):
        return {'input_a': self.input_a, 'input_b': self.input_b, 'nonce': self.nonce}
    
    def get_outputs(self, inputs = None):
        """
        Compute the output of the circuit.
        This is decorated in the base class to ensure computation happens only once.
        """
        if inputs == None:
            inputs = {'input_a': self.input_a, 'input_b': self.input_b, 'nonce': self.nonce}
        print(f"Performing addition operation: {inputs['input_a']} + {inputs['input_b']}")
        return inputs['input_a'] + inputs['input_b']
    
    # def format_inputs(self, inputs):
    #     return {"input": inputs.long().tolist()}
    
    # def format_outputs(self, outputs):
    #     return {"output": outputs.long().tolist()}