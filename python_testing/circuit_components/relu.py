import torch
from python_testing.circuit_components.circuit_helpers import Circuit
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import RunType, get_files, to_json, prove_and_verify
import os
from enum import Enum


class ConversionType(Enum):
    DUAL_MATRIX = "dual_matrix"
    TWOS_COMP = "twos_comp"
    SIGNED_MAG = "signed_mag"

class ReLU(Circuit):
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self, conversion_type = ConversionType.TWOS_COMP):
        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        self.conversion_type = conversion_type
        self.scaling = 2 ** 21
        if conversion_type == ConversionType.DUAL_MATRIX:
            # Specify
            self.name = "relu_dual"
            
            # Function input generation

            self.inputs_1 = torch.randint(low=0, high=100000000, size=(256,))
            self.inputs_2 = torch.randint(low=0, high=2, size=(256,))
            self.outputs = None
        elif conversion_type == ConversionType.TWOS_COMP:
            # Specify
            self.name = "relu_twos_comp"
            
            # Function input generation

            # self.inputs_1 = torch.randint(low=-2**21, high=2**21, size=(1000,))
            self.inputs_1 = torch.randint(low=-2**31, high=2**31, size=(16,4,2))

            self.outputs = None

        elif conversion_type == ConversionType.SIGNED_MAG:
            raise TypeError
            self.name = "relu_twos_comp"
            
            # Function input generation

            self.inputs_1 = torch.randint(low=0, high=100000000, size=(256,))
            self.outputs = None
            

        '''
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################
        '''

    def get_outputs(self):
        return torch.relu(self.inputs_1)
    
    def get_twos_comp_model_data(self, out):
        inputs = {
                'input': self.inputs_1.int().tolist()
                }
        outputs = {
                'output': out.int().tolist(),
            }
        return (inputs, outputs)
    
    def get_inputs(self):
        if self.conversion_type==ConversionType.TWOS_COMP:
            return {'input': self.inputs_1.long()}
        if self.conversion_type==ConversionType.DUAL_MATRIX:
            return {'input': self.inputs_1.long(),'sign': self.inputs_2.int()}
        else:
            raise NotImplementedError("Only twos comp and dual matrix relu is available")
    
    def get_outputs(self, inputs = None):
        """
        Compute the output of the circuit.
        This is decorated in the base class to ensure computation happens only once.
        """
        if inputs == None:
            inputs = self.get_inputs()


        if self.conversion_type==ConversionType.TWOS_COMP:
            return torch.relu(torch.as_tensor(inputs['input']))
        if self.conversion_type==ConversionType.DUAL_MATRIX:
            return torch.mul(torch.as_tensor(inputs['input']), -1*torch.as_tensor(inputs['sign']) + 1)
        else:
            raise NotImplementedError("Only twos comp and dual matrix relu is available")
    
    def format_outputs(self, output):
        return {'output' : output.tolist()}
    
    def format_inputs(self, inputs):
        if self.conversion_type==ConversionType.TWOS_COMP:
            return {'input' :inputs['input'].tolist()}
        if self.conversion_type==ConversionType.DUAL_MATRIX:
            return {'input' : inputs['input'].tolist(), 'sign': inputs['sign'].tolist()}
        else:
            raise NotImplementedError("Only twos comp and dual matrix relu is available")

    
if __name__ == "__main__":
    proof_system = ZKProofSystems.Expander
    proof_folder = "proofs"
    output_folder = "output"
    temp_folder = "temp"
    input_folder = "inputs"
    circuit_folder = ""
    weights_folder = "weights"
    #Rework inputs to function
    conversion_types = [ConversionType.DUAL_MATRIX, ConversionType.TWOS_COMP]
    for conversion_type in conversion_types:
        d = ReLU(conversion_type)
        name = d.name

        d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
        d_2 = ReLU(conversion_type)
        d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
        d_3 = ReLU(conversion_type)
        d_3.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = False)

