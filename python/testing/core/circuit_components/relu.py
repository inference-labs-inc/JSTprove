import torch
from python.testing.core.circuit_components.circuit_helpers import Circuit
from python.testing.core.utils.run_proofs import ZKProofSystems
from python.testing.core.utils.helper_functions import RunType
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

        self.scaling = 21
        self.scale_base = 2
        if conversion_type == ConversionType.DUAL_MATRIX:
            # Specify
            self.name = "relu_dual"
            
            # Function input generation
            self.input_shape = (256,)

            self.inputs_1 = torch.randint(low=0, high=100000000, size=self.input_shape)
            self.inputs_2 = torch.randint(low=0, high=2, size=self.input_shape)
            self.outputs = None
        elif conversion_type == ConversionType.TWOS_COMP:
            # Specify
            self.name = "relu_twos_comp"
            
            # Function input generation

            # self.inputs_1 = torch.randint(low=-2**21, high=2**21, size=(1000,))
            self.input_shape = (16,4,2)
            # self.inputs_1 = torch.randint(low=-2**26, high=2**26, size=self.input_shape)

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

    def get_outputs(self, inputs):
        return torch.relu()
    
    # def get_twos_comp_model_data(self, out):
    #     inputs = {
    #             'input': self.inputs_1.int().tolist()
    #             }
    #     outputs = {
    #             'output': out.int().tolist(),
    #         }
    #     return (inputs, outputs)

    def get_inputs(self):
        if self.conversion_type==ConversionType.TWOS_COMP:
            return {'input': torch.randint(low=-2**26, high=2**26, size=self.input_shape).long()}
        elif self.conversion_type==ConversionType.DUAL_MATRIX:
            return {'input': torch.randint(low=0, high=2**26, size=self.input_shape).long(),'sign': torch.randint(low=0, high=2, size=self.input_shape).long()}
        raise NotImplementedError("Only twos comp and dual matrix relu is available")
    
    def format_inputs(self, inputs):
        """
        Format the inputs for the circuit.
        """
        # Convert inputs to a specific format if necessary
        return inputs
    
    # def get_inputs(self, file_path = None):
    #     if file_path == None:
    #         if self.conversion_type==ConversionType.TWOS_COMP:
    #             return {'input': torch.randint(low=-2**26, high=2**26, size=self.input_shape).long()}
    #         elif self.conversion_type==ConversionType.DUAL_MATRIX:
    #             return {'input': torch.randint(low=0, high=2**26, size=self.input_shape).long(),'sign': torch.randint(low=0, high=2, size=self.input_shape).long()}
    #         raise NotImplementedError("Only twos comp and dual matrix relu is available")
    #     if hasattr(self, "input_shape"):
    #         inputs = self.get_inputs_from_file(file_path, is_scaled=True)
    #         for i in inputs.keys():
    #             inputs[i] = torch.as_tensor(i).reshape(self.input_shape)
    #         return i
    #     else:
    #         raise NotImplementedError("Must define attribute input_shape")
    
    def get_outputs(self, inputs = None):
        """
        Compute the output of the circuit.
        This is decorated in the base class to ensure computation happens only once.
        """
        if inputs == None:
            raise "ERROR"

        if self.conversion_type==ConversionType.TWOS_COMP:
            return torch.relu(torch.as_tensor(inputs['input'])).long()
        if self.conversion_type==ConversionType.DUAL_MATRIX:
            return torch.mul(torch.as_tensor(inputs['input']), -1*torch.as_tensor(inputs['sign']) + 1).long()
        else:
            raise NotImplementedError("Only twos comp and dual matrix relu is available")
    
    def format_outputs(self, output):
        return {'output' : output.tolist()}
    
    def format_inputs(self, inputs):
        if self.conversion_type==ConversionType.TWOS_COMP:
            return {'input' :inputs['input'].long().tolist()}
        if self.conversion_type==ConversionType.DUAL_MATRIX:
            return {'input' : inputs['input'].long().tolist(), 'sign': inputs['sign'].long().tolist()}
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

