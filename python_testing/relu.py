import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify
import os
from enum import Enum


class ConversionType(Enum):
    DUAL_MATRIX = "dual_matrix"
    TWOS_COMP = "twos_comp"
    SIGNED_MAG = "signed_mag"

class ReLU():
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self, conversion_type):
        super().__init__()
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

    def convert_to_relu_form(self, num_bits = 32):

        def twos_comp_integer(val, bits):
            """compute the 2's complement of int value val"""
            return int(f"{val & ((1 << bits) - 1):0{bits}b}", 2)
        
        def twos_comp(val, bits):
            """compute the 2's complement of int value val"""
            mask = 2**torch.arange(bits - 1, -1, -1).to(val.device, val.dtype)
            return val.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
        
        def bin2dec(b, bits):
            mask = 2 ** torch.arange(bits - 1, -1, -1).to(b.device, b.dtype)
            return torch.sum(mask * b, -1)
        
        def test(inputs):
            def to_binary_2s(x, n_bits):
                res = []
                for i in range(n_bits):
                
                    y = x >> i
                    res.append(y & 1)
                return res
            

            def binary_check(bits):
                for i in range(len(bits)):
                    bin_check = 1 - bits[i]
                    x = bin_check * bits[i]
                    assert(x == 0)

            def from_binary_2s(bits, n_bits):
                res = 0
                length = n_bits - 1
                for i in range(length):
                    coef = 1 << i
                    cur = coef * bits[i]
                    res = res + cur
                binary_check(bits)
                cur = 1 << length * bits[length]
                temp = 0
                out = temp - cur
                return out + res
            
            for i in range(len(inputs)):
                for j in range(len(inputs[i])):
                    for k in range(len(inputs[i][j])):
                        bits = to_binary_2s(inputs[i][j][k], 32)
                        total = from_binary_2s(bits, 32)
                        assert(total, inputs[i][j][k])

                

        

        if self.conversion_type ==ConversionType.DUAL_MATRIX:
            self.inputs_2 = (self.inputs_1 < 0).int()
            self.inputs_1 = torch.mul(torch.abs(self.inputs_1), self.scaling)
        elif self.conversion_type:
            self.inputs_1 = torch.mul(self.inputs_1, self.scaling).int()
            # print(self.inputs_1[0][0][2])
            # self.inputs_1 = twos_comp(self.inputs_1, 32)
            # self.inputs_1 =  torch.tensor([twos_comp(val.item(), num_bits) for val in self.inputs_1.flatten()])
            # print(self.inputs_1[0][0][2])


        else:
            pass

    
    
    def base_testing(self, input_folder:str, proof_folder: str, temp_folder: str, weights_folder:str, circuit_folder:str,  proof_system: ZKProofSystems, output_folder: str = None):

        # NO NEED TO CHANGE!
        witness_file, input_file, proof_path, public_path, verification_key, circuit_name, weights_path, output_file = get_files(
            input_folder, proof_folder, temp_folder, circuit_folder, weights_folder, self.name, output_folder, proof_system)
        

        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        ## Perform calculation here

        # outputs = torch.where(self.inputs_1 > self.inputs_2, torch.tensor(1), 
        #              torch.where(self.inputs_1 == self.inputs_2, torch.tensor(0), torch.tensor(-1)))
        if self.conversion_type == ConversionType.TWOS_COMP:
            if self.outputs == None:
                outputs = torch.relu(self.inputs_1)
            else:
                outputs = torch.mul(self.outputs, self.scaling)
        elif self.conversion_type == ConversionType.DUAL_MATRIX:
            if self.outputs == None:
                inputs_3 = torch.mul(torch.sub(1,torch.mul(self.inputs_2,2)),self.inputs_1)
                outputs = torch.relu(inputs_3)
            else:
                outputs = torch.mul(self.outputs, self.scaling)

        ## Define inputs and outputs
        if self.conversion_type == ConversionType.TWOS_COMP:
            inputs, outputs = self.get_twos_comp_model_data(outputs)
        elif self.conversion_type == ConversionType.DUAL_MATRIX:
            try:
                inputs = {
                    'input': [int(i) for i in self.inputs_1.tolist()],
                    'sign': [int(i) for i in self.inputs_2.tolist()]
                    }
                outputs = {
                    'output': [int(i) for i in outputs.tolist()],
                }
            except:
                inputs = {
                    'input': self.inputs_1.int().tolist(),
                    'sign': self.inputs_2.int().tolist()
                    }
                outputs = {
                    'output': outputs.int().tolist(),
                }
        '''
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################
        '''

        # When needed, can specify model parameters into json as well



        # NO NEED TO CHANGE anything below here!
        to_json(inputs, input_file)
        # Write output to json
        to_json(outputs, output_file)

        ## Run the circuit
        prove_and_verify(witness_file, input_file, proof_path, public_path, verification_key, circuit_name, proof_system, output_file)

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

    
if __name__ == "__main__":
    proof_system = ZKProofSystems.Expander
    proof_folder = "analysis"
    output_folder = "output"
    temp_folder = "temp"
    input_folder = "inputs"
    weights_folder = "weights"
    circuit_folder = ""
    #Rework inputs to function
    test_circuit = ReLU(conversion_type = ConversionType.TWOS_COMP)
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder,  proof_system, output_folder)

