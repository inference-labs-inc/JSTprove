# from circom.reward_fn import generate_sample_inputs
import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify
from enum import Enum
import sys


class MatMultType(Enum):
    Traditional = "traditional"
    Naive = "naive"
    Naive_array = "naive_array"
    Naive1 = "naive1"
    Naive2 = "naive2"
    Naive3 = "naive3"

class MatrixMultiplication():
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self, circuit_type: MatMultType = MatMultType.Naive2):
        super().__init__()
        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        # Specify
        self.name = "matrix_multiplication"

        self.circuit_type = circuit_type
        
        # Function input generation

        self.N_ROWS_A: int = 1; # m
        self.N_COLS_A: int = 1568; # n
        self.N_ROWS_B: int = self.N_COLS_A; # n
        self.N_COLS_B: int = 256; # k

        self.scaling = 21

        self.matrix_a = torch.randint(low=0, high=2**self.scaling, size=(self.N_ROWS_A,self.N_COLS_A)) # (m, n) array of random integers between 0 and 100
        self.matrix_b = torch.randint(low=0, high=2**self.scaling, size=(self.N_ROWS_B,self.N_COLS_B)) # (n, k) array of random integers between 0 and 100

        self.quantized = False
        
        '''
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################
        '''

    def get_inputs_for_circuit(self):
        return (self.matrix_a, self.matrix_b, torch.matmul(self.matrix_a, self.matrix_b))

    
    def base_testing(self, input_folder:str, proof_folder: str, temp_folder: str, weights_folder:str, circuit_folder:str, proof_system: ZKProofSystems, output_folder: str = None):

        # NO NEED TO CHANGE!
        witness_file, input_file, proof_path, public_path, verification_key, circuit_name, weights_file, output_file = get_files(
            input_folder, proof_folder, temp_folder, circuit_folder, weights_folder, self.name, output_folder, proof_system)
        
        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        ## Perform calculation here

        (matrix_a, matrix_b, matrix_product_ab) = self.get_inputs_for_circuit()

        ## Define inputs and outputs
        inputs = {
            'matrix_a': matrix_a.tolist(),
            }
        
        weights = {
            'matrix_b': matrix_b.tolist(), 
            'quantized': self.quantized,
            'scaling': self.scaling,
            'circuit_type': self.circuit_type.value
        }
        
        outputs = {
            'matrix_product_ab': matrix_product_ab.tolist(),
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

        to_json(weights, weights_file)

        ## Run the circuit
        prove_and_verify(witness_file, input_file, proof_path, public_path, verification_key, circuit_name, proof_system, output_file)

class QuantizedMatrixMultiplication(MatrixMultiplication):
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self, circuit_type: MatMultType = MatMultType.Naive2):
        super().__init__(circuit_type)

        # Instead get a value between 0-1
        self.matrix_a = torch.rand(size=(self.N_ROWS_A,self.N_COLS_A))
        self.matrix_b = torch.rand(size=(self.N_ROWS_B,self.N_COLS_B))

        self.quantized = True
    
    def get_inputs_for_circuit(self):
        matrix_a = torch.mul(self.matrix_a, 2**self.scaling).long()
        matrix_b = torch.mul(self.matrix_b, 2**self.scaling).long()
        matrix_product_ab = torch.matmul(matrix_a, matrix_b)
        matrix_product_ab = torch.div(matrix_product_ab, 2**self.scaling, rounding_mode="floor").long()

        #This can show the error term with what we are doing
        temp_1 = torch.matmul(self.matrix_a, self.matrix_b)
        temp_1 = torch.mul(temp_1, self.scaling)
        
        print(temp_1[0][0].long()/2**21, matrix_product_ab[0][0]/2**21)


        return (matrix_a, matrix_b, matrix_product_ab)


if __name__ == "__main__":
    proof_system = ZKProofSystems.Expander
    proof_folder = "proofs"
    output_folder = "output"
    temp_folder = "temp"
    input_folder = "inputs"
    weights_folder = "weights"
    circuit_folder = ""
    #Rework inputs to function
    # test_circuit = MatrixMultiplication(MatMultType.Naive2)
    # test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

    test_circuit = QuantizedMatrixMultiplication(MatMultType.Naive2)
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

