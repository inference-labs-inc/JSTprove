# from circom.reward_fn import generate_sample_inputs
import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify


class MatrixHadamardProduct():
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self):
        super().__init__()
        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        # Specify
        self.name = "matrix_hadamard_product"
        
        # Function input generation

        self.N_ROWS_A: int = 256; # m
        self.N_COLS_A: int = 10; # n
        self.N_ROWS_B: int = 256; # m
        self.N_COLS_B: int = 10; # n

        self.matrix_a = torch.randint(low=0, high=100, size=(self.N_ROWS_A,self.N_COLS_A)) # (m, n) array of random integers between 0 and 100
        self.matrix_b = torch.randint(low=0, high=100, size=(self.N_ROWS_B,self.N_COLS_B)) # (m, n) array of random integers between 0 and 100
        
        '''
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################
        '''

    
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

        matrix_hadamard_ab = torch.mul(self.matrix_a, self.matrix_b)
        ## Define inputs and outputs
        inputs = {
            'matrix_a': self.matrix_a.tolist(),
            'matrix_b': self.matrix_b.tolist(), 
            }
        
        outputs = {
            'matrix_hadamard_ab': matrix_hadamard_ab.tolist(),
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
class QuantizedMatrixHadamard(MatrixHadamardProduct):
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self):
        super().__init__()

        # Instead get a value between 0-1
        self.matrix_a = torch.rand(size=(self.N_ROWS_A,self.N_COLS_A)) - torch.rand(size=(self.N_ROWS_A,self.N_COLS_A))
        self.matrix_b = torch.rand(size=(self.N_ROWS_B,self.N_COLS_B)) - torch.rand(size=(self.N_ROWS_B,self.N_COLS_B))

        self.quantized = True
    
    def get_inputs_for_circuit(self):
        matrix_a = torch.mul(self.matrix_a, 2**self.scaling).long()
        matrix_b = torch.mul(self.matrix_b, 2**self.scaling).long()
        matrix_product_ab = torch.matmul(matrix_a, matrix_b)
        matrix_product_ab = torch.div(matrix_product_ab, 2**self.scaling, rounding_mode="floor").long()

        #This can show the error term with what we are doing
        temp_1 = torch.matmul(self.matrix_a, self.matrix_b)
        temp_1 = torch.mul(temp_1, self.scaling)
        
        print(temp_1[0][0].long(), matrix_product_ab[0][0]/2**21)


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
    test_circuit = MatrixHadamardProduct()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

