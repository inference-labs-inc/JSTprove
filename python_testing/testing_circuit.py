import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify


class BaseTests():
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self):
        super().__init__()
        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        # Specify
        self.name = "gemm"
        
        # Function input generation

        '''
        Matrix a has shape (m, n)
        Matrix b has shape (n, k)
        matmul(a,b) has shape (m, k)
        '''

        N_ROWS_a: int = 16; # m
        N_COLS_a: int = 16; # n
        N_COLS_b: int = 1; # k

        self.input_a = torch.randint(low=0, high=100, size=(N_ROWS_a,N_COLS_a)) # (m, n) array of random integers between 0 and 100
        self.input_b = torch.randint(low=0, high=100, size=(N_COLS_a,N_COLS_b)) # (n, k) array of random integers between 0 and 100
        
        '''
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################
        '''

    
    def base_testing(self, input_folder:str, proof_folder: str, temp_folder: str, circuit_folder:str, proof_system: ZKProofSystems, output_folder: str = None):

        # NO NEED TO CHANGE!
        witness_file, input_file, proof_path, public_path, verification_key, circuit_name, output_file = get_files(
            input_folder, proof_folder, temp_folder, circuit_folder, self.name, output_folder, proof_system)
        

        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        ## Perform calculation here

        vanilla_matrix_product = torch.matmul(self.input_a, self.input_b)

        ## Define inputs and outputs
        inputs = {
            'input_a': self.input_a.tolist(),
            'input_b': self.input_b.tolist(),          
            }
        
        outputs = {
            'matrix_product': vanilla_matrix_product.tolist(),
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

    
if __name__ == "__main__":
    proof_system = ZKProofSystems.Expander
    proof_folder = "proofs"
    output_folder = "output"
    temp_folder = "temp"
    input_folder = "inputs"
    circuit_folder = ""
    #Rework inputs to function
    test_circuit = BaseTests()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, circuit_folder, proof_system, output_folder)

