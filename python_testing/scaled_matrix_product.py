# from circom.reward_fn import generate_sample_inputs
import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify


class ScaledMatrixProduct():
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self):
        super().__init__()
        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        # Specify
        self.name = "scaled_matrix_product"
        
        # Function input generation

        N_ROWS_A: int = 3; # m
        N_COLS_A: int = 4; # n
        N_ROWS_B: int = 4; # n
        N_COLS_B: int = 2; # k

        self.alpha = torch.randint(0, 100, ())
        self.matrix_a = torch.randint(low=0, high=100, size=(N_ROWS_A,N_COLS_A)) # (m, n) array of random integers between 0 and 100
        self.matrix_b = torch.randint(low=0, high=100, size=(N_ROWS_B,N_COLS_B)) # (n, k) array of random integers between 0 and 100
        
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

        scaled_matrix_product_alpha_ab = self.alpha*torch.matmul(self.matrix_a, self.matrix_b)

        ## Define inputs and outputs
        inputs = {
            'alpha' : self.alpha.tolist(),
            'matrix_a': self.matrix_a.tolist(),
            'matrix_b': self.matrix_b.tolist(),          
            }
        
        outputs = {
            'scaled_matrix_product_alpha_ab' : scaled_matrix_product_alpha_ab.tolist(),
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
    weights_folder = "weights"
    #Rework inputs to function
    test_circuit = ScaledMatrixProduct()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

