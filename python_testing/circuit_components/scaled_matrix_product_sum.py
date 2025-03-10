# from circom.reward_fn import generate_sample_inputs
import torch
from python_testing.circuit_components.circuit_helpers import Circuit
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify


class ScaledMatrixProductSum(Circuit):
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self):
        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        # Specify
        self.name = "scaled_matrix_product_sum"
        
        # Function input generation

        N_ROWS_A: int = 3; # m
        N_COLS_A: int = 4; # n
        N_ROWS_B: int = 4; # n
        N_COLS_B: int = 2; # k
        N_ROWS_C: int = 3; # m
        N_COLS_C: int = 2; # k

        self.alpha = torch.randint(0, 100, ())
        self.matrix_a = torch.randint(low=0, high=100, size=(N_ROWS_A,N_COLS_A)) # (m, n) array of random integers between 0 and 100
        self.matrix_b = torch.randint(low=0, high=100, size=(N_ROWS_B,N_COLS_B)) # (n, k) array of random integers between 0 and 100
        self.matrix_c = torch.randint(low=0, high=100, size=(N_ROWS_C,N_COLS_C)) # (m, k) array of random integers between 0 and 100

        '''
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################
        '''
    def get_outputs(self):
        return self.alpha * torch.matmul(self.matrix_a, self.matrix_b) + self.matrix_c
    
    def get_model_params(self, output):
        ## Define inputs and outputs
        inputs = {
            'alpha' : self.alpha.tolist(),
            'matrix_a': self.matrix_a.tolist(),
            'matrix_b': self.matrix_b.tolist(),
            'matrix_c': self.matrix_c.tolist(),          
            }
        
        outputs = {
            'scaled_matrix_product_sum_alpha_ab_plus_c' : output.tolist(),
        }
        return inputs, {}, outputs
    
if __name__ == "__main__":
    proof_system = ZKProofSystems.Expander
    proof_folder = "proofs"
    output_folder = "output"
    temp_folder = "temp"
    input_folder = "inputs"
    circuit_folder = ""
    weights_folder = "weights"
    #Rework inputs to function
    test_circuit = ScaledMatrixProductSum()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

