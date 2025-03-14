# from circom.reward_fn import generate_sample_inputs
import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify
from python_testing.circuit_components.circuit_helpers import Circuit


class MatrixAddition(Circuit):
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self):
        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        # Specify
        self.name = "matrix_addition"
        
        # Function input generation

        N_ROWS_A: int = 17571; # m
        N_COLS_A: int = 1; # n
        N_ROWS_B: int = 17571; # m
        N_COLS_B: int = 1; # n

        self.matrix_a = torch.randint(low=0, high=100, size=(N_ROWS_A,N_COLS_A)) # (m, n) array of random integers between 0 and 100
        self.matrix_b = torch.randint(low=0, high=100, size=(N_ROWS_B,N_COLS_B)) # (m, n) array of random integers between 0 and 100
        
        '''
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################
        '''

    def get_outputs(self):
        return torch.add(self.matrix_a, self.matrix_b)
    
    def get_model_params(self, outputs):
        inputs = {
            'matrix_a': self.matrix_a.tolist(),
            'matrix_b': self.matrix_b.tolist(),          
            }
        
        outputs = {
            'matrix_sum_ab': outputs.tolist(),
        }
        return inputs, {}, outputs
    
if __name__ == "__main__":
    proof_system = ZKProofSystems.Expander
    proof_folder = "proofs"
    output_folder = "output"
    temp_folder = "temp"
    input_folder = "inputs"
    weights_folder = "weights"
    circuit_folder = ""
    #Rework inputs to function
    test_circuit = MatrixAddition()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, weights_folder, circuit_folder, proof_system, output_folder)

