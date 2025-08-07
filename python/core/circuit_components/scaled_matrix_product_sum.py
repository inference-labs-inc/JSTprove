# from circom.reward_fn import generate_sample_inputs
import torch
from python.core.circuit_components.circuit_helpers import Circuit
from python.core.utils.run_proofs import ZKProofSystems
from python.core.utils.helper_functions import RunType


class ScaledMatrixProductSum(Circuit):
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self, relu = False):
        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        # Specify
        self.name = "scaled_matrix_product_sum"
        
        # Function input generation

        self.N_ROWS_A: int = 3; # m
        self.N_COLS_A: int = 4; # n
        self.N_ROWS_B: int = 4; # n
        self.N_COLS_B: int = 2; # k
        self.N_ROWS_C: int = 3; # m
        self.N_COLS_C: int = 2; # k
        self.relu = relu

        self.alpha = torch.randint(0, 100, ())
        self.matrix_a = torch.randint(low=0, high=100, size=(self.N_ROWS_A,self.N_COLS_A)) # (m, n) array of random integers between 0 and 100
        self.matrix_b = torch.randint(low=0, high=100, size=(self.N_ROWS_B,self.N_COLS_B)) # (n, k) array of random integers between 0 and 100
        self.matrix_c = torch.randint(low=0, high=100, size=(self.N_ROWS_C,self.N_COLS_C)) # (m, k) array of random integers between 0 and 100
        self.input_variables = ["matrix_a", "matrix_b", "matrix_c"]
        self.scale_base = 1
        self.scaling = 1


        '''
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################
        '''
    @property
    def matrix_a_shape(self):
        return [self.N_ROWS_A, self.N_COLS_A]
    
    @property
    def matrix_b_shape(self):
        return (self.N_ROWS_B,self.N_COLS_B)
    
    @property
    def matrix_c_shape(self):
        return (self.N_ROWS_C,self.N_COLS_C)
    
    def get_inputs(self):
        return {'matrix_a': self.matrix_a, 'matrix_b': self.matrix_b, 'matrix_c': self.matrix_c, 'alpha': self.alpha}
    
    def get_outputs(self, inputs = None):
        """
        Compute the output of the circuit.
        This is decorated in the base class to ensure computation happens only once.
        """
        if inputs == None:
            inputs = {'input_a': self.matrix_a, 'input_b': self.matrix_b, 'matrix_c': self.matrix_c, 'alpha': self.alpha}

        a = torch.as_tensor(inputs['matrix_a'])
        b = torch.as_tensor(inputs['matrix_b'])
        alpha = inputs['alpha']
        c = torch.as_tensor(inputs['matrix_c'])
        out = alpha*torch.matmul(a, b) + c
        return out
    
    def format_outputs(self, output):
        return {'scaled_matrix_product_sum_alpha_ab_plus_c' : output.tolist()}
    
    def format_inputs(self, inputs):
        return {
            'matrix_a': inputs['matrix_a'].tolist(),
            'matrix_b': inputs['matrix_b'].tolist(), 
            'matrix_c': inputs['matrix_c'].tolist(), 
            'alpha': inputs['alpha'].tolist()
            }
    
    
if __name__ == "__main__":
    proof_system = ZKProofSystems.Expander
    proof_folder = "proofs"
    output_folder = "output"
    temp_folder = "temp"
    input_folder = "inputs"
    circuit_folder = ""
    weights_folder = "weights"
    #Rework inputs to function
    d = ScaledMatrixProductSum()
    name = d.name

    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    d_2 = ScaledMatrixProductSum()
    d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
    d_3 = ScaledMatrixProductSum()
    d_3.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = False)

