import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import RunType, get_files, to_json, prove_and_verify
from python_testing.circuit_components.circuit_helpers import Circuit


class MatMul(Circuit):
    def __init__(self):
        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        # Specify
        self.name = "matmul"
        
        # Function input generation

        self.N_ROWS_A: int = 40; # m
        self.N_COLS_A: int = 40; # n
        self.N_ROWS_B: int = 40; # m
        self.N_COLS_B: int = 40; # n

        self.matrix_a = torch.randint(low=-50, high=50, size=(self.N_ROWS_A,self.N_COLS_A)) # (m, n) array of random integers between -50 and 50
        self.matrix_b = torch.randint(low=-50, high=50, size=(self.N_ROWS_B,self.N_COLS_B)) # (m, n) array of random integers between -50 and 50

        self.input_variables = ["matrix_a", "matrix_b"]
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
    
    def get_inputs(self):
        return {'matrix_a': self.matrix_a, 'matrix_b': self.matrix_b}
    
    def get_outputs(self, inputs = None):
        """
        Compute the output of the circuit.
        This is decorated in the base class to ensure computation happens only once.
        """
        if inputs == None:
            inputs = {'input_a': self.matrix_a, 'input_b': self.matrix_b}

        a = torch.as_tensor(inputs['matrix_a'])
        b = torch.as_tensor(inputs['matrix_b'])
        out = torch.matmul(a, b)
        return out
    
    def format_outputs(self, output):
        return {'matrix_product_ab' : output.tolist()}
    
    def format_inputs(self, inputs):
        return {
            'matrix_a': inputs['matrix_a'].tolist(),
            'matrix_b': inputs['matrix_b'].tolist(), 
            }
    
    
    

    
if __name__ == "__main__":
    proof_system = ZKProofSystems.Expander
    proof_folder = "proofs"
    output_folder = "output"
    temp_folder = "temp"
    input_folder = "inputs"
    weights_folder = "weights"
    circuit_folder = ""
    d = MatMul()
    name = d.name

    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
    d_2 = MatMul()
    d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
    d_3 = MatMul()
    d_3.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = False)