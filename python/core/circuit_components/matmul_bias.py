import torch
from python.core.utils.run_proofs import ZKProofSystems
from python.core.utils.helper_functions import RunType
from python.core.circuit_components.circuit_helpers import Circuit


class MatMulBias(Circuit):
    def __init__(self):
        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        # Specify
        self.name = "matmul_bias"
        
        # Matrix dimensions
        self.N_ROWS_A: int = 30  # ℓ
        self.N_COLS_A: int = 30  # m
        self.N_ROWS_B: int = 30  # m
        self.N_COLS_B: int = 30  # n

        self.matrix_a = torch.randint(low=-50, high=50, size=(self.N_ROWS_A, self.N_COLS_A))  # (ℓ, m)
        self.matrix_b = torch.randint(low=-50, high=50, size=(self.N_ROWS_B, self.N_COLS_B))  # (m, n)
        self.matrix_c = torch.randint(low=-50, high=50, size=(self.N_ROWS_A, self.N_COLS_B))  # (ℓ, n) bias matrix
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
        return (self.N_ROWS_A,self.N_COLS_B)
    
    def get_inputs(self):
        return {
            'matrix_a': self.matrix_a,
            'matrix_b': self.matrix_b,
            'matrix_c': self.matrix_c,
        }
    
    def get_outputs(self, inputs = None):
        """
        Compute the output of the circuit.
        This is decorated in the base class to ensure computation happens only once.
        """
        if inputs is None:
            inputs = {
                'matrix_a': self.matrix_a,
                'matrix_b': self.matrix_b,
                'matrix_c': self.matrix_c,
            }

        a = torch.as_tensor(inputs['matrix_a'])
        b = torch.as_tensor(inputs['matrix_b'])
        c = torch.as_tensor(inputs['matrix_c'])
        out = torch.matmul(a, b) + c
        return out
    
    def format_outputs(self, output):
        return {'matrix_product_ab_plus_c': output.tolist()}
    
    def format_inputs(self, inputs):
        return {
            'matrix_a': inputs['matrix_a'].tolist(),
            'matrix_b': inputs['matrix_b'].tolist(),
            'matrix_c': inputs['matrix_c'].tolist(),
        }

if __name__ == "__main__":
    proof_system = ZKProofSystems.Expander
    name = "matmul_bias"

    d = MatMulBias()
    d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")

    d_2 = MatMulBias()
    d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json=True)

    d_3 = MatMulBias()
    d_3.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json=False)
