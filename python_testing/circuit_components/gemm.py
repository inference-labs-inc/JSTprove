# import torch
# from python_testing.utils.run_proofs import ZKProofSystems
# from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify
# from python_testing.circuit_components.circuit_helpers import Circuit

# import torch.nn as nn



# class Gemm(Circuit):
#     #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
#     def __init__(self):
#         # super().__init__()
#         '''
#         #######################################################################################################
#         #################################### This is the block for changes ####################################
#         #######################################################################################################
#         '''
#         # Specify
#         self.name = "gemm"
        
#         # Function input generation

#         N_ROWS_A: int = 3; # m
#         N_COLS_A: int = 4; # n
#         N_ROWS_B: int = 4; # n
#         N_COLS_B: int = 2; # k
#         N_ROWS_C: int = 3; # m
#         N_COLS_C: int = 2; # k
        
#         self.quantized = False

#         self.scaling = 21

#         self.alpha = torch.randint(0, 100, ())
#         self.beta = torch.randint(0, 100, ())
#         self.matrix_a = torch.randint(low=0, high=2**self.scaling, size=(N_ROWS_A,N_COLS_A)) # (m, n) array of random integers between 0 and 100
#         self.matrix_b = torch.randint(low=0, high=2**self.scaling, size=(N_ROWS_B,N_COLS_B)) # (n, k) array of random integers between 0 and 100
#         self.matrix_c = torch.randint(low=0, high=2**self.scaling, size=(N_ROWS_C,N_COLS_C)) # (m, k) array of random integers between 0 and 100

#         '''
#         #######################################################################################################
#         #######################################################################################################
#         #######################################################################################################
#         '''
        
#     def get_model_params(self, gemm):
#         inputs = {
#             'input': self.matrix_a.tolist(),  
#             }
        
#         weights = {
#             'alpha' : self.alpha.tolist(),
#             'beta' : self.beta.tolist(),
#             'weights': self.matrix_b.tolist(),
#             'bias': self.matrix_c.tolist(),  
#             'quantized': self.quantized,
#             'scaling': self.scaling
#         }
        
#         outputs = {
#             'output' : gemm.tolist(),
#         }
        
#         return inputs,weights,outputs
    
    

#     def get_outputs(self):
#         gemm = self.matrix_a
#         gemm = self.alpha * torch.matmul(self.matrix_a, self.matrix_b)
#         gemm = gemm + self.beta*self.matrix_c
#         return gemm
    
#     def get_weights(self):
#         return {
#             'alpha' : self.alpha.tolist(),
#             'beta' : self.beta.tolist(),
#             'weights': self.matrix_b.tolist(),
#             'bias': self.matrix_c.tolist(),  
#             'quantized': self.quantized,
#             'scaling': self.scaling
#         }
    
    
#     def get_inputs(self):
#         return {'input': self.matrix_a.tolist()}
    
#     def get_outputs(self, inputs = None):
#         """
#         Compute the output of the circuit.
#         This is decorated in the base class to ensure computation happens only once.
#         """
#         if inputs == None:
#             inputs = {'input_a': self.matrix_a, 'input_b': self.matrix_b}

#         a = torch.as_tensor(inputs['matrix_a'])
#         b = torch.as_tensor(inputs['matrix_b'])
#         out = torch.add(a, b)
#         return out
    
#     def format_outputs(self, output):
#         return {'matrix_sum_ab' : output.tolist()}
    
#     def format_inputs(self, inputs):
#         return {
#             'matrix_a': inputs['matrix_a'].tolist(),
#             'matrix_b': inputs['matrix_b'].tolist(), 
#             }


# class QuantizedGemm(Gemm):
#     #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
#     def __init__(self):
#         super().__init__()

#         self.quantized = True
    
#     def get_outputs(self):
#         gemm = self.alpha * torch.matmul(self.matrix_a, self.matrix_b) #+ self.beta*self.matrix_c
#         gemm = gemm + self.beta*self.matrix_c
#         gemm = torch.div(gemm, 2**self.scaling, rounding_mode="floor").long()
#         return gemm

    
    
    

    
# if __name__ == "__main__":
#     proof_system = ZKProofSystems.Expander
#     proof_folder = "proofs"
#     output_folder = "output"
#     temp_folder = "temp"
#     input_folder = "inputs"
#     weights_folder = "weights"
#     circuit_folder = ""
#     d = MatrixAddition()
#     name = d.name

#     d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
#     d_2 = MatrixAddition()
#     d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
#     d_3 = MatrixAddition()
#     d_3.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = False)


