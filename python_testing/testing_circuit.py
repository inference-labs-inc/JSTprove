from circom.reward_fn import generate_sample_inputs
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

        self.input_a = torch.randint(low=0, high=100, size=(256,256))
        self.input_b = torch.randint(low=0, high=100, size=(256,256))
        self.input_c = torch.randint(low=0, high=100, size=(256,1))
        self.input_alpha = torch.randint(low=0, high=100, size=(1,)).item()
        self.input_beta = torch.randint(low=0, high=100, size=(1,)).item()
        # self.scaling = 100000000
        
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

        def _gemm11(a, b, c, alpha, beta):  # type: ignore
            o = np.dot(a.T, b.T) * alpha
            if c is not None and beta != 0:
                o += c * beta
            return o

        c_matrix = self.input_c.numpy() if self.input_c else None

        outputs = _gemm11(self.input_a.numpy(), 
                          self.input_b.numpy(), 
                          c_matrix, 
                          self.input_alpha, 
                          self.input_beta) 

        ## Define inputs and outputs
        inputs = {
            'input_a': [int(i) for i in self.input_a.tolist()],
            'input_b': [int(i) for i in self.input_b.tolist()],          
            'input_alpha': [int(i) for i in self.input_alpha.tolist()],
            'input_beta': [int(i) for i in self.input_beta.tolist()],
            }
        if c is not None:
            inputs['input_c'] = [int(i) for i in self.input_c.tolist()],
        outputs = {
            'outputs': [int(i) for i in outputs.tolist()],
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

