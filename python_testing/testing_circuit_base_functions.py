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
        self.name = "testing"
        
        # Function input generation

        LENGTH = 10000

        self.inputs_1 = torch.randint(low=0, high=100, size=(LENGTH,))
        self.inputs_2 = torch.randint(low=0, high=100, size=(LENGTH,))
        # self.scaling = 100000000
        '''
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################
        '''

    
    def base_testing(self, input_folder:str, proof_folder: str, temp_folder: str, circuit_folder:str, weights_folder:str,  proof_system: ZKProofSystems, output_folder: str = None):

        # NO NEED TO CHANGE!
        witness_file, input_file, proof_path, public_path, verification_key, circuit_name, _, output_file = get_files(
            input_folder, proof_folder, temp_folder, circuit_folder, weights_folder,self.name, output_folder, proof_system)
        

        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        ## Perform calculation here

        outputs = torch.add(self.inputs_1,self.inputs_2)
        outputs = torch.square(outputs)
        outputs = torch.square(outputs)
        outputs = torch.square(outputs)
        # outputs = torch.square(outputs)


        ## Define inputs and outputs
        inputs = {
            'inputs_1': [int(i) for i in self.inputs_1.tolist()],
            'inputs_2': [int(i) for i in self.inputs_2.tolist()]
            }
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

class Comparison():
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self):
        super().__init__()
        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        # Specify
        self.name = "comparison"
        
        # Function input generation

        self.inputs_1 = torch.randint(low=0, high=2**21, size=(256,))
        self.inputs_2 = torch.randint(low=0, high=1, size=(256,))
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

        # outputs = torch.where(self.inputs_1 > self.inputs_2, torch.tensor(1), 
        #              torch.where(self.inputs_1 == self.inputs_2, torch.tensor(0), torch.tensor(-1)))
        inputs_3 = torch.mul(torch.sub(torch.mul(self.inputs_2,2),1),self.inputs_1)
        outputs = torch.relu(inputs_3)

        ## Define inputs and outputs
        inputs = {
            'inputs_1': [int(i) for i in self.inputs_1.tolist()],
            'inputs_2': [int(i) for i in self.inputs_2.tolist()]
            }
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
class ReLU():
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self):
        super().__init__()
        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        # Specify
        self.name = "relu_dual"
        
        # Function input generation

        self.inputs_1 = torch.randint(low=0, high=100000000, size=(256,))
        self.inputs_2 = torch.randint(low=0, high=2, size=(256,))
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

        # outputs = torch.where(self.inputs_1 > self.inputs_2, torch.tensor(1), 
        #              torch.where(self.inputs_1 == self.inputs_2, torch.tensor(0), torch.tensor(-1)))
        inputs_3 = torch.mul(torch.sub(1, torch.mul(self.inputs_2,2)),self.inputs_1)
        outputs = torch.relu(inputs_3)

        ## Define inputs and outputs
        inputs = {
            'inputs_1': [int(i) for i in self.inputs_1.tolist()],
            'inputs_2': [int(i) for i in self.inputs_2.tolist()]
            }
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
    proof_folder = "analysis"
    output_folder = "output"
    temp_folder = "temp"
    input_folder = "inputs"
    circuit_folder = ""
    weights_folder = "weights"
    #Rework inputs to function
    # test_circuit = Comparison()
    # test_circuit = ReLU()
    test_circuit = BaseTests()
    test_circuit.base_testing(input_folder,proof_folder, temp_folder, circuit_folder, weights_folder, proof_system, output_folder)

