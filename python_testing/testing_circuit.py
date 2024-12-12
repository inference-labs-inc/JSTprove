from circom.reward_fn import generate_sample_inputs
import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify


class BaseTests():
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self):
        self.name = "testing"
        super().__init__()
        self.scaling = 100000000
        self.inputs_1 = torch.randint(low=0, high=100, size=(256,))
        self.inputs_2 = torch.randint(low=0, high=100, size=(256,))
    
    
    def base_testing(self, input_folder:str, proof_folder: str, temp_folder: str, circuit_folder:str, proof_system: ZKProofSystems, output_folder: str = None):

        # This is the function to run
        witness_file, input_file, proof_path, public_path, verification_key, circuit_name, output_file = get_files(
            input_folder, proof_folder, temp_folder, circuit_folder, self.name, output_folder, proof_system)
        
        ## Perform calculation here

        outputs = torch.add(self.inputs_1,self.inputs_2)

        ## Define inputs and outputs
        inputs = {
            'inputs_1': [int(i) for i in self.inputs_1.tolist()],
            'inputs_2': [int(i) for i in self.inputs_2.tolist()]
            }
        outputs = {
            'outputs': [int(i) for i in outputs.tolist()],
        }

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

