import torch
from python.testing.python_testing.utils.run_proofs import ZKProofSystems
from python.testing.python_testing.utils.helper_functions import RunType
from python.testing.python_testing.circuit_components.circuit_helpers import Circuit


class Extrema(Circuit):
    # Inputs are a batch of integer vectors; outputs are their maximums
    def __init__(self):
        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        # Specify
        self.name = "extrema"

        # Function input generation
        BATCH_SIZE = 32  # number of test cases
        VEC_LEN = 6      # length of each vector to compute max over

        # Generate a batch of vectors with random nonnegative integers
        self.input_vecs = torch.randint(low=-2**31, high=2**31, size=(BATCH_SIZE, VEC_LEN))

        self.scale_base = 1
        self.scaling = 1
        self.input_shape = [BATCH_SIZE, VEC_LEN]


        '''
        #######################################################################################################
        #######################################################################################################
        #######################################################################################################
        '''
    def get_inputs(self):
        return {"input": self.input_vecs}
    
    def get_outputs(self, inputs):
        return torch.max(torch.as_tensor(inputs["input"]), dim=1).values
    
    def format_outputs(self, output):
        return {"max_val": output.tolist()}
    
    def format_inputs(self, input):
        return {"input": input["input"].tolist()}

if __name__ == "__main__":
    proof_system = ZKProofSystems.Expander
    proof_folder = "proofs"
    output_folder = "output"
    temp_folder = "temp"
    input_folder = "inputs"
    weights_folder = "weights"
    circuit_folder = ""

    test_circuit = Extrema()
    test_circuit.base_testing(
            run_type=RunType.COMPILE_CIRCUIT,
            dev_mode=True,
            input_folder=input_folder,
            proof_folder=proof_folder,
            temp_folder=temp_folder,
            output_folder=output_folder,
            weights_folder=weights_folder,
            circuit_folder=circuit_folder,
            proof_system=proof_system
        )
    test_circuit.base_testing(
        run_type=RunType.GEN_WITNESS,
        dev_mode=True,
        input_folder=input_folder,
        proof_folder=proof_folder,
        temp_folder=temp_folder,
        output_folder=output_folder,
        weights_folder=weights_folder,
        circuit_folder=circuit_folder,
        proof_system=proof_system,
        write_json=True,
    )