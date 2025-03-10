import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify


class Circuit():
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self):
        raise NotImplementedError("__init__ must be implemented for Class Circuit")

    def base_testing(self, input_folder:str = "inputs", proof_folder: str = "analysis", temp_folder: str= "temp", weights_folder:str = "weights", circuit_folder:str = "", proof_system: ZKProofSystems = ZKProofSystems.Expander, output_folder: str = "output"):

        # NO NEED TO CHANGE!
        witness_file, input_file, proof_path, public_path, verification_key, circuit_name, weights_file, output_file = get_files(
            input_folder, proof_folder, temp_folder, circuit_folder, weights_folder, self.name, output_folder, proof_system)
        

        '''
        #######################################################################################################
        #################################### This is the block for changes ####################################
        #######################################################################################################
        '''
        ## Perform calculation here

        outputs = self.get_outputs()

        ## Define inputs and outputs
        inputs, weights, outputs = self.get_model_params(outputs)
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

        to_json(weights, weights_file)

        ## Run the circuit
        prove_and_verify(witness_file, input_file, proof_path, public_path, verification_key, circuit_name, proof_system, output_file)

    def get_model_params(self):
        raise NotImplementedError("get_model_params must be implemented")

    def get_outputs(self):
        raise NotImplementedError("get_outputs must be implemented")



