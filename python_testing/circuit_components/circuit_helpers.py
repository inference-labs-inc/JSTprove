import torch
from python_testing.utils.run_proofs import ZKProofSystems, ZKProofsExpander
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify
from enum import Enum

class RunType(Enum):
    BASE_TESTING = 'base_testing'
    END_TO_END = 'end_to_end'
    COMPILE_CIRCUIT = 'compile_circuit'
    GEN_WITNESS = 'gen_witness'
    PROVE_WITNESS = 'prove_witness'
    GEN_VERIFY = 'gen_verify'

class Circuit():
    #Inputs are defined in the __init__ as per the inputs of the function, alternatively, inputs can be generated here
    def __init__(self):
        raise NotImplementedError("__init__ must be implemented for Class Circuit")

    def base_testing(self, input_folder:str = "inputs", proof_folder: str = "analysis", temp_folder: str= "temp", weights_folder:str = "weights", circuit_folder:str = "", proof_system: ZKProofSystems = ZKProofSystems.Expander, output_folder: str = "output", run_type: RunType = RunType.BASE_TESTING, dev_mode = False):

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
        self.parse_proof_run_type(witness_file, input_file, proof_path, public_path, verification_key, circuit_name, proof_system, output_file, run_type, dev_mode)



    def parse_proof_run_type(self, witness_file, input_file, proof_path, public_path, verification_key, circuit_name, proof_system, output_file, run_type, dev_mode = False):
        if run_type == RunType.BASE_TESTING:
            prove_and_verify(witness_file, input_file, proof_path, public_path, verification_key, circuit_name, proof_system, output_file)
        elif run_type == RunType.END_TO_END:
            ZKProofsExpander(circuit_name).run_end_to_end(input_file, output_file, circuit_name, demo = False, dev_mode = dev_mode)
        elif run_type == RunType.COMPILE_CIRCUIT:
            ZKProofsExpander(circuit_name).run_compile_circuit(circuit_name, dev_mode)
        elif run_type == RunType.GEN_WITNESS:
            ZKProofsExpander(circuit_name).run_gen_witness(circuit_name, "", input_file, output_file, dev_mode)
        elif run_type == RunType.PROVE_WITNESS:
            ZKProofsExpander(circuit_name).run_prove_witness(circuit_name, "", dev_mode)
        elif run_type == RunType.GEN_VERIFY:
            ZKProofsExpander(circuit_name).run_gen_verify(circuit_name, dev_mode)

        else:
            print(f"Unknown entry: {run_type}")
            raise
    
    # def run_end_to_end(self, input_folder:str = "inputs", proof_folder: str = "analysis", temp_folder: str= "temp", weights_folder:str = "weights", circuit_folder:str = "", proof_system: ZKProofSystems = ZKProofSystems.Expander, output_folder: str = "output"):

    #     # NO NEED TO CHANGE!
    #     witness_file, input_file, proof_path, public_path, verification_key, circuit_name, weights_file, output_file = get_files(
    #         input_folder, proof_folder, temp_folder, circuit_folder, weights_folder, self.name, output_folder, proof_system)
        

    #     '''
    #     #######################################################################################################
    #     #################################### This is the block for changes ####################################
    #     #######################################################################################################
    #     '''
    #     ## Perform calculation here

    #     outputs = self.get_outputs()

    #     ## Define inputs and outputs
    #     inputs, weights, outputs = self.get_model_params(outputs)
    #     '''
    #     #######################################################################################################
    #     #######################################################################################################
    #     #######################################################################################################
    #     '''

    #     # When needed, can specify model parameters into json as well



    #     # NO NEED TO CHANGE anything below here!
    #     to_json(inputs, input_file)

    #     # Write output to json
    #     to_json(outputs, output_file)

    #     to_json(weights, weights_file)

    #     assert(output_file is not None, "Output_path must be specified")
    #     circuit = ZKProofsExpander(circuit_name)
    #     circuit.run_end_to_end(input_file, output_file, demo = False)

    def get_model_params(self):
        raise NotImplementedError("get_model_params must be implemented")

    def get_outputs(self):
        raise NotImplementedError("get_outputs must be implemented")



