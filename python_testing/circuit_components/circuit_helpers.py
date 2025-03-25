import torch
from python_testing.utils.run_proofs import ZKProofSystems, ZKProofsExpander
from python_testing.utils.helper_functions import (
    get_files, to_json, prove_and_verify, compute_and_store_output, 
    prepare_io_files, compile_circuit, generate_witness, 
    generate_verification, run_end_to_end
)
from enum import Enum

class RunType(Enum):
    BASE_TESTING = 'base_testing'
    END_TO_END = 'end_to_end'
    COMPILE_CIRCUIT = 'compile_circuit'
    GEN_WITNESS = 'gen_witness'
    PROVE_WITNESS = 'prove_witness'
    GEN_VERIFY = 'gen_verify'

class Circuit:
    """Base class for all ZK circuits."""
    
    def __init__(self):
        # Default folder paths - can be overridden in subclasses
        self.input_folder = "inputs"
        self.proof_folder = "analysis"
        self.temp_folder = "temp"
        self.circuit_folder = ""
        self.weights_folder = "weights"
        self.output_folder = "output"
        self.proof_system = ZKProofSystems.Expander
        
        # This will be set by prepare_io_files decorator
        self._file_info = None

    
    @compute_and_store_output
    def get_outputs(self):
        """
        Compute circuit outputs. This method should be implemented by subclasses.
        The decorator will ensure it's only computed once.
        """
        raise NotImplementedError("get_outputs must be implemented")
    
    def get_model_params(self, output):
        """
        Get model parameters. This method should be implemented by subclasses.
        
        Args:
            output: Output computed by get_outputs
        
        Returns:
            Tuple of (inputs, weights, outputs)
        """
        raise NotImplementedError("get_model_params must be implemented")
    
    @prepare_io_files
    def base_testing(self, run_type=RunType.BASE_TESTING, 
                     witness_file=None, input_file=None, proof_path=None, public_path=None, 
                     verification_key=None, circuit_name=None, weights_path=None, output_file=None,
                     proof_system=None):
        """
        Run the circuit with the specified run type.
        All file paths are handled by the decorator.
        
        Args:
            run_type: Type of run to perform
            
        Returns:
            The outputs dictionary
        """
        # Run the appropriate proof operation based on run_type
        self.parse_proof_run_type(
            witness_file, input_file, proof_path, public_path, 
            verification_key, circuit_name, proof_system, output_file, run_type
        )
        
        return self._file_info['outputs']
    
    def parse_proof_run_type(self, witness_file, input_file, proof_path, public_path, 
                             verification_key, circuit_name, proof_system, output_file, run_type):
        """
        Run the appropriate proof operation based on run_type.
        This function can be called directly if needed.
        """
        try:
            if run_type == RunType.BASE_TESTING:
                prove_and_verify(witness_file, input_file, proof_path, public_path, 
                                verification_key, circuit_name, proof_system, output_file)
            elif run_type == RunType.END_TO_END:
                run_end_to_end(circuit_name, input_file, output_file, proof_system)
            elif run_type == RunType.COMPILE_CIRCUIT:
                compile_circuit(circuit_name, proof_system)
            elif run_type == RunType.GEN_WITNESS:
                generate_witness(circuit_name, witness_file, input_file, output_file, proof_system)
            elif run_type == RunType.PROVE_WITNESS:
                generate_witness(circuit_name, witness_file, input_file, output_file, proof_system)
            elif run_type == RunType.GEN_VERIFY:
                generate_verification(circuit_name, proof_system)
            else:
                print(f"Unknown entry: {run_type}")
                raise ValueError(f"Unknown run type: {run_type}")
        except Exception as e:
            print(f"Warning: Operation {run_type} failed: {e}")
            print("Input and output files have still been created correctly.")
    
    # Individual operations that can be called separately
    def compile(self):
        """Compile the circuit."""
        if not self._file_info:
            # Ensure we have file info
            self.base_testing(RunType.COMPILE_CIRCUIT)
            return
        
        compile_circuit(self._file_info['circuit_name'], self._file_info['proof_system'])
    
    def generate_witness(self):
        """Generate witness for the circuit."""
        if not self._file_info:
            # Ensure we have file info
            self.base_testing(RunType.GEN_WITNESS)
            return
        
        generate_witness(
            self._file_info['circuit_name'],
            self._file_info['witness_file'],
            self._file_info['input_file'],
            self._file_info['output_file'],
            self._file_info['proof_system']
        )
    
    def generate_verification(self):
        """Generate verification for the circuit."""
        if not self._file_info:
            # Ensure we have file info
            self.base_testing(RunType.GEN_VERIFY)
            return
        
        generate_verification(self._file_info['circuit_name'], self._file_info['proof_system'])
    
    def run_proof(self):
        """Run proof for the circuit."""
        if not self._file_info:
            # Ensure we have file info
            self.base_testing()
            return
        
        prove_and_verify(
            self._file_info['witness_file'],
            self._file_info['input_file'],
            self._file_info['proof_path'],
            self._file_info['public_path'],
            self._file_info['verification_key'],
            self._file_info['circuit_name'],
            self._file_info['proof_system'],
            self._file_info['output_file']
        )
    
    def run_end_to_end(self):
        """Run end-to-end proof."""
        if not self._file_info:
            # Ensure we have file info
            self.base_testing(RunType.END_TO_END)
            return
        
        run_end_to_end(
            self._file_info['circuit_name'],
            self._file_info['input_file'],
            self._file_info['output_file'],
            self._file_info['proof_system']
        )