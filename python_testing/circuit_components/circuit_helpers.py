from pathlib import Path
from typing import Optional
import subprocess
import torch
from python_testing.utils.run_proofs import ZKProofSystems, ZKProofsExpander
from python_testing.utils.helper_functions import (
    get_files, read_from_json, to_json, prove_and_verify, compute_and_store_output, 
    prepare_io_files, compile_circuit, generate_witness, 
    generate_verification, run_end_to_end, generate_proof, RunType
)


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
        self.required_keys = None

        # self.check_attributes()

    def check_attributes(self):
        """Check if the necessary attributes are defined in subclasses."""
        if not hasattr(self, 'required_keys') or not hasattr(self, 'name') or not hasattr(self, 'scaling') or not hasattr(self, 'scale_base'):
            raise NotImplementedError("Subclasses must define 'required_keys', 'name', 'scaling' and 'scale_base'.")
    
    def parse_inputs(self, **kwargs):
        if self.required_keys is None:
            raise NotImplementedError("self.required_keys must be specified in circuit definition")
        for key in self.required_keys:
            # print(key)
            if key not in kwargs:
                raise KeyError(f"Missing required parameter: {key}")
            
            value = kwargs[key]
            
            # # Validate type (ensure integer)
            if not isinstance(value, (int, list)):
                raise ValueError(f"Expected an integer for {key}, but got {type(value).__name__}")
            
            setattr(self, key, value)

    
    @compute_and_store_output
    def get_outputs(self):
        """
        Compute circuit outputs. This method should be implemented by subclasses.
        The decorator will ensure it's only computed once.
        """
        raise NotImplementedError("get_outputs must be implemented")
    
    def get_inputs(self, file_path:str = None, is_scaled = False):
        """
        Compute circuit outputs. This method should be implemented by subclasses.
        The decorator will ensure it's only computed once.
        """
        raise NotImplementedError("get_inputs must be implemented")
    
    # def get_model_params(self, output):
    #     """
    #     Get model parameters. This method should be implemented by subclasses.
        
    #     Args:
    #         output: Output computed by get_outputs
        
    #     Returns:
    #         Tuple of (inputs, weights, outputs)
    #     """
    #     raise NotImplementedError("get_model_params must be implemented")

    @prepare_io_files
    def base_testing(self, run_type=RunType.BASE_TESTING, 
                     witness_file=None, input_file=None, proof_file=None, public_path=None, 
                     verification_key=None, circuit_name=None, weights_path=None, output_file=None,
                     proof_system=None,
                     dev_mode = False,
                     ecc = True,
                     circuit_path: Optional[str] = None,
                     write_json: Optional[bool] = False):
        """
        Run the circuit with the specified run type.
        All file paths are handled by the decorator.
        
        Args:
            run_type: Type of run to perform
            ecc: Boolean flag to determine whether to use Expander Compiler Collection or Expander
            
        Returns:
            The outputs dictionary
        """
        if circuit_path is None:
            circuit_path = f"{circuit_name}.txt"

        if not self._file_info:
            raise KeyError("Must make sure to specify _file_info")
        weights_path = self._file_info.get("weights")

        # Run the appropriate proof operation based on run_type
        self.parse_proof_run_type(

            witness_file, input_file, proof_file, public_path, 
            verification_key, circuit_name, circuit_path, proof_system, output_file, weights_path, run_type, dev_mode, ecc, write_json
        )
        
        return 
    
    def parse_proof_run_type(self, witness_file, input_file, proof_path, public_path, 
                             verification_key, circuit_name, circuit_path, proof_system, output_file, weights_path, run_type, dev_mode = False, ecc = True, write_json = False):
        """
        Run the appropriate proof operation based on run_type.
        This function can be called directly if needed.
        """

        is_scaled = True
        
        try:
            if run_type == RunType.BASE_TESTING:
                prove_and_verify(witness_file, input_file, proof_path, public_path, 
                                verification_key, circuit_name, proof_system, output_file, dev_mode, ecc)
            elif run_type == RunType.END_TO_END:
                self._compile_preprocessing(weights_path)
                input_file = self._gen_witness_preprocessing(input_file, output_file, write_json, is_scaled)
                run_end_to_end(circuit_name, circuit_path, input_file, output_file, proof_system, dev_mode)
            elif run_type == RunType.COMPILE_CIRCUIT:
                self._compile_preprocessing(weights_path)
                compile_circuit(circuit_name, circuit_path, proof_system, dev_mode)
            elif run_type == RunType.GEN_WITNESS:
                input_file = self._gen_witness_preprocessing(input_file, output_file, write_json, is_scaled)
                generate_witness(circuit_name, circuit_path, witness_file, input_file, output_file, proof_system, dev_mode)
            elif run_type == RunType.PROVE_WITNESS:
                generate_proof(circuit_name, circuit_path, witness_file, proof_path, proof_system, dev_mode, ecc=ecc)
            elif run_type == RunType.GEN_VERIFY:
                generate_verification(circuit_name, circuit_path, input_file, output_file, witness_file, proof_path, proof_system, dev_mode, ecc=ecc)
            else:
                print(f"Unknown entry: {run_type}")
                raise ValueError(f"Unknown run type: {run_type}")
        except Exception as e:
            print(f"Warning: Operation {run_type} failed: {e}")
            print("Input and output files have still been created correctly.")


    def contains_float(self, obj):
        if isinstance(obj, float):
            return True
        elif isinstance(obj, dict):
            return any(self.contains_float(v) for v in obj.values())
        elif isinstance(obj, list):
            return any(self.contains_float(i) for i in obj)
        return False
    
    def rename_inputs(self, input_file):
        inputs = read_from_json(input_file)
        if "input" in inputs.keys():
            return input_file
        has_input_been_found = False
        new_inputs = {}
        for k in inputs.keys():
            if "input" in k:
                if has_input_been_found:
                    raise ValueError("Multiple inputs found in input file, please change names of the inputs. Only one variable can have 'input' in its name")
                has_input_been_found = True
                new_inputs["input"] = inputs[k]
            else:
                new_inputs[k] = inputs[k]
        
        path = Path(input_file)
        new_input_file = path.with_name(path.stem + "_renamed" + path.suffix)
        to_json(new_inputs, new_input_file)

        return new_input_file
    

    def rescale_inputs(self, input_file):
        """
        Rescale inputs from the input file.
        This function can be called directly if needed.
        """
        inputs = read_from_json(input_file)
        if not self.contains_float(inputs):
            return input_file

        if not hasattr(self, "scale_base") or not hasattr(self, "scaling"):
            raise NotImplementedError("scale_base and scaling must be specified in circuit definition in order to rescale inputs")
        
        for k in inputs.keys():
            inputs[k] = torch.mul(torch.as_tensor(inputs[k]),(self.scale_base**self.scaling)).long().tolist()
        
        path = Path(input_file)
        new_input_file = path.with_name(path.stem + "_reshaped" + path.suffix)
        to_json(inputs, new_input_file)

        return new_input_file
    
    def reshape_inputs(self, input_file):
        """
        Rescale inputs from the input file.
        This function can be called directly if needed.
        """
        inputs = read_from_json(input_file)
        for k in inputs.keys():
            if "input" in k and isinstance(inputs[k], list):
                    if not hasattr(self, "input_shape"):
                        raise NotImplementedError("input_shape must be specified in circuit definition in order to rescale inputs")
                    inputs[k] = torch.as_tensor(inputs[k]).reshape(self.input_shape).tolist()
                    
        path = Path(input_file)
        new_input_file = path.stem + "_reshaped" + path.suffix
        to_json(inputs, new_input_file)

        return new_input_file
    
    def rescale_and_reshape_inputs(self, input_file):
        inputs = read_from_json(input_file)
        for k in inputs.keys():
            # Reshape inputs
            if "input" in k and isinstance(inputs[k], list):
                    if not hasattr(self, "input_shape"):
                        raise NotImplementedError("input_shape must be specified in circuit definition in order to rescale inputs")
                    inputs[k] = torch.as_tensor(inputs[k]).reshape(self.input_shape).tolist()
            # Rescale inputs
            if self.contains_float(inputs[k]):
                inputs[k] = torch.mul(torch.as_tensor(inputs[k]),(self.scale_base**self.scaling)).long().tolist()
                    
        path = Path(input_file)
        new_input_file = path.stem + "_reshaped" + path.suffix
        to_json(inputs, new_input_file)

        return new_input_file
    
    def adjust_inputs(self, input_file):
        inputs = read_from_json(input_file)

        has_input_been_found = False
        new_inputs = {}

        for k in inputs.keys():
            # Reshape inputs
            if "input" in k and isinstance(inputs[k], list):
                    if not hasattr(self, "input_shape"):
                        raise NotImplementedError("input_shape must be specified in circuit definition in order to rescale inputs")
                    inputs[k] = torch.as_tensor(inputs[k]).reshape(self.input_shape).tolist()
            # Rescale inputs
            if self.contains_float(inputs[k]):
                inputs[k] = torch.mul(torch.as_tensor(inputs[k]),(self.scale_base**self.scaling)).long().tolist()
        
            # Rename inputs
            if "input" in k:
                if has_input_been_found:
                    raise ValueError("Multiple inputs found in input file, please change names of the inputs. Only one variable can have 'input' in its name")
                has_input_been_found = True
                new_inputs["input"] = inputs[k]
            else:
                new_inputs[k] = inputs[k]

        if "input" not in new_inputs.keys() and "output" in new_inputs.keys():
            new_inputs["input"] = inputs["output"]
            del inputs["output"]


                    
        path = Path(input_file)
        new_input_file = path.stem + "_reshaped" + path.suffix
        to_json(new_inputs, new_input_file)

        return new_input_file


    def _gen_witness_preprocessing(self, input_file, output_file, write_json, is_scaled):
        # Rescale and reshape

        self.load_quantized_model(self._file_info.get("quantized_model_path"))
        if write_json == True:
            inputs = self.get_inputs()
            output = self.get_outputs(inputs)
            print(inputs)

            input = self.format_inputs(inputs)
            outputs = self.format_outputs(output)

            print("TO JSON")
            to_json(input, input_file)
            to_json(outputs, output_file)
        else:
            # input_file = self.rescale_inputs(input_file)
            # input_file = self.reshape_inputs(input_file)
            # input_file = self.rescale_and_reshape_inputs(input_file)
            # input_file = self.rename_inputs(input_file)
            input_file = self.adjust_inputs(input_file)


            inputs = self.get_inputs_from_file(input_file, is_scaled = is_scaled)
                # Compute output (with caching via decorator)
            output = self.get_outputs(inputs)
            outputs = self.format_outputs(output)
            to_json(outputs, output_file)
        return input_file
    
    def _compile_preprocessing(self, weights_path):
        #### TODO Fix the next couple lines
        func_model_and_quantize = getattr(self, 'get_model_and_quantize', None)
        if callable(func_model_and_quantize):
            func_model_and_quantize()
        if hasattr(self, "flatten"):
            weights = self.get_weights(flatten = True)
        else:
            weights = self.get_weights(flatten = False)

        self.save_quantized_model(self._file_info.get("quantized_model_path"))
        if type(weights) == list:
            for (i, w) in enumerate(weights):
                if i == 0:
                    to_json(w, weights_path)
                else:
                    val = i + 1
                    to_json(w, weights_path[:-5] + f"{val}" + weights_path[-5:])
        elif type(weights) == dict:
            to_json(weights, weights_path)
        else:
            raise NotImplementedError("Weights type is incorrect")

    # # Individual operations that can be called separately
    # def compile(self):
    #     """Compile the circuit."""
    #     if not self._file_info:
    #         # Ensure we have file info
    #         self.base_testing(RunType.COMPILE_CIRCUIT)
    #         return
        
    #     compile_circuit(self._file_info['circuit_name'], self._file_info['proof_system'])

    # def generate_witness(self):
    #     """Generate witness for the circuit."""
    #     if not self._file_info:
    #         # Ensure we have file info
    #         self.base_testing(RunType.GEN_WITNESS)
    #         return
        
    #     generate_witness(
    #         self._file_info['circuit_name'],
    #         self._file_info['witness_file'],
    #         self._file_info['input_file'],
    #         self._file_info['output_file'],
    #         self._file_info['proof_system']
    #     )
    
    # def generate_verification(self):
    #     """Generate verification for the circuit."""
    #     if not self._file_info:
    #         # Ensure we have file info
    #         self.base_testing(RunType.GEN_VERIFY)
    #         return
        
    #     generate_verification(self._file_info['circuit_name'], self._file_info['proof_system'])
    
    # def run_proof(self):
    #     """Run proof for the circuit."""
    #     if not self._file_info:
    #         # Ensure we have file info
    #         self.base_testing()
    #         return
        
    #     prove_and_verify(
    #         self._file_info['witness_file'],
    #         self._file_info['input_file'],
    #         self._file_info['proof_path'],
    #         self._file_info['public_path'],
    #         self._file_info['verification_key'],
    #         self._file_info['circuit_name'],
    #         self._file_info['proof_system'],
    #         self._file_info['output_file']
    #     )
    
    # def run_end_to_end(self):
    #     """Run end-to-end proof."""
    #     if not self._file_info:
    #         # Ensure we have file info
    #         self.base_testing(RunType.END_TO_END)
    #         return
        
    #     run_end_to_end(
    #         self._file_info['circuit_name'],
    #         self._file_info['input_file'],
    #         self._file_info['output_file'],
    #         self._file_info['proof_system']
    #     )
    def save_model(self, file_path: str):
        pass
    
    def load_model(self, file_path: str):
        pass

    def save_quantized_model(self, file_path: str):
        pass

    
    def load_quantized_model(self, file_path: str):
        pass

    def get_weights(self, flatten = False):
        return {}
    
    def get_inputs_from_file(self, input_file, is_scaled = True):
        if is_scaled:
            return read_from_json(input_file)
        
        out = {}
        read = read_from_json(input_file)
        for k in read.keys():
            out[k] = torch.as_tensor(read[k])*(self.scale_base**self.scaling)
            out[k] = out[k].tolist()
        return  out
    
    def format_outputs(self, output):
        return {"output":output}