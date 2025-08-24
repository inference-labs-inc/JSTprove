from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from python.core.utils.helper_functions import ZKProofSystems
from python.core.utils.helper_functions import (
    read_from_json, to_json, compute_and_store_output,
    prepare_io_files, compile_circuit, generate_witness, 
    generate_verification, run_end_to_end, generate_proof, RunType
)


class Circuit:
    """
    Base class for all ZK circuits.

    This class defines the standard interface and common utilities for
    building, testing, and running ZK circuits.
    Subclasses are expected to implement circuit-specific logic such as
    input preparation, output computation, and model handling.
    """
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

    def check_attributes(self):
        """
        Check if the necessary attributes are defined in subclasses.
        Must be overridden in subclasses
        """
        if not hasattr(self, 'required_keys') or not hasattr(self, 'name') or not hasattr(self, 'scale_exponent') or not hasattr(self, 'scale_base'):
            raise NotImplementedError("Subclasses must define 'required_keys', 'name', 'scale_exponent' and 'scale_base'.")
    
    def parse_inputs(self, **kwargs):
        """Parse and validate required input parameters for the circuit into an instance attribute.

        Raises:
            NotImplementedError: If `required_keys` is not set.
            KeyError: If any required parameter is missing.
            ValueError: If any parameter value is not an integer or list of integers.
        """ 
        if self.required_keys is None:
            raise NotImplementedError("self.required_keys must be specified in circuit definition")
        for key in self.required_keys:
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
        """
        raise NotImplementedError("get_outputs must be implemented")
    
    def get_inputs(self, file_path:str = None, is_scaled = False):
        """
        Compute and return the circuit's input values. This method should be implemented by subclasses.
        """
        raise NotImplementedError("get_inputs must be implemented")

    @prepare_io_files
    def base_testing(self, 
                    run_type: RunType = RunType.END_TO_END, 
                    witness_file: str = None,
                    input_file: str = None,
                    proof_file: str = None,
                    public_path: str = None, 
                    verification_key: str = None,
                    circuit_name: str = None,
                    weights_path: str = None,
                    output_file: str = None,
                    proof_system: str = None,
                    dev_mode: str = False,
                    ecc: str = True,
                    circuit_path: Optional[str] = None,
                    write_json: Optional[bool] = False, 
                    bench: bool = False,
                    quantized_path: str = None):
        """Run the circuit in a specified mode (testing, proving, compiling, etc.).

        File path resolution is handled automatically by the `prepare_io_files` decorator.

        Args:
            run_type (RunType, optional): Type of run (compile_circuit, generate_witness, prove, verify). Defaults to RunType.END_TO_END.
            witness_file (str, optional): Path to witness file. Defaults to None.
            input_file (str, optional): Path to input JSON file. Defaults to None.
            proof_file (str, optional): Path to proof file. Defaults to None.
            public_path (str, optional): Path to public inputs file. Defaults to None.
            verification_key (str, optional): Path to verification key file. Defaults to None.
            circuit_name (str, optional): Name of the circuit. Defaults to None.
            weights_path (str, optional): Path to weights file. Defaults to None.
            output_file (str, optional): Path to output JSON file. Defaults to None.
            proof_system (str, optional): Proof system identifier. Defaults to None.
            dev_mode (str, optional):  Enable developer mode. Defaults to False.
            ecc (str, optional): Use ECC version of Expander for prove and verify. Defaults to True.
            circuit_path (Optional[str], optional): Path to compiled circuit file. Defaults to None.
            write_json (Optional[bool], optional): Whether to write inputs directly to JSON. Defaults to False.
            bench (bool, optional): Enable benchmarking mode. Defaults to False.
            quantized_path (str, optional): Path to quantized model file. Defaults to None.

        Raises:
            KeyError: If `_file_info` is not set by the decorator.
        """ 
        if circuit_path is None:
            circuit_path = f"{circuit_name}.txt"

        if not self._file_info:
            raise KeyError("Must make sure to specify _file_info")
        # TODO may need to have a better way to get/store weights path
        weights_path = self._file_info.get("weights")

        # Run the appropriate proof operation based on run_type
        self.parse_proof_run_type(
            witness_file = witness_file,
            input_file = input_file,
            proof_path = proof_file,
            public_path = public_path,
            verification_key = verification_key,
            circuit_name = circuit_name,
            circuit_path = circuit_path,
            proof_system = proof_system,
            output_file = output_file,
            weights_path = weights_path,
            quantized_path = quantized_path,
            run_type = run_type,
            dev_mode = dev_mode,
            ecc = ecc,
            write_json = write_json,
            bench = bench,
        )
        
        return 
    
    def parse_proof_run_type(self,
                            witness_file: str,
                            input_file: str,
                            proof_path: str,
                            public_path: str,
                            verification_key: str,
                            circuit_name: str,
                            circuit_path: str,
                            proof_system: ZKProofSystems,
                            output_file: str,
                            weights_path: str,
                            quantized_path: str,
                            run_type: RunType,
                            dev_mode: bool = False,
                            ecc: bool = True,
                            write_json: bool = False,
                            bench: bool = False):
        """Dispatch proof-related operations based on the selected run type.

        Args:
            witness_file (str): Path to witness file.
            input_file (str): Path to input JSON file.
            proof_path (str): Path to proof file.
            public_path (str): Path to public inputs file.
            verification_key (str): Path to verification key file.
            circuit_name (str): Name of the circuit.
            circuit_path (str): Path to compiled circuit file.
            proof_system (ZKProofSystems): Proof system enum.
            output_file (str): Path to output JSON file.
            weights_path (str): Path to weights file.
            quantized_path (str): Path to quantized model file.
            run_type (RunType): Type of proof run.
            dev_mode (bool, optional): Enable developer mode. Defaults to False.
            ecc (bool, optional): Use ECC mode, for prove/verify. Defaults to True.
            write_json (bool, optional): Write inputs to JSON. Defaults to False.
            bench (bool, optional): Enable benchmarking. Defaults to False.

        Raises:
            ValueError: If `run_type` is unknown.
        """
        is_scaled = True
        
        try:
            if run_type == RunType.END_TO_END:
                self._compile_preprocessing(weights_path = weights_path, quantized_path = quantized_path)
                input_file = self._gen_witness_preprocessing(input_file = input_file, output_file = output_file, quantized_path = quantized_path, write_json = write_json, is_scaled = is_scaled)
                run_end_to_end(circuit_name = circuit_name, circuit_path = circuit_path, input_file = input_file, output_file = output_file, proof_system = proof_system, dev_mode = dev_mode, ecc = ecc)
            elif run_type == RunType.COMPILE_CIRCUIT:
                self._compile_preprocessing(weights_path = weights_path, quantized_path = quantized_path)
                compile_circuit(circuit_name = circuit_name, circuit_path = circuit_path, proof_system = proof_system, dev_mode = dev_mode, bench = bench)
            elif run_type == RunType.GEN_WITNESS:
                input_file = self._gen_witness_preprocessing(input_file = input_file, output_file = output_file, quantized_path = quantized_path, write_json = write_json, is_scaled = is_scaled)
                generate_witness(circuit_name = circuit_name, circuit_path = circuit_path, witness_file = witness_file, input_file = input_file, output_file = output_file, proof_system = proof_system, dev_mode = dev_mode, bench = bench)
            elif run_type == RunType.PROVE_WITNESS:
                generate_proof(circuit_name = circuit_name, circuit_path = circuit_path, witness_file = witness_file, proof_file = proof_path, proof_system = proof_system, dev_mode = dev_mode, ecc = ecc, bench = bench)
            elif run_type == RunType.GEN_VERIFY:
                input_file = self.adjust_inputs(input_file)
                generate_verification(circuit_name = circuit_name, circuit_path = circuit_path, input_file = input_file, output_file = output_file, witness_file = witness_file, proof_file = proof_path, proof_system = proof_system, dev_mode = dev_mode, ecc=ecc, bench = bench)
            else:
                print(f"Unknown entry: {run_type}")
                raise ValueError(f"Unknown run type: {run_type}")
        except Exception as e:
            print(f"Warning: Operation {run_type} failed: {e}")
            print("Input and output files have still been created correctly.")
            # raise e


    def contains_float(self, obj: Any) -> bool:
        """ Recursively check whether an object contains any float values.

        Args:
            obj (Any): The object to inspect. Can be a float, list, dict, or any other type.

        Returns:
            bool: True if any float is found within the object (including nested lists/dicts), False otherwise.
        """
        if isinstance(obj, float):
            return True
        elif isinstance(obj, dict):
            return any(self.contains_float(v) for v in obj.values())
        elif isinstance(obj, list):
            return any(self.contains_float(i) for i in obj)
        return False
    
    def adjust_shape(self, shape: Any) -> List[int]:
        """Normalize a shape representation into a valid list of positive integers.

        Args:
            shape (Any): The shape, which can be a list of ints or a dict containing one shape list.

        Raises:
            ValueError: If `shape` is a dict containing more than one shape definition.

        Returns:
            List[int]: The adjusted shape where all non-positive values are replaced with 1.
        """   
        if isinstance(shape, dict):
                # Get the first shape from the dict (assuming only one input is relevant here)
            if len(shape.values()) == 1:
                shape = next(iter(shape.values()))
            else:
                raise ValueError("Shape inputs in get_inputs_from_file() has too many inputs")
        shape = [s if s > 0 else 1 for s in shape]
        return shape
    
    def scale_and_round(self, value: Any) -> Any:
        """ Scale and round numeric values to integers based on circuit scaling parameters.

        Args:
            value (Any): The values to process.

        Returns:
            Any: The scaled and rounded values, preserving the original structure.
        """
        if self.contains_float(value):
            return torch.round(torch.tensor(value) * (self.scale_base ** self.scale_exponent)).long().tolist()
        return value
    
    def adjust_inputs(self, input_file: str) -> str:
        """Load, scale, reshape, and rewrite circuit input values for compatibility.

        The process involves:
            1. Reads the input JSON file.
            2. Scales and rounds numeric values.
            3. Reshapes them according to predefined shape attributes.
            4. Writes the adjusted inputs to a new JSON file.

        Args:
            input_file (str): Path to the input JSON file.

        Raises:
            ValueError: If multiple 'input' entries are found when only one is allowed.
            NotImplementedError: If required shape attributes are missing.

        Returns:
            str: Path to the newly written adjusted input file.
        """
        # TODO dont write to file, instead handle internally and send straight to rust

        inputs = read_from_json(input_file)
        input_variables = getattr(self, "input_variables", ["input"])
        new_inputs = {}
        # TODO what if multiple inputs
        

        if input_variables == ["input"]:

            has_input_been_found = False
            for k in inputs:
                v = inputs[k]
                v = self.scale_and_round(v)
                if "input" in k:
                    if has_input_been_found:
                        raise ValueError("Multiple inputs found containing 'input'. Only one allowed when input_variables = ['input']")
                    has_input_been_found = True
                    input_shape_attr = "input_shape"
                    if not hasattr(self, input_shape_attr):
                        raise NotImplementedError(f"{input_shape_attr} must be defined to reshape input")
                    
                    shape = getattr(self, input_shape_attr)
                    shape = self.adjust_shape(shape)
                    
                    v = torch.tensor(v).reshape(shape).tolist()

                    new_inputs["input"] = v
                else:
                    new_inputs[k] = v
            if "input" not in new_inputs.keys() and "output" in new_inputs.keys():
                new_inputs["input"] = inputs["output"]
                del inputs["output"]

        else:
            for k in inputs:
                v = inputs[k]
                v = self.scale_and_round(v)
                if k in input_variables:
                    input_shape_attr = f"{k}_shape"
                    if not hasattr(self, input_shape_attr):
                        raise NotImplementedError(f"{input_shape_attr} must be defined to reshape {k}")                    
                    v = torch.tensor(v).reshape(getattr(self, input_shape_attr)).tolist()
                new_inputs[k] = v
        # Save reshaped inputs

        path = Path(input_file)
        new_input_file = path.stem + "_reshaped" + path.suffix
        to_json(new_inputs, new_input_file)

        return new_input_file



    def _gen_witness_preprocessing(self, input_file: str, output_file: str, quantized_path: str, write_json: bool, is_scaled: bool) -> str:
        """Preprocess inputs and outputs before witness generation.

        Args:
            input_file (str): Path to the input JSON file.
            output_file (str): Path to save computed outputs.
            quantized_path (str): Path to quantized model file.
            write_json (bool): Whether to compute new inputs and write to JSON.
            is_scaled (bool): Whether the inputs are already scaled.

        Returns:
            str: Path to the final processed input file.
        """
        # Rescale and reshape
        if quantized_path:
            self.load_quantized_model(quantized_path)
        else:
            self.load_quantized_model(self._file_info.get("quantized_model_path"))
        
        
        if write_json == True:
            inputs = self.get_inputs()
            outputs = self.get_outputs(inputs)
            
            inputs = self.format_inputs(inputs)

            output = self.format_outputs(outputs)

            to_json(inputs, input_file)
            to_json(output, output_file)
        else:
            input_file = self.adjust_inputs(input_file)
            inputs = self.get_inputs_from_file(input_file, is_scaled = is_scaled)
            # Compute output (with caching via decorator)
            output = self.get_outputs(inputs)
            outputs = self.format_outputs(output)
            to_json(outputs, output_file)
        return input_file
    
    def _compile_preprocessing(self, weights_path: str, quantized_path: str):
        """Prepare model weights and quantized files for circuit compilation.

        Args:
            weights_path (str): Path to save model weights in JSON format.
            quantized_path (str): Path to save the quantized model.

        Raises:
            NotImplementedError: If model weights type is unsupported.
        """  
        #### TODO Fix the next couple lines
        func_model_and_quantize = getattr(self, 'get_model_and_quantize', None)
        if callable(func_model_and_quantize):
            func_model_and_quantize()
        if hasattr(self, "flatten"):
            weights = self.get_weights(flatten = True)
        else:
            weights = self.get_weights()

        if quantized_path:
            self.save_quantized_model(quantized_path)

        else:
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
        elif isinstance(weights, tuple):
            to_json(weights, weights_path)
        else:
            raise NotImplementedError("Weights type is incorrect")

    def save_model(self, file_path: str):
        """
        Save the current model to a file. Should be overridden in subclasses

        Args:
            file_path (str): Path to save the model.
        """
        pass
    
    def load_model(self, file_path: str):
        """
        Load the model from a file. Should be overridden in subclasses

        Args:
            file_path (str): Path to load the model.
        """
        pass

    def save_quantized_model(self, file_path: str):
        """
        Save the current quantized model to a file. Should be overridden in subclasses

        Args:
            file_path (str): Path to save the model.
        """
        pass

    
    def load_quantized_model(self, file_path: str):
        """
        Load the quantized model from a file. Should be overridden in subclasses

        Args:
            file_path (str): Path to load the model.
        """
        pass

    def get_weights(self) -> Dict:
        """Retrieve model weights. Should be overridden in subclasses

        Returns:
            Dict: Model weights.
        """ 
        return {}
    
    def get_inputs_from_file(self, input_file: str, is_scaled: bool = True) -> Dict[str, List[int]]:
        """Load input values from a JSON file, scaling if necessary.

        Args:
            input_file (str): Path to the input JSON file.
            is_scaled (bool, optional): If False, scale inputs according to circuit settings. Defaults to True.

        Returns:
            Dict[str, List[int]]: Mapping from input names to integer lists of inputs.
        """        
        if is_scaled:
            return read_from_json(input_file)
        
        out = {}
        read = read_from_json(input_file)
        for k in read.keys():
            out[k] = torch.as_tensor(read[k])*(self.scale_base**self.scale_exponent)
            out[k] = out[k].tolist()
        return  out
    
    def format_outputs(self, output: Any) -> Dict:
        """Format raw model outputs into a standard dictionary format. Can be overridden in subclasses

        Args:
            output (Any): Raw model output.

        Returns:
            Dict: Dictionary containing the formatted output under the key 'output'.
        """
        return {"output":output}
