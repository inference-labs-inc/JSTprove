import json
from typing import Any, Dict, List, Tuple
import torch
class GeneralLayerFunctions():
    """
    A collection of utility functions for reading, generating, scaling, and 
    formatting model inputs/outputs. This is primarily intended for 
    preparing inputs for ONNX models or similar layer-based models.
    """
    def read_input(self, file_name: str) -> Any:
        """Read model input data from a JSON file.

        Args:
            file_name (str): Path to the JSON file containing input data.

        Returns:
            Any: The value of the "input" field from the JSON file.
        """        
        with open(file_name, 'r') as file:
            data = json.load(file)
            return data["input"]
        
    def get_inputs_from_file(self, file_name: str, is_scaled: bool = False) -> torch.Tensor:
        """Load and optionally scale inputs from a file.

        Args:
            file_name (str): Path to the file containing input data.
            is_scaled (bool, optional): If True, returns unscaled values. If False, applies scaling using 
                                        `self.scale_base ** self.scale_exponent`. Defaults to False.

        Returns:
            torch.Tensor: The loaded, reshaped, and potentially rescaled input tensor.
        """
        inputs = self.read_input(file_name)
        if is_scaled:
            out =  torch.as_tensor(inputs).long()
        else:
            out =  torch.mul(torch.as_tensor(inputs),self.scale_base**self.scale_exponent).long()


        if hasattr(self, "input_shape"):
            shape = self.input_shape
            if hasattr(self, 'adjust_shape') and callable(getattr(self, 'adjust_shape')):
                shape = self.adjust_shape(shape)

            out = out.reshape(shape)
        return out
    
    def get_inputs(self, file_path: str = None, is_scaled: bool = False) -> torch.Tensor:
        """Retrieve model inputs, either from a file or by generating new inputs.

        Args:
            file_path (str, optional): Path to the input file. If None, 
                                       new random inputs are generated. Defaults to None.
            is_scaled (bool, optional): Whether to skip scaling of loaded inputs. Defaults to False.

        Raises:
            NotImplementedError: If `self.input_shape` is not defined.

        Returns:
            torch.Tensor: The input tensor shaped according to `self.input_shape`.
        """
        if file_path == None:
            return self.create_new_inputs()
        if hasattr(self, "input_shape"):
            return self.get_inputs_from_file(file_path, is_scaled=is_scaled).reshape(self.input_shape)
        else:
            raise NotImplementedError("Must define attribute input_shape")
    
    def create_new_inputs(self) :
        """Generate new random input tensors.

        Returns:
            __type__:
                - If `self.input_shape` is a list/tuple, returns a single tensor.
                - If `self.input_shape` is a dict, returns a dictionary mapping 
                  input names to tensors.
        """
        # ONNX inputs will be in this form, and require inputs to not be scaled up
        if isinstance(self.input_shape, dict):
            keys = self.input_shape.keys()
            if len(keys) == 1:
                # If unknown dim in batch spot, assume batch size of 1
                input_shape = self.input_shape[list(keys)[0]]
                input_shape[0] = 1 if input_shape[0] < 1 else input_shape[0]
                return self.get_rand_inputs(input_shape)
            inputs = {}
            for key in keys:
                # If unknown dim in batch spot, assume batch size of 1
                input_shape = self.input_shape[keys[key]]
                input_shape[0] = 1 if input_shape[0] < 1 else input_shape[0]
                inputs[key] = self.get_rand_inputs(input_shape)
            return inputs
        
        return torch.mul(self.get_rand_inputs(self.input_shape), self.scale_base**self.scale_exponent).long()
    
    def get_rand_inputs(self, input_shape: List[int]) -> torch.Tensor:
        """Generate random input values in the range [-1, 1).

        Args:
            input_shape (List[int]): Shape of the tensor to generate.

        Returns:
            torch.Tensor: A tensor of random values in [-1, 1).
        """
        return torch.rand(input_shape)*2 - 1

    def format_inputs(self, inputs: torch.Tensor) -> Dict[str, List[int]]:
        """Format input tensors for JSON serialization.

        Args:
            inputs (torch.Tensor): The input tensor.

        Returns:
            Dict[str, List[int]]: A dictionary with the key "input" containing the tensor as a list of integers.
        """
        return {"input": inputs.long().tolist()}
    
    def format_outputs(self, outputs: torch.Tensor) -> Dict[str, List[int]]:
        """Format output tensors for JSON serialization, including rescaled outputs for readability.

        Args:
            outputs (torch.Tensor): _deThe output tensor.cription_

        Returns:
            Dict[str, List[int]]: A dictionary containing:
                  - "output": the raw output tensor as a list of integers.
                  - "rescaled_output": the output divided by the scaling factor.
        """       
        if hasattr(self, "scale_exponent") and hasattr(self, "scale_base"):
            return {"output": outputs.long().tolist(), "rescaled_output": torch.div(outputs, self.scale_base**(self.scale_exponent)).tolist()}
        return {"output": outputs.long().tolist()}
    
    def format_inputs_outputs(self, inputs: torch.Tensor, outputs: torch.Tensor) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
        """Format both inputs and outputs for JSON serialization.

        Args:
            inputs (torch.Tensor): Model inputs.
            outputs (torch.Tensor): Model outputs.

        Returns:
            Tuple[Dict[str, List[int]], Dict[str, List[int]]]: A tuple containing the formatted inputs and formatted outputs.
        """
        return self.format_inputs(inputs), self.format_outputs(outputs)
