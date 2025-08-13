import math
import numpy as np
import torch
# from core.utils.pytorch_helpers import Layer, ZKTorchModel, RunType, filter_dict_for_dataclass
from python.core.model_processing.converters.onnx_converter import ONNXOpQuantizer, ONNXConverter
from python.core.circuits.zk_model_base import ZKModelBase
from python.core.utils.helper_functions import RunType

class GenericModelONNX(ONNXConverter, ZKModelBase):
    def __init__(self, model_name, models_folder = None, model_file_path: str = None, quantized_model_file_path: str = None, layers = None):
        self.max_value = 2**32
        self.name = "onnx_generic_circuit"
        self.op_quantizer = ONNXOpQuantizer()
        self.rescale_config = {} 
        # self.rescale_config = {"/conv1/Conv": False} 
        self.model_file_name = self.find_model(model_name)
        self.model_params = {}
        # Temp hardcoded
        self.scale_base = 2
        self.scaling = 18

    def find_model(self, model_name):
        # Look for model in models_for_circuit_folder
        if not ".onnx" in model_name:
            model_name = model_name + ".onnx"
        if "models_onnx" in model_name:
            return model_name
        return f"models_onnx/{model_name}"

    def determine_scaling(self, scaling, scale_base, max_value):
        # Determine scaling based on the max_value we want to be present in the circuit. we also need the model and dummy inputs to determine this. This will require analysis before a future implementation.
        if scaling != -1:
            return scaling
        # return int(18 / math.log2(scale_base))
        raise ValueError("Scaling not specified")
    
    def adjust_inputs(self, input_file):
        input_shape = self.input_shape.copy()
        shape = self.adjust_shape(input_shape)
        self.input_shape = [math.prod(shape)]
        x = super().adjust_inputs(input_file)
        self.input_shape = input_shape.copy()
        return x
    
    def get_outputs(self, inputs):
        return torch.as_tensor(np.array(super().get_outputs(inputs))).flatten()
    
    def format_inputs(self, inputs):
        x = {"input": inputs}
        for key in x:
            x[key] = torch.as_tensor(x[key]).flatten().tolist()
            x[key] = (torch.as_tensor(x[key]) * self.scale_base**self.scaling).long().tolist()
        return x

if __name__ == "__main__":
    names = ["doom"]
    for name in names:
        d = GenericModelONNX(name)
        d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
        d_2 = GenericModelONNX(name)
        d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)


