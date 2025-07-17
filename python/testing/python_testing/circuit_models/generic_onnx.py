import math
import numpy as np
import torch
# from python_testing.utils.pytorch_helpers import Layer, ZKTorchModel, RunType, filter_dict_for_dataclass
from python.testing.python_testing.utils.onnx_converter import ZKONNXModel, ONNXOpQuantizer
from python.testing.python_testing.utils.helper_functions import RunType


class GenericModelONNX(ZKONNXModel):
    def __init__(self, model_name, models_folder = None, model_file_path: str = None, quantized_model_file_path: str = None, layers = None):
        self.max_value = 2**32
        self.name = "onnx_generic_circuit"
        self.op_quantizer = ONNXOpQuantizer()
        self.rescale_config = {} 


        # self.rescale_config = {"/conv1/Conv": False} 

        # self.path = self.find_model(model_name)
        # self.model_type = self.get_model_type(self.path)
        self.model_file_name = self.find_model(model_name)


        self.model_params = {}

        # (required_keys, input_shape, scaling, scale_base, rescale_config) = self.load_metadata(self.model_file_name)

        # Now computed in onnx_converter.load_model()
        # self.required_keys = required_keys

        # self.scale_base = scale_base if scale_base != -1 else 2
        # self.scaling = self.determine_scaling(scaling, self.scale_base, self.max_value)

        # Temp hardcoded

        self.scale_base = 2
        self.scaling = 18

        #  Temp empty, can choose which layers to rescale for optimization purposes



    def find_model(self, model_name):
        # Look for model in models_for_circuit_folder
        if not ".onnx" in model_name:
            model_name = model_name + ".onnx"
        if "models_onnx" in model_name:
            return model_name
        return f"models_onnx/{model_name}"

    def determine_scaling(self, scaling, scale_base, max_value):
        # Determine scaling based on the max_value we want to be present in the circuit. we also need the model and dummy inputs to determine this. This is not a trivial task, and something we can think more about
        if scaling != -1:
            return scaling
        
        return int(18 / math.log2(scale_base))
    
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
        # x = super().format_inputs(inputs)
        x = {"input": inputs}
        for key in x:
            x[key] = torch.as_tensor(x[key]).flatten().tolist()
            # print(x[key])
            # TODO this is not a good long term fix. ONLY WORKS FOR WHEN CREATING INPUTS
            # print(self.scale_base**self.scaling)
            x[key] = (torch.as_tensor(x[key]) * self.scale_base**self.scaling).long().tolist()
            # print(x[key])
        # x = super().format_inputs(inputs)
        return x




# Must specify the model name
# Look into the models_onnx folder for the name.onnx file

# Create mechanism to load all of this into existing code framework
    

if __name__ == "__main__":
    # names = ["demo_1", "demo_2", "demo_3", "demo_4", "demo_5"]
    names = ["demo"]
    for n in names:
        # name = f"{n}_conv1"
        name = "test_doom_cut"
        d = GenericModelONNX(name)
        # # # d.base_testing()
        # # # d.base_testing(run_type=RunType.END_TO_END, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
        d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
        # # d.save_quantized_model("quantized_model.pth")
        d_2 = GenericModelONNX(name)
        # # # # d_2.load_quantized_model("quantized_model.pth")
        d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
        # d_2.base_testing(run_type=RunType.PROVE_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")
        # d_2.base_testing(run_type=RunType.GEN_VERIFY, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")


