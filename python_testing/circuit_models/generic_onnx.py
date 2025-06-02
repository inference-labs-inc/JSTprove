import importlib
import json
import math
import os
import torch
import inspect
from python_testing.utils.pytorch_helpers import Layer, ZKTorchModel, RunType, filter_dict_for_dataclass

import sys


from dataclasses import asdict, dataclass, field, fields
from typing import List, Dict, Optional




class GenericModelONNX(ZKTorchModel):
    def __init__(self, model_name, models_folder = None, model_file_path: str = None, quantized_model_file_path: str = None, layers = None):
        self.max_value = 2**32
        self.name = "onnx_generic_circuit"

        self.rescale_config = {}

        # self.path = self.find_model(model_name)
        # self.model_type = self.get_model_type(self.path)
        self.model_file_name = self.find_model(model_name)


        self.model_params = {}

        (required_keys, input_shape, scaling, scale_base, rescale_config) = self.load_metadata(self.path)

        self.required_keys = required_keys

        self.input_shape = ""

        self.scale_base = scale_base if scale_base != -1 else 2
        self.scaling = self.determine_scaling(scaling, self.scale_base, self.max_value)

        self.rescale_config = rescale_config
        #  May need to figure out how to do this generically. In general we want rescale in all but the last layer, however in the case of a sliced layer, we likely want to rescale all layers



    def find_model(self, model_name):
        # Look for model in models_for_circuit_folder
        if "models_onnx" in model_name:
            return model_name
        return f"models_onnx/{model_name}"
    
    # def get_model_type(self, path):
    #     model_path = path + "/model.py"
    #     print(model_path)
    #     print(os.getcwd())
    #     assert os.path.exists(model_path), f"Model file {model_path} does not exist"
    #     module = importlib.import_module(model_path.replace("/", ".").replace(".py", ""))


    #     model_classes = [cls for name, cls in inspect.getmembers(module, inspect.isclass)
    #                      if cls.__module__ == module.__name__]
    #     print(model_classes)
    #     assert len(model_classes) == 1, f"Expected one model class in {model_path}, found {len(model_classes)}"
    #     return model_classes[0]

    def load_metadata(self, model):
        # Load metadata.json

        # Get relevant information/hardcoded in
        # return [required_keys, input_shape, scaling, scale_base, rescale_config]
        pass


    def layer_analyzer(self, layer_data):
        '''
        This function should read in a layer from the metadata, and extract the relvant information to be included in the weights 
        '''
        # This will likely be implemented in onnx_converter
        pass
        

    
    def determine_scaling(self, scaling, scale_base, max_value):
        # Determine scaling based on the max_value we want to be present in the circuit. we also need the model and dummy inputs to determine this. This is not a trivial task, and something we can think more about
        if scaling != -1:
            return scaling
        
        return int(21 / math.log2(scale_base))
    
    def adjust_inputs(self, input_file):
        input_shape = self.input_shape.copy()
        self.input_shape = [math.prod(input_shape)]
        x = super().adjust_inputs(input_file)
        self.input_shape = input_shape.copy()
        return x
    
    def get_outputs(self, inputs):
        return super().get_outputs(inputs).flatten()
    
    def format_inputs(self, inputs):
        x = super().format_inputs(inputs)
        for key in x.keys():
            x[key] = torch.as_tensor(x[key]).flatten().tolist()
        return x




# Must specify the model name
# Look into the models_onnx folder for the name.onnx file

# Create mechanism to load all of this into existing code framework
    

if __name__ == "__main__":
    # names = ["demo_1", "demo_2", "demo_3", "demo_4", "demo_5"]
    names = ["demo"]
    for n in names:
        # name = f"{n}_conv1"
        name = "doom"
        d = GenericModelONNX(name)
        # # d.base_testing()
        # # d.base_testing(run_type=RunType.END_TO_END, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
        d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
        # # d.save_quantized_model("quantized_model.pth")
        d_2 = GenericModelONNX(name)
        # # # # d_2.load_quantized_model("quantized_model.pth")
        d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
        d_2.base_testing(run_type=RunType.PROVE_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")
        d_2.base_testing(run_type=RunType.GEN_VERIFY, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")


