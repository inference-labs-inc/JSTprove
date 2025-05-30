import importlib
import json
import math
import os
import pkgutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib.util
import inspect
from python_testing.utils.pytorch_helpers import Layer, ZKModel, RunType, filter_dict_for_dataclass

import sys


from dataclasses import asdict, dataclass, field, fields
from typing import List, Dict, Optional


class GenericModelTorch(ZKModel):
    def __init__(self, model_name, models_folder = None, model_file_path: str = None, quantized_model_file_path: str = None, layers = None):
        self.max_value = 2**32
        self.name = "generic_demo_v2"
        # self.required_keys = ["input"]
        # self.scale_base = 2
        # self.scaling = 21


        self.input_shape = [1, 4, 28, 28]
        # self.rescale_config = {self.layers[-1]: False}
        self.rescale_config = {}

        self.path = self.find_model(model_name)
        self.model_type = self.get_model_type(self.path)
        print(self.model_type)

        # self.model_params = {"layers": self.layers, "n_actions": 10}
        self.model_params = {}

        (required_keys, input_shape, scaling, scale_base, rescale_config) = self.load_metadata(self.path)
        # self.model_file_name = self.path + "/doom_checkpoint.pt"

        self.required_keys = required_keys
        # self.name = name
        self.input_shape = input_shape if input_shape is not None else None
        self.scale_base = self.scale_base if scale_base != -1 else 2
        self.scaling = self.determine_scaling(scaling, self.scale_base, self.max_value)
        self.rescale_config = rescale_config
        #  May need to figure out how to do this generically. In general we want rescale in all but the last layer, however in the case of a sliced layer, we likely want to rescale all layers



    def find_model(self, model_name):
        # Look for model in models_for_circuit_folder
        return "models_for_circuit/doom"
    
    def get_model_type(self, path):
        model_path = path + "/model.py"
        assert os.path.exists(model_path), f"Model file {model_path} does not exist"


        # spec = importlib.util.spec_from_file_location("dynamic_model", model_path)
        # module = importlib.util.module_from_spec(spec)
        # spec.loader.exec_module(module)
        module = importlib.import_module(model_path.replace("/", ".").replace(".py", ""))


        model_classes = [cls for name, cls in inspect.getmembers(module, inspect.isclass)
                         if cls.__module__ == module.__name__]
        print(model_classes)
        assert len(model_classes) == 1, f"Expected one model class in {model_path}, found {len(model_classes)}"
        return model_classes[0]

    def load_metadata(self, path):
        # Load metadata.json
        metadata_path = os.path.join(path, "metadata.json")
        assert os.path.exists(metadata_path), f"Model file {metadata_path} does not exist"

        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        self.layers = []
        layers = self.layer_analyzer(metadata.get("segments"))
        for l in range(len(layers)):
            if isinstance(layers[l], Layer):
                self.layers.append(asdict(layers[l]))

        # Store metadata values
        required_keys = metadata.get("required_keys", ["input"])
        
        input_shape = metadata.get("input_shape", [1, 4, 28, 28])
        # TODO figure out input shape properly
        scaling = metadata.get("scaling", -1)
        scale_base = metadata.get("scale_base", -1)
        rescale_config = metadata.get("rescale_config", {})

        return [required_keys, input_shape, scaling, scale_base, rescale_config]
    
    def layer_analyzer(self, layer_data):
        '''
        This function should read in a layer from the metadata, and extract the relvant information to be included in the weights 
        '''
        layers = []
        for l in layer_data:
            l['name'] = l['segment_name']
            layers.append(Layer(**filter_dict_for_dataclass(Layer, l)))
        return layers
        

    
    def determine_scaling(self, scaling, scale_base, max_value):
        # Determine scaling based on the max_value we want to be present in the circuit. we also need the model and dummy inputs to determine this. This is not a trivial task, and something we can think more about
        if scaling != -1:
            return scaling
        
        return int(21 / math.log2(scale_base))




# Must specify the model name
# Look into the models_for_circuit folder for the folder with the model name
# assert that all required files and parts of the files are present. That should be

    # model.pt
    # model.py - with a single model inside
    # metadata - containing all of the layers to be analyzed and sent to rust

# Create mechanism to load all of this into existing code framework
    

if __name__ == "__main__":
    # names = ["demo_1", "demo_2", "demo_3", "demo_4", "demo_5"]
    names = ["demo"]
    for n in names:
        # name = f"{n}_conv1"
        name = "doom"
        d = GenericModelTorch(name)
        # # d.base_testing()
        # # d.base_testing(run_type=RunType.END_TO_END, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
        d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, circuit_path=f"{name}_circuit.txt")
        # # d.save_quantized_model("quantized_model.pth")
        d_2 = GenericModelTorch(name)
        # # # # d_2.load_quantized_model("quantized_model.pth")
        d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
        d_2.base_testing(run_type=RunType.PROVE_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")
        d_2.base_testing(run_type=RunType.GEN_VERIFY, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")


