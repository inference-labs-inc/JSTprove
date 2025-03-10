import json
import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify
import os
from python_testing.circuit_components.relu import ReLU, ConversionType

from python_testing.circuit_components.convolution import Convolution, QuantizedConv
# from python_testing.matrix_multiplication import QuantizedMatrixMultiplication
from python_testing.circuit_components.gemm import QuantizedGemm, Gemm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from python_testing.utils.pytorch_helpers import ZKModel
from onnx2pytorch import ConvertModel
import onnx




class DoomModel(nn.Module):
    def __init__(self, input_shape=(1, 28, 28)):
        super(DoomModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=5, stride=2)
        self.relu = nn.ReLU()
        self.flattened_size = self._get_flattened_size(input_shape)
        self.d1 = nn.Linear(self.flattened_size, 48)
        self.d2 = nn.Linear(48, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.flatten(start_dim=1)
        x = self.d1(x)
        x = self.relu(x)
        return self.d2(x)

    def _get_flattened_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            x = self.conv1(dummy_input)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.relu(x)
            return x.numel()
    
class Doom(ZKModel):
    def __init__(self, file_name="model/doom_2.onnx"):
        self.layers = {}
        self.name = "doom_2"

        self.scaling = 21

        onnx_model = onnx.load(file_name)
        self.file_name = file_name

        # Check if the model is valid
        onnx.checker.check_model(onnx_model)

        # model = ConvertModel(onnx_model)
        self.w_and_b = self.weights_onnx_to_torch_format(onnx_model)
        final_layer = "d2"

        layer_params = {}
        
        for i, node in enumerate(onnx_model.graph.node):
            layer_name = node.name.split('/')[1]  # Split by '/' and get the second element

            if node.op_type == "Conv":
                # Extract attributes specific to Conv layers
                conv_attrs = {attr.name: attr for attr in node.attribute}
                kernel_shape = conv_attrs.get("kernel_shape", None)
                strides = conv_attrs.get("strides", None)
                pads = conv_attrs.get("pads", None)
                dilations = conv_attrs.get("dilations", None)
                group = conv_attrs.get("group", None)
                layer_params[layer_name] = {"strides": strides.ints, "kernel_shape": kernel_shape.ints, "pads": pads.ints, "dilations": dilations.ints, "group": [group.i]}
            elif node.op_type == "Gemm":
                layer_params[layer_name] = {"quant": True}
            if node.op_type == "Flatten":
                layer_params["reshape"] = {"shape": [-1, 48]}

            if layer_name == final_layer:
                layer_params[layer_name]["quant"] = False

        self.layer_params = layer_params

        

        self.model = onnx_model
        self.input_shape = [1, 28, 28]

        self.input_data_file = "doom_data/doom_input.json"


    def get_model_params(self):
        exclude_keys = ['quantized', 'scaling']
        
        input_arr = self.get_inputs(self.input_data_file).reshape([1,4,28,28])[:,0,:,:].reshape([1,1,28,28])        
        

        inputs = {"input": input_arr.long().tolist()}
        weights = {}
        weights_2 = {}
        input = {}
        output = {}
        first_inputs = np.asarray(torch.tensor(self.read_input()).reshape([1,4,28,28])[:,0,:,:].reshape([1,1,28,28]))
        outputs = self.read_output(self.file_name, first_inputs, is_torch=False)
        temp = torch.tensor(self.read_input()).reshape([1,4,28,28])[:,0,:,:].reshape([1,1,28,28])
        
        layers = ["conv1", "relu", "conv2", "relu", "reshape", "fc1", "relu", "fc2"]
        layer_translation = {"fc1": "d1", "fc2": "d2"}

        previous_output_tensor = input_arr
        layer_params = self.layer_params

        for layer in layers:
            if not layer in "reshape":
                translated_layer = layer_translation.get(layer, layer)
                l = self.w_and_b.get(translated_layer)
                (input, weight, output) = self.get_layer(input_arr, layer, l, **layer_params.get(translated_layer, {"": None}))
                if weight:
                    # if "fc2" in layer:
                    #     weights_2.update({f"{layer}_" + key if key not in exclude_keys else key: value for key, value in weight.items()})
                    # else:
                        if "pads" in weight.keys():
                            weight["pads"] = list(weight["pads"])
                        if "strides" in weight.keys():
                            weight["strides"] = list(weight["strides"])
                        weights.update({f"{layer}_" + key if key not in exclude_keys else key: value for key, value in weight.items()})
                input_arr = torch.LongTensor(input["input"])
                output_tensor = torch.LongTensor(output["output"])
                try:
                    self.check_4d_eq(input_arr,previous_output_tensor)
                except IndexError:
                    self.check_2d_eq(input_arr,previous_output_tensor)

                previous_output_tensor = output_tensor
                input_arr = output_tensor
            else:
                input_arr = torch.reshape(previous_output_tensor, layer_params["reshape"]["shape"])
                previous_output_tensor = input_arr
        weights["quantized"] = True


        
        for i in range(previous_output_tensor.shape[0]):
            for j in range(previous_output_tensor.shape[1]):
                error_margin = 0.001
                x = previous_output_tensor[i][j]/(2**(2*self.scaling)) / outputs[0][i][j]
                assert(x < (1 + error_margin))
                assert(x > (1 - error_margin))
        return inputs, [weights],output

    

if __name__ == "__main__":
    Doom().run_circuit()
