import json
import torch
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify
import os
from python_testing.relu import ReLU, ConversionType

from python_testing.convolution import Convolution, QuantizedConv
# from python_testing.matrix_multiplication import QuantizedMatrixMultiplication
from python_testing.gemm import QuantizedGemm, Gemm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from python_testing.utils.pytorch_helpers import ZKModel



class CNNDemo(nn.Module):
    def __init__(self, n_actions=10, layers=["conv1", "relu", "conv2", "relu", "conv3", "relu", "reshape", "fc1", "relu", "fc2"]):
        super(CNNDemo, self).__init__()
        self.layers = layers 

        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)

        # Default to the shape after conv layers (depends on whether each conv layer is used)
        self.fc_input_dim = 16 * 28 * 28
        if "conv2" in layers:
            self.fc_input_dim = 16 * 28 * 28
        if "conv3" in layers:
            self.fc_input_dim = 16 * 7 * 7
        if "conv4" in layers:
            self.fc_input_dim = 16 * 4 * 4

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)

        self.final = nn.Linear(256, n_actions)

        # If the last layer in the list is one of the fully connected layers, set them to `final`
        if layers[-1] == "fc2":
            self.fc2 = self.final
        elif layers[-1] == "fc3":
            self.fc3 = self.final
        elif layers[-1] == "fc4":
            self.fc4 = self.final
        elif layers[-1] == "fc1":
            self.fc1 = nn.Linear(self.fc_input_dim, n_actions)

    def forward(self, x):
        print(self.layers)
        x = F.relu(self.conv1(x))
        if "conv2" in self.layers:
            x = F.relu(self.conv2(x))
        if "conv3" in self.layers:
            x = F.relu(self.conv3(x))
        if "conv4" in self.layers:
            x = F.relu(self.conv4(x))

        x = x.reshape(-1, self.fc_input_dim)  # Flatten before fully connected layers
        x = self.fc1(x)

        if "fc2" in self.layers:
            print(x.shape)
            x = F.relu(x)
            x = self.fc2(x)
            print(x.shape)
        if "fc3" in self.layers:
            print(x.shape)
            x = F.relu(x)
            x = self.fc3(x)
        if "fc4" in self.layers:
            x = F.relu(x)
            x = self.fc4(x)

        # x = self.final(x)  # Always apply the final layer

        return x
    
class Doom(ZKModel):
    def __init__(self):
        self.layers = {}
        self.name = "demo_cnn"

        self.scaling = 21
        # self.layers = ["conv1", "relu", "reshape", "fc1"]
        # self.layers = ["conv1", "relu", "conv2", "relu", "reshape", "fc1"]
        # self.layers = ["conv1", "relu", "conv2", "relu", "conv3", "relu",  "reshape", "fc1"]
        # self.layers = ["conv1", "relu", "conv2", "relu", "conv3", "relu", "conv4", "relu",  "reshape", "fc1"]

        # self.layers = ["conv1", "relu", "reshape", "fc1", "relu", "fc2"]
        self.layers = ["conv1", "relu", "reshape", "fc1", "relu", "fc2", "relu", "fc3"]
        # self.layers = ["conv1", "relu", "reshape", "fc1", "relu", "fc2", "relu", "fc3", "relu", "fc4"]

        # self.layers = ["conv1", "relu", "conv2", "relu", "conv3", "relu", "conv4", "relu", "reshape", "fc1", "relu", "fc2", "relu", "fc3", "relu", "fc4"]


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNNDemo(layers=self.layers).to(device)
        model.eval()
        self.model = model
        self.input_shape = [1, 4, 28, 28]

        self.input_data_file = "doom_data/doom_input.json"



    def get_model_params(self):
        exclude_keys = ['quantized', 'scaling']
        
        input_arr = self.get_inputs(self.input_data_file).reshape(self.input_shape)
        inputs = {"input": input_arr.long().tolist()}
        weights = {"layers":self.layers}
        weights_2 = {}
        input = {}
        output = {}
        first_inputs = torch.tensor(self.read_input()).reshape(self.input_shape)
        outputs = self.read_output(self.model, first_inputs)
        
        

        previous_output_tensor = input_arr

        for layer in self.layers:
            layer_params = {layer:{"quant":True}}
            if any(char.isdigit() for char in layer):
                l = self.model.__getattr__(layer)
                try:
                    layer_params = {layer:{"strides": l.stride}}
                except:
                    pass
                if layer == self.layers[-1]:
                    layer_params[layer]["quant"] = False


            else:
                l = layer
                if "reshape" in layer:
                    layer_params = {layer:{"shape": [-1,self.model.fc_input_dim]}}

            # layer_params = self.model.__getattr__(layers[1])
            #Rework inputs to function
            if not layer in "reshape":
                (input, weight, output) = self.get_layer(input_arr, layer, l, **layer_params.get(layer, {"": None}))
                if weight:
                    if ("fc1" in layer) or ("fc2" in layer) or ("fc3" in layer) or ("fc4" in layer):
                        weights_2.update({f"{layer}_" + key if key not in exclude_keys else key: value for key, value in weight.items()})
                    else:
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
            


        
        for i in range(previous_output_tensor.shape[0]):
            for j in range(previous_output_tensor.shape[1]):
                # error_margin = 0.001
                # x = previous_output_tensor[i][j]/(2**(2*self.scaling)) / outputs[i][j]
                # assert(x < (1 + error_margin))
                # assert(x > (1 - error_margin))
                assert(abs(previous_output_tensor[i][j]/(2**(2*self.scaling)) - outputs[i][j]) < 0.01)
        return inputs,[weights,weights_2],output

    

if __name__ == "__main__":
    Doom().run_circuit()