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



class DoomAgent(nn.Module):
    def __init__(self, n_actions=7):
        super(DoomAgent, self).__init__()

        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        self.fc_input_dim = 32 * 7 * 7

        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.reshape(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def act(self, state, epsilon=0.0):
        if np.random.random() < epsilon:
            return np.random.randint(7)

        with torch.no_grad():
            state_tensor = (
                torch.FloatTensor(state).unsqueeze(0).to(next(self.parameters()).device)
            )
            q_values = self.forward(state_tensor)
            return q_values.argmax().item()
    
class Doom(ZKModel):
    def __init__(self, file_name="model/doom_checkpoint.pth"):
        self.layers = {}
        self.name = "doom"

        self.scaling = 21

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DoomAgent().to(device)
        checkpoint = torch.load(file_name, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.model = model
        self.input_shape = [1, 4, 28, 28]

        self.input_data_file = "doom_data/doom_input.json"


    def get_model_params(self):
        exclude_keys = ['quantized', 'scaling']
        
        input_arr = self.get_inputs(self.input_data_file).reshape(self.input_shape)
        inputs = {"input": input_arr.long().tolist()}
        weights = {}
        weights_2 = {}
        input = {}
        output = {}
        first_inputs = torch.tensor(self.read_input()).reshape(self.input_shape)
        outputs = self.read_output(self.model, first_inputs)
        
        layers = ["conv1", "relu", "conv2", "relu", "conv3", "relu", "reshape", "fc1", "relu", "fc2"]

        previous_output_tensor = input_arr

        for layer in layers:
            layer_params = {layer:{"quant":True}}
            if any(char.isdigit() for char in layer):
                l = self.model.__getattr__(layer)
                try:
                    layer_params = {layer:{"strides": l.stride}}
                except:
                    pass
                if layer == "fc2":
                    layer_params[layer]["quant"] = False


            else:
                l = layer
                if "reshape" in layer:
                    layer_params = {layer:{"shape": [-1,1568]}}

            # layer_params = self.model.__getattr__(layers[1])
            #Rework inputs to function
            if not layer in "reshape":
                (input, weight, output) = self.get_layer(input_arr, layer, l, **layer_params.get(layer, {"": None}))
                if weight:
                    if "fc2" in layer:
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
                error_margin = 0.0001
                x = previous_output_tensor[i][j]/(2**(2*self.scaling)) / outputs[i][j]
                assert(x < (1 + error_margin))
                assert(x > (1 - error_margin))
        return inputs,[weights,weights_2],output

    

if __name__ == "__main__":
    Doom().run_circuit()
