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


class LayerInfo():
    def __init__(self, name, input_shape, output_shape, weight_shape = None):
        self.name = name
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weight_shape = weight_shape
        self.inputs = None
        self.outputs = None
        self.weights = None
        self.bias = None

    def update_inputs(self, inputs):
        self.inputs = inputs.reshape(self.input_shape)

    def update_outputs(self, outputs):
        self.outputs = outputs.reshape(self.output_shape)

    def update_weights(self, weights):
        if self.weight_shape:
            self.weights = weights.reshape(self.weight_shape)
        else:
            self.weights = None

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

class Doom():
    def __init__(self, file_name = "model/doom_checkpoint.pth"):
        self.layers = {}

        self.scaling = 21

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DoomAgent().to(device)
        checkpoint = torch.load(file_name, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        self.model = model
        self.input_shape = [1, 4, 28, 28]



        
        
        
    def read_tensor_from_file(self, file_name):
        """Reads a tensor from a file and returns it as a PyTorch tensor."""
        with open(file_name, 'r') as f:
            data = f.read().split()
            # Convert data to a float and then to a PyTorch tensor
            tensor_data = torch.tensor([float(d) for d in data])
        return tensor_data
    
    def read_weights(self, model, layer_name):
        """Reads the weights for the layers of the model from files."""
        pass


    def read_input(self, file_name = "doom_data/doom_input.json"):
        """Reads the inputs to each layer of the model from text files."""
        with open(file_name, 'r') as file:
            data = json.load(file)
            return data["input_data"]



    def read_output(self, model, input_data):
        """Reads the outputs for each layer of the model from text files."""
        with torch.no_grad():  # Disable gradient calculation during inference
            output = model(torch.tensor(input_data))
            return output

    def run_circuit(self):
        """Simulates running the model by passing inputs through layers with weights."""
        print("Running circuit...")
        proof_system = ZKProofSystems.Expander
        proof_folder = "analysis"
        output_folder = "output"
        temp_folder = "temp"
        input_folder = "inputs"
        weights_folder = "weights"
        circuit_folder = ""
        name = "doom"

        witness_file, input_file, proof_path, public_path, verification_key, circuit_name, weights_file, output_file = get_files(
                input_folder, proof_folder, temp_folder, circuit_folder, weights_folder, name, output_folder, proof_system)

        exclude_keys = ['quantized', 'scaling']
        
        input_arr = self.get_inputs().reshape(self.input_shape)
        inputs = {"input": input_arr.long().tolist()}
        weights = {}
        weights_2 = {}
        input = {}
        output = {}
        first_inputs = torch.tensor(self.read_input()).reshape(self.input_shape)
        outputs = self.read_output(self.model, first_inputs)
        
        layers = ["conv1", "relu", "conv2", "relu", "conv3", "relu", "reshape", "fc1", "relu", "fc2"]
        l = self.model.__getattr__(layers[0])
        previous_output_tensor = input_arr
        x = self.model.conv1(first_inputs)
        x = F.relu(x)
        x = self.model.conv2(x)
        x = F.relu(x)

        x = self.model.conv3(x)

        x = x.reshape([-1,1568])
        x = self.model.fc1(x)

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



        # NO NEED TO CHANGE anything below here!
        to_json(inputs, input_file)

        # Write output to json
        outputs = {"output": value for key, value in output.items()}
        # outputs = {"outputs": reshape_out.tolist()}
        to_json(outputs, output_file)

        to_json(weights, weights_file)
        to_json(weights_2, weights_file[:-5] + '2' + weights_file[-5:])


        # ## Run the circuit
        prove_and_verify(witness_file, input_file, proof_path, public_path, verification_key, circuit_name, proof_system, output_file)



    def check_4d_eq(self, input_tensor_1, input_tensor_2):
        for i in range(input_tensor_1.shape[0]):
            for j in range(input_tensor_1.shape[1]):
                for k in range(input_tensor_1.shape[2]):
                    for l in range(input_tensor_1.shape[3]):
                        # print(input_tensor_1[i][j][k][l],  input_tensor_2[i][j][k][l])
                        assert(abs(input_tensor_1[i][j][k][l] -  input_tensor_2[i][j][k][l]) < 1)

    def check_2d_eq(self, input_tensor_1, input_tensor_2):
        for i in range(input_tensor_1.shape[0]):
            for j in range(input_tensor_1.shape[1]):
                assert(abs(input_tensor_1[i][j] -  input_tensor_2[i][j]) < 1)


    def get_layer(self, inputs, layer_name, layer, **kwargs):
        if layer_name == "input":
            return self.get_inputs()
        elif "conv" in layer_name:
            return self.get_circuit_conv(inputs, layer, kwargs.get("strides", (1,1)))
        elif "relu" in layer_name:
            return self.get_relu(inputs)
        elif "fc" in  layer_name:
            return self.get_mat_mult(inputs, layer, kwargs.get("quant", True))
        else:
            raise(ValueError("Layer not found"))


    def get_inputs(self):
        inputs = self.read_input()
        return torch.mul(torch.tensor(inputs),2**self.scaling).long()

    def get_relu(self, inputs):
        relu_circuit = ReLU(conversion_type = ConversionType.TWOS_COMP)
        relu_circuit.inputs_1 = inputs
        out = relu_circuit.get_outputs()
        input, output = relu_circuit.get_twos_comp_model_data(out)
        return (input, None, output)
    
    def get_circuit_conv(self, inputs, layer, strides = (1,1)):
        weights = layer.weight
        bias = layer.bias
        # layers = self.layers[layer_name]
        conv_circuit = QuantizedConv()
        conv_circuit.input_arr = inputs
        conv_circuit.weights = torch.mul(weights, 2**self.scaling).long()
        conv_circuit.bias = torch.mul(bias, 2**(self.scaling*2)).long()
        

        conv_circuit.scaling = self.scaling
        conv_circuit.strides = strides

        return conv_circuit.get_model_params(conv_circuit.get_output())
    
    def get_mat_mult(self, inputs, layer, quant = True):
        weights = layer.weight
        bias = layer.bias
        # layers = self.layers[layer]

        if quant:
            mat_mult_circuit = QuantizedGemm()
        else:
            mat_mult_circuit = Gemm()

        mat_mult_circuit.matrix_a = inputs.long()
        mat_mult_circuit.matrix_b = torch.transpose(torch.mul(weights, 2**self.scaling),0,1).long()

        # Scale up matrix c, twofold, to account for the multiplication that has just taken place
        mat_mult_circuit.matrix_c = torch.reshape(torch.mul(bias, 2**(self.scaling*2)), [mat_mult_circuit.matrix_a.shape[0],mat_mult_circuit.matrix_b.shape[1]]).long()
        
        mat_mult_circuit.scaling = self.scaling
        mat_mult_circuit.alpha = torch.tensor(1)
        mat_mult_circuit.beta = torch.tensor(1)

        gemm = mat_mult_circuit.get_outputs()
        return mat_mult_circuit.get_model_params(gemm)

        

    

if __name__ == "__main__":
    Doom().run_circuit()


    
    

    
    # proof_system = ZKProofSystems.Expander
    # proof_folder = "analysis"
    # output_folder = "output"
    # temp_folder = "temp"
    # input_folder = "inputs"
    # weights_folder = "weights"
    # circuit_folder = ""
    # #Rework inputs to function
    # test_circuit = ReLU(conversion_type = ConversionType.TWOS_COMP)
    # test_circuit.base_testing(input_folder,proof_folder, temp_folder, circuit_folder, weights, proof_system, output_folder)

