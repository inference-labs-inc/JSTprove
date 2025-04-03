import torch
import torch.nn as nn
import torch.nn.functional as F
from python_testing.utils.pytorch_helpers import ZKModel, RunType
from python_testing.utils.helper_functions import read_from_json

import sys



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
            self.fc_input_dim = 16 * 14 * 14
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
            x = F.relu(x)
            x = self.fc2(x)
        if "fc3" in self.layers:
            x = F.relu(x)
            x = self.fc3(x)
        if "fc4" in self.layers:
            x = F.relu(x)
            x = self.fc4(x)
            
        return x

def get_used_layers(model, input_tensor):
    used_layers = []

    def hook_fn(module, input, output):
        for name, layer in model.named_children():
            if layer is module:
                used_layers.append((name, module))  # Store layer name and module
                break
    
    hooks = []
    for name, layer in model.named_children():  # Register hook for each layer
        hooks.append(layer.register_forward_hook(hook_fn))
    
    with torch.no_grad():  # Run inference
        model(input_tensor)
    
    for hook in hooks:  # Remove hooks after execution
        hook.remove()
    
    return used_layers
class Demo(ZKModel):
    def __init__(self):
        self.layers = {}
        self.name = "demo_cnn"

        self.scaling = 21
        # self.layers = ["conv1", "relu", "reshape", "fc1"]
        # self.layers = ["conv1", "relu", "conv2", "relu", "reshape", "fc1"]
        # self.layers = ["conv1", "relu", "conv2", "relu", "conv3", "relu",  "reshape", "fc1"]
        self.layers = ["conv1", "relu", "conv2", "relu", "conv3", "relu", "conv4", "relu",  "reshape", "fc1"]

        # self.layers = ["conv1", "relu", "reshape", "fc1", "relu", "fc2"]
        # self.layers = ["conv1", "relu", "reshape", "fc1", "relu", "fc2", "relu", "fc3"]
        # self.layers = ["conv1", "relu", "reshape", "fc1", "relu", "fc2", "relu", "fc3", "relu", "fc4"]

        # self.layers = ["conv1", "relu", "conv2", "relu", "conv3", "relu", "conv4", "relu", "reshape", "fc1", "relu", "fc2", "relu", "fc3", "relu", "fc4"]
        


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNNDemo(layers=self.layers).to(device)
        # for name, layer in model.named_children():
        #     print(name, layer)
        # print(model.state_dict().keys())
        model.eval()
        self.model = model
        self.input_shape = [1, 4, 28, 28]
        self.exclude_keys = ['quantized', 'scaling']

        self.input_data_file = "doom_data/doom_input.json"
        first_inputs = torch.tensor(self.read_input()).reshape(self.input_shape)
        used_layers = get_used_layers(model, first_inputs)
        print("Used layers:", used_layers)
        # sys.exit()

        # torch.onnx.export(model, first_inputs, f = "demo_cnn_full.onnx")
    
    def get_weights(self):
        layers = self.layers
        model = self.model
        exclude_keys = self.exclude_keys
        weights = {"layers": layers}
        weights_2 = {}
        inputs = torch.zeros(self.input_shape)
        layer_params = {}
        for layer in layers:
            layer_params.update({layer:{"quant":True}})
            if layer == "reshape":
                layer_params[layer].update({"shape": [-1,self.model.fc_input_dim]})
                inputs = inputs.reshape((-1, self.model.fc_input_dim))
                continue
            layer_params[layer].update({"quant": True})
            
            if any(char.isdigit() for char in layer):
                l = model.__getattr__(layer)
                try:
                    layer_params.update({layer:{"strides": l.stride}})
                except:
                    pass
                if layer == layers[-1]:
                    layer_params[layer]["quant"] = False
            else:
                l = layer
                if "reshape" in layer:
                    layer_params = {layer:{"shape": [-1,self.model.fc_input_dim]}}

            # Extract weight information
            _, weight, output = self.get_layer_weights(inputs, layer, l, **layer_params.get(layer, {"": None}))
                
            inputs = torch.LongTensor(output["output"])
            if weight:
                if any(f"fc{i}" in layer for i in range(1, 5)):
                    weights_2.update({f"{layer}_" + key if key not in exclude_keys else key: value for key, value in weight.items()})
                else:
                    weights.update({f"{layer}_" + key if key not in exclude_keys else key: value for key, value in weight.items()})
        

        self.layer_params = layer_params
        return weights, weights_2, 


    def get_outputs(self):
        layer_params = self.layer_params
        input_arr = self.get_inputs(self.input_data_file).reshape(self.input_shape)
        inputs = {"input": input_arr.long().tolist()}
        first_inputs = torch.tensor(self.read_input()).reshape(self.input_shape)
        outputs = self.read_output(self.model, first_inputs)
        
        previous_output_tensor = input_arr
        
        for layer in self.layers:
            if not layer in "reshape":
                if any(char.isdigit() for char in layer):
                    l = self.model.__getattr__(layer)
                else:
                    l = layer
                (input, weight, output) = self.get_layer_out(input_arr, layer, l, **layer_params.get(layer, {"": None}))

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
                assert abs(previous_output_tensor[i][j] / (2 ** (2 * self.scaling)) - outputs[i][j]) < 0.01
        
        return inputs, output
    
    def get_model_params(self, output = None):
        weights, weights_2 = self.get_weights()
        inputs, output = self.get_outputs()
        
        return inputs,[weights,weights_2],output
    



    

if __name__ == "__main__":
    # names = ["demo_1", "demo_2", "demo_3", "demo_4", "demo_5"]
    names = ["demo_1"]
    for name in names:
        d = Demo()
        # d.base_testing()
        # d.base_testing(run_type=RunType.END_TO_END, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
        d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")
        d.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
        # d.base_testing(run_type=RunType.PROVE_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")
        # d.base_testing(run_type=RunType.GEN_VERIFY, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")


