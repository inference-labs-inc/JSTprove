import torch
import torch.nn as nn
import torch.nn.functional as F
from python_testing.utils.pytorch_helpers import ZKModel, RunType
from python_testing.utils.helper_functions import read_from_json, to_json

import sys


class QuantizedLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, scale: int, rescale_output: bool = True):
        super().__init__()
        self.scale = scale
        self.shift = int(scale).bit_length() - 1  # assumes power of 2 scale
        self.rescale_output = rescale_output
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # Quantize weights and biases to integers
        self.weight = nn.Parameter((original_linear.weight.data * scale).long(), requires_grad=False)
        bias = original_linear.bias.data * scale * scale
        # if not self.rescale_output:
        #     bias = bias 
        self.bias = nn.Parameter(bias.long(), requires_grad=False)

    def forward(self, x):
        # Assume x is already scaled (long), do matmul in int domain
        x = x.long()
        # print(x)

        out = torch.matmul(x, self.weight.t())

        out += self.bias

        if self.rescale_output:
            out = out >> self.shift  # scale down

        return out
    
def get_input_shapes_by_layer(model: nn.Module, input_shape):
    hooks = []
    input_shapes = {}

    def register_hook(module_name):
        def hook(module, input, output):
            input_shapes[module_name] = input[0].shape
        return hook

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf layers
            hook = module.register_forward_hook(register_hook(name))
            hooks.append(hook)

    # Run a dummy input through the model
    model.eval()
    dummy_input = torch.randn(*input_shape)
    with torch.no_grad():
        _ = model(dummy_input)

    # Remove hooks
    for h in hooks:
        h.remove()

    return input_shapes
class QuantizedConv2d(nn.Module):
    def __init__(self, original_conv: nn.Conv2d, scale: int, rescale_output: bool = True):
        super().__init__()
        self.scale = scale
        self.shift = int(scale).bit_length() - 1  # assumes scale is a power of 2
        self.rescale_output = rescale_output

        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        self.in_channels = original_conv.in_channels
        self.kernel_size = original_conv.kernel_size

        # Convert weights and biases to long after scaling
        weight = original_conv.weight.data * scale
        self.weight = nn.Parameter(weight.long(), requires_grad=False)

        if original_conv.bias is not None:
            bias = original_conv.bias.data * scale * scale
            # if not self.rescale_output:
            #     bias = bias * scale
            self.bias = nn.Parameter(bias.long(), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x):
        x = x.long()  # ensure input is long
        out = F.conv2d(
            x,
            self.weight,
            bias=None,  # do bias separately to handle shift
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        
        if self.rescale_output:
            out = out >> self.shift
        return out


class CNNDemo(nn.Module):
    def __init__(self, n_actions=10, layers=["conv1", "relu", "conv2", "relu", "conv3", "relu", "reshape", "fc1", "relu", "fc2"]):
        super(CNNDemo, self).__init__()
        self.layers = layers 

        # Convolutional layers
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        # for i in range(2,40):
        #     self.__setattr__(f"conv{i}", nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1))

        # Default to the shape after conv layers (depends on whether each conv layer is used)
        self.fc_input_dim = 16 * 28 * 28

        # Fully connected layers
        self.fc1 = nn.Linear(self.fc_input_dim, 256)
        # for i in range(2,1000):
        #     self.__setattr__(f"fc{i}", nn.Linear(256, 256))

        self.final = nn.Linear(256, n_actions)

        # If the last layer in the list is one of the fully connected layers, set them to `final`
        if layers[-1] == "fc1":
            self.fc1 = nn.Linear(self.fc_input_dim, n_actions)
        else: 
            self.__setattr__(layers[-1], self.final)


    def forward(self, x):
        print(self.layers)
        for l in self.layers:
            if "fc1" == l:
                x = self.fc1(x)
                continue
            if "fc" in l:
                layer_fn = self.__getattr__(l)  # Get the function
                if callable(layer_fn):  # Ensure it's callable
                    x = layer_fn(x)  # Call it with parameter x
            if "reshape" in l:
                x = x.reshape(-1, self.fc_input_dim)  # Flatten before fully connected layers
            if "conv" in l:
                layer_fn = self.__getattr__(l)  # Get the function
                if callable(layer_fn):  # Ensure it's callable
                    x = layer_fn(x)  # Call it with parameter x
            if "relu" in l:
                x = F.relu(x)
        return x

def get_used_layers(model, input_shape):
    used_layers = []
    dummy_input = torch.randn(*input_shape)

    def hook_fn(module, input, output):
        for name, layer in model.named_children():
            if layer is module:
                used_layers.append((name, module))  # Store layer name and module
                break
    
    hooks = []
    for name, layer in model.named_children():  # Register hook for each layer
        hooks.append(layer.register_forward_hook(hook_fn))
    
    with torch.no_grad():  # Run inference
        model(dummy_input)
    
    for hook in hooks:  # Remove hooks after execution
        hook.remove()
    
    return used_layers

# class QuantizedCNNDemo(CNNDemo):
#     def __init__(self, base_model: CNNDemo, scale: int):
#         super().__init__(n_actions=base_model.final.out_features, layers=base_model.layers)
#         self.scale = scale

#         # Replace all conv layers
#         for name, module in base_model.named_modules():
#             if isinstance(module, nn.Conv2d):
#                 quant_layer = QuantizedConv2d(module, scale)
#                 self._set_layer(name, quant_layer)
#             elif isinstance(module, nn.Linear):
#                 quant_layer = QuantizedLinear(module, scale)
#                 self._set_layer(name, quant_layer)

#     def _set_layer(self, name, new_module):
#         parts = name.split(".")
#         obj = self
#         for p in parts[:-1]:
#             obj = getattr(obj, p)
#         setattr(obj, parts[-1], new_module)

def quantize_cnn_demo(model: CNNDemo, scale: int, rescale_config: dict = None):
    rescale_config = rescale_config or {}
    quantized_model = CNNDemo(n_actions=model.final.out_features, layers=model.layers)
    quantized_model.fc_input_dim = model.fc_input_dim  # keep same shape

    # Replace conv and fc layers
    for name, module in model.named_modules():
        rescale = rescale_config.get(name, True)

        if isinstance(module, nn.Conv2d):
            quantized_layer = QuantizedConv2d(module, scale, rescale_output=rescale)
            setattr(quantized_model, name, quantized_layer)
        elif isinstance(module, nn.Linear):
            quantized_layer = QuantizedLinear(module, scale, rescale_output=rescale)
            setattr(quantized_model, name, quantized_layer)

    return quantized_model

def expand_padding(padding_2):
    pad_h, pad_w = padding_2
    return (pad_w, pad_w, pad_h, pad_h)
class Demo(ZKModel):
    def __init__(self, model_file_path: str = None, quantized_model_file_path: str = None):
        self.layers = {}
        self.name = "demo_cnn"

        self.scaling = 21
        self.layers = []
        # self.layers = ["conv1", "relu", "reshape", "fc1"]
        # self.layers = ["conv1", "relu", "conv2", "relu", "reshape", "fc1"]
        # self.layers = ["conv1", "relu", "conv2", "relu", "conv3", "relu",  "reshape", "fc1"]
        # self.layers = ["conv1", "relu", "conv2", "relu", "conv3", "relu", "conv4", "relu",  "reshape", "fc1"]

        # self.layers = ["conv1", "relu", "reshape", "fc1", "relu", "fc2"]
        # self.layers = ["conv1", "relu", "reshape", "fc1", "relu", "fc2", "relu", "fc3"]
        # self.layers = ["conv1", "relu", "reshape", "fc1", "relu", "fc2", "relu", "fc3", "relu", "fc4"]
        # self.layers = ["conv1", "relu", "reshape", "fc1", "relu", "fc2", "relu", "fc3", "relu", "fc4", "relu", "fc5", "relu", "fc6", "relu", "fc7", "relu", "fc8", "relu", "fc9", "relu", "fc10"]
        for i in range(1,2):
            self.layers.append(f"conv{i}")
            self.layers.append("relu")
        self.layers.append("reshape")
        self.layers.append("fc1")


        # for i in range(2,4):
        #     self.layers.append("relu")
        #     self.layers.append(f"fc{i}")

        # self.layers = ["conv1", "relu", "conv2", "relu", "conv3", "relu", "conv4", "relu", "reshape", "fc1", "relu", "fc2", "relu", "fc3", "relu", "fc4"]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CNNDemo(layers=self.layers).to(device)
        model.eval()
        self.model = model
        rescale_config = {"fc1": False}

        self.quantized_model = quantize_cnn_demo(model, 2**self.scaling, rescale_config=rescale_config)
        self.quantized_model.eval()

        self.input_shape = [1, 4, 28, 28]
        self.exclude_keys = ['quantized', 'scaling']

        self.input_data_file = "doom_data/doom_input.json"
        self.first_inputs = torch.rand(self.input_shape)

    def save_model(self, file_path: str):
        torch.save(self.model.state_dict(), file_path)

    
    def load_model(self, file_path: str):
        self.model.load_state_dict(torch.load(file_path))

    def save_quantized_model(self, file_path: str):
        torch.save(self.quantized_model.state_dict(), file_path)

    
    def load_quantized_model(self, file_path: str):
        self.quantized_model.load_state_dict(torch.load(file_path))



    def get_weights(self):
        input_shapes = get_input_shapes_by_layer(self.quantized_model, self.input_shape)  # example input
        used_layers = get_used_layers(self.quantized_model, self.input_shape) 
        # Can combine the above into 1 function

        
        weights = {}
        weights["layers"] = self.layers
        weights["scaling"] = self.scaling
        
        for name, module in used_layers:
            print(name, module)
            if isinstance(module, (nn.Conv2d, QuantizedConv2d)):
                weights.setdefault("conv_weights", []).append(module.weight.tolist())
                weights.setdefault("conv_bias", []).append(module.bias.tolist())
                weights.setdefault("conv_strides", []).append(module.stride)
                weights.setdefault("conv_kernel_shape", []).append(module.kernel_size)
                weights.setdefault("conv_group", []).append([module.groups])
                weights.setdefault("conv_dilation", []).append(module.dilation)
                weights.setdefault("conv_pads", []).append(expand_padding(module.padding))
                weights.setdefault("conv_input_shape", []).append(input_shapes[name])

            
            if isinstance(module, (nn.Linear, QuantizedLinear)):
                print(name)
                weights.setdefault("fc_weights", []).append(module.weight.transpose(0, 1).tolist())
                weights.setdefault("fc_bias", []).append(module.bias.unsqueeze(0).tolist())
        # sys.exit()
        # weights.setdefault("fc_weights", []).append(self.quantized_model.fc1.weight.transpose(0, 1).tolist())
        # weights.setdefault("fc_bias", []).append(self.quantized_model.fc1.bias.unsqueeze(0).tolist())
        

        return [weights, {}]


    def get_outputs(self):
        input_arr = self.get_inputs(self.input_data_file).reshape(self.input_shape)
        inputs = {"input": input_arr.long().tolist()}
        out = self.quantized_model(torch.mul(input_arr, 1))
        output_model = {"output": out.long().tolist()}
        return inputs, output_model
        
    
    def get_model_params(self, output = None):
        inputs, outputs = self.get_outputs()
        weights = self.get_weights()
        return inputs, weights, outputs
    
    def get_model_params(self, output = None):
        weights, weights_2 = self.get_weights()
        inputs, output = self.get_outputs()
        
        return inputs,[weights,weights_2],output




    

if __name__ == "__main__":
    # names = ["demo_1", "demo_2", "demo_3", "demo_4", "demo_5"]
    names = ["demo_5"]
    for n in names:
        # name = f"{n}_conv1"
        name = n
        d = Demo()
        # d.base_testing()
        # d.base_testing(run_type=RunType.END_TO_END, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
        d.base_testing(run_type=RunType.COMPILE_CIRCUIT, dev_mode=True, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")
        d.save_quantized_model("quantized_model.pth")
        d_2 = Demo()
        d_2.load_quantized_model("quantized_model.pth")
        d_2.base_testing(run_type=RunType.GEN_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt", write_json = True)
        # d.base_testing(run_type=RunType.PROVE_WITNESS, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")
        # d.base_testing(run_type=RunType.GEN_VERIFY, dev_mode=False, witness_file=f"{name}_witness.txt", circuit_path=f"{name}_circuit.txt")


