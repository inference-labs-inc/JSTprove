from dataclasses import dataclass, fields
import inspect
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import onnx
import onnxruntime as ort


from python_testing.circuit_components.circuit_helpers import RunType
from python_testing.utils.pytorch_partial_models import QuantizedConv2d, QuantizedLinear
from python_testing.utils.model_converter import ZKModelBase, ModelConverter




@dataclass
class Parameter:
    shape: List[int]
    size: int

@dataclass
class Layer:
    index: int
    name: str
    type: str
    # shape: List[int]
    # size: int

    parameters: int
    in_channels: Optional[int] = None
    out_channels: Optional[int] = None
    kernel_size: Optional[List[int]] = None
    stride: Optional[List[int]] = None
    padding: Optional[List[int]] = None
    activation: Optional[str] = None
    input_reshape: Optional[Dict] = None

def filter_dict_for_dataclass(cls, d):
    allowed_keys = {f.name for f in fields(cls)}
    return {k: v for k, v in d.items() if k in allowed_keys}



class PytorchConverter(ModelConverter):
    def save_model(self, file_path: str):
        torch.save(self.model.state_dict(), file_path)
    
    def load_model(self, file_path: str, model_type = None):
        self.model.load_state_dict(torch.load(file_path))

    def save_quantized_model(self, file_path: str):
        torch.save(self.quantized_model, file_path)
    
    def load_quantized_model(self, file_path: str):
        # self.quantized_model.load_state_dict(torch.load(file_path))
        self.quantized_model = torch.load(file_path, weights_only=False)


    def expand_padding(self, padding_2):
        if len(padding_2) != 2:
            raise(ValueError("Expand padding requires initial padding of dimension 2"))
        pad_h, pad_w = padding_2
        return (pad_w, pad_w, pad_h, pad_h)
    
    def get_used_layers(self, model, input_shape):
        used_layers = []
        dummy_input = torch.randn(*input_shape)

        def hook_fn(name):
            def fn(module, input, output):
                used_layers.append((name, module))
            return fn

        hooks = []
        for name, module in model.named_modules():
            # Only leaf modules to avoid duplicate calls
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(hook_fn(name)))

        with torch.no_grad():
            model(dummy_input)

        for hook in hooks:
            hook.remove()

        return used_layers

    def get_input_and_output_shapes_by_layer(self, model: nn.Module, input_shape):
        hooks = []
        input_shapes = {}
        output_shapes = {}
        name_count = {}  # regular dict instead of defaultdict


        def register_hook(name):
            def hook(module, input, output):
                if name not in name_count:
                    name_count[name] = 0
                count = name_count[name]
                key = f"{name}_{count}"

                input_shapes[key] = input[0].shape
                output_shapes[key] = output.shape if isinstance(output, torch.Tensor) else [o.shape for o in output]

                name_count[name] += 1
            return hook

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                hook = module.register_forward_hook(register_hook(name))
                hooks.append(hook)

        # Run a dummy input through the model
        model.eval()
        dummy_input = torch.randn(*input_shape)
        with torch.no_grad():
                _ = model(dummy_input)

        for h in hooks:
            h.remove()
        return input_shapes, output_shapes
    
    def clone_model_with_same_args(self, model):
        cls = type(model)
        sig = inspect.signature(cls.__init__)
        
        # Get constructor arguments (excluding 'self')
        kwargs = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if hasattr(model, name):
                kwargs[name] = getattr(model, name)

        return cls(**kwargs)

    def quantize_model(self, model, scale: int, rescale_config: dict = None):
        rescale_config = rescale_config or {}
        quantized_model = self.clone_model_with_same_args(model)
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
    
    def get_model_layers(self, model):
        from torch.fx import symbolic_trace, Tracer, GraphModule

        supported_layers = [QuantizedConv2d, QuantizedLinear, nn.ReLU, nn.MaxPool2d]
        supported_activations = {
            F.relu: nn.ReLU,
            # F.leaky_relu: nn.LeakyReLU,
            # F.silu: nn.SiLU,
            # F.gelu: nn.GELU,
            # F.sigmoid: nn.Sigmoid,
            # F.tanh: nn.Tanh,
        }

        class CustomTracer(Tracer):
            def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
                if isinstance(m, tuple(supported_layers)):  # Add other types here
                    return True
                return super().is_leaf_module(m, module_qualified_name)
        

        tracer = CustomTracer()
        graph = tracer.trace(model)
        traced = GraphModule(model, graph)

        node_names = []
        submodules = []
        layers = []
        index = 0
        for node in graph.nodes:
            print(f"{node.op}: {node.name}, target={node.target}, args={node.args}, kwargs={node.kwargs}")
            node_names.append(node.name)
            if node.op == "placeholder":
                pass

            if node.op == "call_function":
                if node.target in supported_activations:
                    activation_class = supported_activations[node.target]
                    name = node.name
                    submodules.append(activation_class())
                    layers.append(Layer(index, name, activation_class.__name__, 0))
                
            
            if node.op == "call_method":
                if node.target == "reshape":
                    # node.args[1:] contains the shape (e.g., (-1, 1568))
                    shape = node.args[1:]
                    name = node.name  # e.g., "reshape_1"
                    
                    # Create a custom reshape layer
                    class Reshape(nn.Module):
                        def __init__(self, shape):
                            super().__init__()
                            self.shape = shape

                        def forward(self, x):
                            return x.reshape(*self.shape)

                    submodules.append(Reshape(shape))
                if node.target == "flatten":
                    start_dim = node.args[1] if len(node.args) > 1 else 1
                    end_dim = node.args[2] if len(node.args) > 2 else -1
                    name = node.name

                    flatten = nn.Flatten(start_dim, end_dim)
                    submodules.append(flatten)
                    # print(node.args)
                    # submodules.append()
            
            if node.op == "call_module":

                submodule = traced.get_submodule(node.name)
                submodules.append(submodule)
            index +=1
        print(submodules)
        print(layers)

    def extract_scaled_modules(self, traced, scaled_types=None):
        """
        Extract all instances of scaled layers (e.g., scaledConv2, scaledFC) from a traced FX graph.

        Args:
            traced: torch.fx.GraphModule
            scaled_types: list of (name, type) tuples (e.g., [("scaled_conv", scaledConv2)])

        Returns:
            Dict with layer type keys and list of detected layers.
        """
        if scaled_types is None:
            scaled_types = []

        structure = {name: [] for name, _ in scaled_types}

        for node in traced.graph.nodes:
            print(node.op)
            if node.op == "call_module":
                submodule = traced.get_submodule(node.target)
                for type_name, type_class in scaled_types:
                    if isinstance(submodule, type_class):
                        entry = {
                            "name": node.name,
                            "class": type_class.__name__,
                            "scale": getattr(submodule, "scale", None),
                            "shift": getattr(submodule, "shift", None),
                            "has_bias": submodule.bias is not None,
                            "rescale_output": getattr(submodule, "rescale_output", None),
                        }
                        structure[type_name].append(entry)

        return structure


    def get_weights(self, flatten = False):
        if flatten:
            in_shape = [1, np.prod(self.input_shape)]
        else:
            in_shape = self.input_shape
        input_shapes, output_shapes = self.get_input_and_output_shapes_by_layer(self.quantized_model, in_shape)  # example input

        used_layers = self.get_used_layers(self.quantized_model, in_shape) 
        # Can combine the above into 1 function
        def to_tuple(x):
            return (x,) if isinstance(x, int) else tuple(x)
        weights = {}
        weights["scaling"] = self.scaling
        weights["scale_base"] = self.scale_base
        weights["input_shape"] = self.input_shape
        weights['layer_input_shapes'] = list(input_shapes.values())
        weights['layer_output_shapes'] = list(output_shapes.values())

        
        weights["layers"] = getattr(self, "layers", [])

        weights["not_rescale_layers"] = []
        rescaled_layers = getattr(self, "rescale_config", {})
        for key in rescaled_layers.keys():
            if not rescaled_layers[key]:
                weights["not_rescale_layers"].append(key)

        
        name_counters = {}

        for name, module in used_layers:
            # Set count to 0 if name not seen before, otherwise increment
            count = name_counters[name] if name in name_counters else 0
            disambiguated_name = f"{name}_{count}"
            name_counters[name] = count + 1

            if isinstance(module, (nn.Conv2d, QuantizedConv2d)):
                weights.setdefault("conv_weights", []).append(module.weight.tolist())
                weights.setdefault("conv_bias", []).append(module.bias.tolist())
                weights.setdefault("conv_strides", []).append(module.stride)
                weights.setdefault("conv_kernel_shape", []).append(module.kernel_size)
                weights.setdefault("conv_group", []).append([module.groups])
                weights.setdefault("conv_dilation", []).append(module.dilation)
                weights.setdefault("conv_pads", []).append(self.expand_padding(module.padding))
                weights.setdefault("conv_input_shape", []).append(input_shapes[disambiguated_name])

            if isinstance(module, (nn.Linear, QuantizedLinear)):
                weights.setdefault("fc_weights", []).append(module.weight.transpose(0, 1).tolist())
                weights.setdefault("fc_bias", []).append(module.bias.unsqueeze(0).tolist())

            if isinstance(module, nn.MaxPool2d):
                weights.setdefault("maxpool_kernel_size", []).append(to_tuple(module.kernel_size))
                weights.setdefault("maxpool_stride", []).append(to_tuple(module.stride))
                weights.setdefault("maxpool_dilation", []).append(to_tuple(module.dilation))
                weights.setdefault("maxpool_padding", []).append(to_tuple(module.padding))
                weights.setdefault("maxpool_ceil_mode", []).append(module.ceil_mode)
                weights.setdefault("maxpool_input_shape", []).append(to_tuple(input_shapes[disambiguated_name]))

            weights["output_shape"] = output_shapes[disambiguated_name]


        return weights
    
    def get_model(self, device):
        try:
            model = self.model_type(**getattr(self, 'model_params', {})).to(device)
            return model
        except AttributeError:
            raise NotImplementedError(f"Must specify the model type as a pytorch model (as variable self.model_type) in object {self.__class__.__name__}")
        except TypeError as e: 
            raise NotImplementedError(f"{e}. \n Must specify the model parameters of the pytorch model (as dictionary in self.model_params) in object {self.__class__.__name__}.")

    def get_model_and_quantize(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self.get_model(device)
        
        if hasattr(self, 'model_file_name'):
            print(f"Loading model from file {self.model_file_name}")
            checkpoint = torch.load(self.model_file_name, map_location=device)
            if "model_state_dict" in checkpoint.keys():
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
        else:
            print("Creating new model as no saved file path was specified")
        model.eval()
        self.model = model
        self.quantized_model = self.quantize_model(model, self.scale_base**self.scaling, rescale_config=getattr(self,"rescale_config", {}))
        self.quantized_model.eval()

    def test_accuracy(self):
        inputs = torch.rand(self.input_shape)*2 - 1
        print(self.model(inputs))
        q_inputs = inputs*(self.scale_base**self.scaling)
        print(self.quantized_model(q_inputs)/(self.scale_base**(2*self.scaling)))

    def get_outputs(self, inputs):
        return self.quantized_model(inputs)




# TODO CHANGE THIS NESTED STRUCTURE, DONE FOR EASE FOR NOW, BUT IT NEEDS IMPROVEMENT
class ZKTorchModel(PytorchConverter, ZKModelBase):
    def __init__(self):
        raise NotImplementedError("Must implement __init__")
