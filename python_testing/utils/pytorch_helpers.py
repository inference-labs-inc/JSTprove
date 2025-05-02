import inspect
import json
from typing import Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
import onnx
import onnxruntime as ort

from python_testing.circuit_components.circuit_helpers import Circuit, RunType
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import get_files, to_json, prove_and_verify,prepare_io_files
from types import SimpleNamespace
from python_testing.utils.pytorch_partial_models import QuantizedConv2d, QuantizedLinear

class GeneralLayerFunctions():
    def check_4d_eq(self, input_tensor_1, input_tensor_2):
        for i in range(input_tensor_1.shape[0]):
            for j in range(input_tensor_1.shape[1]):
                for k in range(input_tensor_1.shape[2]):
                    for l in range(input_tensor_1.shape[3]):
                        assert(abs(input_tensor_1[i][j][k][l] - input_tensor_2[i][j][k][l]) < 1)

    def weights_onnx_to_torch_format(self, onnx_model):
        w_and_b = {}
        for i in onnx_model.graph.initializer:
            layer_name, param_type = i.name.split(".")  # Split into layer name and param type
            if layer_name not in w_and_b:
                w_and_b[layer_name] = {}

            w_and_b[layer_name][param_type] = torch.tensor(onnx.numpy_helper.to_array(i))

        for e in w_and_b.keys():
            w_and_b[e] = SimpleNamespace(**w_and_b[e])
        return w_and_b

    def check_2d_eq(self, input_tensor_1, input_tensor_2):
        for i in range(input_tensor_1.shape[0]):
            for j in range(input_tensor_1.shape[1]):
                assert(abs(input_tensor_1[i][j] -  input_tensor_2[i][j]) < 1)

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
            return data["input"]



    def read_output(self, model, input_data, is_torch = True):
        """Reads the outputs for each layer of the model from text files."""
        if is_torch:
            with torch.no_grad():  # Disable gradient calculation during inference
                output = model(torch.as_tensor(input_data))
                return output
        else:
            session = ort.InferenceSession(model)

            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: input_data})
            return outputs
        
    def get_inputs_from_file(self, file_name, is_scaled: bool = False):
        inputs = self.read_input(file_name)
        if is_scaled:
            out =  torch.as_tensor(inputs).long()
        else:
            out =  torch.mul(torch.as_tensor(inputs),self.scale_base**self.scaling).long()

        if hasattr(self, "input_shape"):
            out = out.reshape(self.input_shape)
        return out
    
    def get_outputs(self, inputs):
        return self.quantized_model(inputs)
    
    def get_inputs(self, file_path:str = None, is_scaled = False):
        if file_path == None:
            return self.create_new_inputs()
        if hasattr(self, "input_shape"):
            return self.get_inputs_from_file(file_path, is_scaled=is_scaled).reshape(self.input_shape)
        else:
            raise NotImplementedError("Must define attribute input_shape")
    
    def create_new_inputs(self):
        print(self.scale_base, self.scaling, self.scale_base**self.scaling)
        return torch.mul(torch.rand(self.input_shape)*2 - 1, self.scale_base**self.scaling).long()

    def format_inputs(self, inputs):
        return {"input": inputs.long().tolist()}
    
    def format_outputs(self, outputs):
        if hasattr(self, "scaling") and hasattr(self, "scale_base"):
            return {"output": outputs.long().tolist(), "rescaled_output": torch.div(outputs, self.scale_base**(2*self.scaling)).tolist()}
        return {"output": outputs.long().tolist()}
    
    def format_inputs_outputs(self, inputs, outputs):
        return self.format_inputs(inputs), self.format_outputs(outputs)

class PytorchConverter():
    def save_model(self, file_path: str):
        torch.save(self.model.state_dict(), file_path)
    
    def load_model(self, file_path: str):
        self.model.load_state_dict(torch.load(file_path))

    def save_quantized_model(self, file_path: str):
        # torch.save(self.quantized_model.state_dict(), file_path)
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
    
    def get_input_shapes_by_layer(self, model: nn.Module, input_shape):
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

    def get_weights(self):
        input_shapes = self.get_input_shapes_by_layer(self.quantized_model, self.input_shape)  # example input
        used_layers = self.get_used_layers(self.quantized_model, self.input_shape) 
        # Can combine the above into 1 function
        
        weights = {}
        weights["scaling"] = self.scaling
        weights["sacle_base"] = self.scale_base

        
        for name, module in used_layers:

            if isinstance(module, (nn.Conv2d, QuantizedConv2d)):
                weights.setdefault("conv_weights", []).append(module.weight.tolist())
                weights.setdefault("conv_bias", []).append(module.bias.tolist())
                weights.setdefault("conv_strides", []).append(module.stride)
                weights.setdefault("conv_kernel_shape", []).append(module.kernel_size)
                weights.setdefault("conv_group", []).append([module.groups])
                weights.setdefault("conv_dilation", []).append(module.dilation)
                weights.setdefault("conv_pads", []).append(self.expand_padding(module.padding))
                weights.setdefault("conv_input_shape", []).append(input_shapes[name])
            
            if isinstance(module, (nn.Linear, QuantizedLinear)):
                weights.setdefault("fc_weights", []).append(module.weight.transpose(0, 1).tolist())
                weights.setdefault("fc_bias", []).append(module.bias.unsqueeze(0).tolist())

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
            model.load_state_dict(checkpoint["model_state_dict"])
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


# TODO CHANGE THIS NESTED STRUCTURE, DONE FOR EASE FOR NOW, BUT IT NEEDS IMPROVEMENT
class ZKModel(PytorchConverter, GeneralLayerFunctions, Circuit):
    def __init__(self):
        raise NotImplementedError("Must implement __init__")
    
    
    @prepare_io_files
    def base_testing(self, run_type=RunType.BASE_TESTING, 
                     witness_file=None, input_file=None, proof_file=None, public_path=None, 
                     verification_key=None, circuit_name=None, weights_path=None, output_file=None,
                     proof_system: ZKProofSystems = ZKProofSystems.Expander,
                     dev_mode = False,
                     ecc = True,
                     circuit_path: Optional[str] = None,
                     write_json: Optional[bool] = False):
        """Simulates running the model by passing inputs through layers with weights."""
        print("Running circuit...")

        print(circuit_name, circuit_path)

        if not weights_path:
            weights_path = f"weights/{circuit_name}_weights.json"

        self.parse_proof_run_type(witness_file, input_file, proof_file, public_path, verification_key, circuit_name, circuit_path, proof_system, output_file, weights_path, run_type, dev_mode, ecc, write_json)