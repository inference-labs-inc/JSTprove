import json
from typing import Optional
import torch
import onnx
import onnxruntime as ort


from python_testing.circuit_components.circuit_helpers import Circuit, RunType
from python_testing.utils.run_proofs import ZKProofSystems
from python_testing.utils.helper_functions import prepare_io_files
from types import SimpleNamespace
from abc import ABC, abstractmethod



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


    # TODO this should be split to onnx and pytorch I think
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
    
    # def get_outputs(self, inputs):
    #     return self.quantized_model(inputs)
    
    def get_inputs(self, file_path:str = None, is_scaled = False):
        if file_path == None:
            return self.create_new_inputs()
        if hasattr(self, "input_shape"):
            return self.get_inputs_from_file(file_path, is_scaled=is_scaled).reshape(self.input_shape)
        else:
            raise NotImplementedError("Must define attribute input_shape")
    
    def create_new_inputs(self):
        # print(self.scale_base, self.scaling, self.scale_base**self.scaling)
        return torch.mul(torch.rand(self.input_shape)*2 - 1, self.scale_base**self.scaling).long()

    def format_inputs(self, inputs):
        return {"input": inputs.long().tolist()}
    
    def format_outputs(self, outputs):
        if hasattr(self, "scaling") and hasattr(self, "scale_base"):
            # Must change how rescaled_outputs is specified TODO
            return {"output": outputs.long().tolist(), "rescaled_output": torch.div(outputs, self.scale_base**(2*self.scaling)).tolist()}
        return {"output": outputs.long().tolist()}
    
    def format_inputs_outputs(self, inputs, outputs):
        return self.format_inputs(inputs), self.format_outputs(outputs)

class ZKModelBase(GeneralLayerFunctions, Circuit):
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
                     write_json: Optional[bool] = False,
                     bench = False,
                     quantized_path = None):
        """Simulates running the model by passing inputs through layers with weights."""
        print("Running circuit...")

        print(circuit_name, circuit_path)

        if not weights_path:
            weights_path = f"weights/{circuit_name}_weights.json"

        self.parse_proof_run_type(witness_file, input_file, proof_file, public_path, verification_key, circuit_name, circuit_path, proof_system, output_file, weights_path, quantized_path, run_type, dev_mode, ecc, write_json, bench)


class ModelConverter(ABC):

    @abstractmethod
    def save_model(self, file_path: str):
        pass
    
    @abstractmethod
    def load_model(self, file_path: str, model_type = None):
        pass

    @abstractmethod
    def save_quantized_model(self, file_path: str):
        pass

    @abstractmethod
    def load_quantized_model(self, file_path: str):
        pass


    # def expand_padding(self, padding_2):
    #     if len(padding_2) != 2:
    #         raise(ValueError("Expand padding requires initial padding of dimension 2"))
    #     pad_h, pad_w = padding_2
    #     return (pad_w, pad_w, pad_h, pad_h)
    
    @abstractmethod
    def get_used_layers(self, model, input_shape):
        pass

    @abstractmethod
    def get_input_and_output_shapes_by_layer(self, model, input_shape):
        pass
    
    @abstractmethod
    def quantize_model(self, model, scale: int, rescale_config: dict = None):
        pass
    

    # TODO JG suggestion - can maybe make the layers into a factory here, similar to how its done in Rust? Can refactor to this later imo.
    @abstractmethod
    def get_weights(self, flatten = False):
        pass
        # if flatten:
        #     in_shape = [1, np.prod(self.input_shape)]
        # else:
        #     in_shape = self.input_shape
        # input_shapes, output_shapes = self.get_input_and_output_shapes_by_layer(self.quantized_model, in_shape)  # example input

        # used_layers = self.get_used_layers(self.quantized_model, in_shape) 
        # # Can combine the above into 1 function
        # def to_tuple(x):
        #     return (x,) if isinstance(x, int) else tuple(x)
        # weights = {}
        # weights["scaling"] = self.scaling
        # weights["scale_base"] = self.scale_base
        # weights["input_shape"] = self.input_shape
        # weights['layer_input_shapes'] = list(input_shapes.values())
        # weights['layer_output_shapes'] = list(output_shapes.values())

        
        # weights["layers"] = getattr(self, "layers", [])

        # weights["not_rescale_layers"] = []
        # rescaled_layers = getattr(self, "rescale_config", {})
        # for key in rescaled_layers.keys():
        #     if not rescaled_layers[key]:
        #         weights["not_rescale_layers"].append(key)

        
        # name_counters = {}

        # for name, module in used_layers:
        #     # Set count to 0 if name not seen before, otherwise increment
        #     count = name_counters[name] if name in name_counters else 0
        #     disambiguated_name = f"{name}_{count}"
        #     name_counters[name] = count + 1

        #     if isinstance(module, (nn.Conv2d, QuantizedConv2d)):
        #         weights.setdefault("conv_weights", []).append(module.weight.tolist())
        #         weights.setdefault("conv_bias", []).append(module.bias.tolist())
        #         weights.setdefault("conv_strides", []).append(module.stride)
        #         weights.setdefault("conv_kernel_shape", []).append(module.kernel_size)
        #         weights.setdefault("conv_group", []).append([module.groups])
        #         weights.setdefault("conv_dilation", []).append(module.dilation)
        #         weights.setdefault("conv_pads", []).append(self.expand_padding(module.padding))
        #         weights.setdefault("conv_input_shape", []).append(input_shapes[disambiguated_name])

        #     if isinstance(module, (nn.Linear, QuantizedLinear)):
        #         weights.setdefault("fc_weights", []).append(module.weight.transpose(0, 1).tolist())
        #         weights.setdefault("fc_bias", []).append(module.bias.unsqueeze(0).tolist())

        #     if isinstance(module, nn.MaxPool2d):
        #         weights.setdefault("maxpool_kernel_size", []).append(to_tuple(module.kernel_size))
        #         weights.setdefault("maxpool_stride", []).append(to_tuple(module.stride))
        #         weights.setdefault("maxpool_dilation", []).append(to_tuple(module.dilation))
        #         weights.setdefault("maxpool_padding", []).append(to_tuple(module.padding))
        #         weights.setdefault("maxpool_ceil_mode", []).append(module.ceil_mode)
        #         weights.setdefault("maxpool_input_shape", []).append(to_tuple(input_shapes[disambiguated_name]))

        #     weights["output_shape"] = output_shapes[disambiguated_name]


        # return weights
    
    # @abstractmethod
    # def get_model(self, device):
    #     pass

    @abstractmethod
    def get_model_and_quantize(self):
        pass

    @abstractmethod
    def test_accuracy(self):
        pass

    @abstractmethod
    def get_outputs(self, inputs):
        pass