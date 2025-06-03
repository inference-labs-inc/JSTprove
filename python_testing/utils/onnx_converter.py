from dataclasses import dataclass, fields
import inspect
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import onnx
import onnxruntime as ort
import os


from python_testing.circuit_components.circuit_helpers import RunType
from python_testing.utils.pytorch_partial_models import QuantizedConv2d, QuantizedLinear
from python_testing.utils.model_converter import ZKModelBase, ModelConverter

class ONNXConverter(ModelConverter):

    # For saving and loading: https://onnx.ai/onnx/intro/python.html, larger models will require a different structure
    def save_model(self, file_path: str):
        onnx.save(self.model, file_path)
    
    def load_model(self, file_path: str, model_type = None):
        onnx_model = onnx.load(file_path)
        # Fix, can remove this next line 
        onnx.checker.check_model(onnx_model)
        self.model = onnx_model

    def save_quantized_model(self, file_path: str):
        onnx.save(self.quantized_model, file_path)

    # Not sure this is ideal
    def load_quantized_model(self, file_path: str):
        onnx_model = onnx.load(file_path)
        # Fix, can remove this next line 
        onnx.checker.check_model(onnx_model)
        self.quantized_model = onnx_model
        print(os.path.exists(file_path))
        self.ort_sess =  ort.InferenceSession(file_path, providers=["CPUExecutionProvider"])


    # def expand_padding(self, padding_2):
    #     if len(padding_2) != 2:
    #         raise(ValueError("Expand padding requires initial padding of dimension 2"))
    #     pad_h, pad_w = padding_2
    #     return (pad_w, pad_w, pad_h, pad_h)
    
    def get_used_layers(self, model, input_shape):
        pass

    def get_input_and_output_shapes_by_layer(self, model, input_shape):
        pass
    
    def quantize_model(self, model, scale: int, rescale_config: dict = None):
        pass
    

    # TODO JG suggestion - can maybe make the layers into a factory here, similar to how its done in Rust? Can refactor to this later imo.
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

    def get_model_and_quantize(self):
        pass

    def test_accuracy(self):
        inputs = torch.rand(self.input_shape)*2 - 1
        session = ort.InferenceSession(self.model_file_name)
        outputs = session.run(None, {"input": inputs})
        print(outputs)

        q_inputs = inputs*(self.scale_base**self.scaling)

        session = ort.InferenceSession(self.model_file_name)
        outputs = session.run(None, {"input": inputs})

        print(outputs/(self.scale_base**(2*self.scaling)))

    def get_outputs(self, inputs):
        input_name = self.ort_sess.get_inputs()[0].name
        output_name = self.ort_sess.get_outputs()[0].name
        return self.ort_sess.run([output_name], {input_name: inputs})