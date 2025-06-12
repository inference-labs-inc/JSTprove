from dataclasses import dataclass, fields
import inspect
import json
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import onnx
from onnx import NodeProto, TensorProto, shape_inference

import onnxruntime as ort
import os


from python_testing.circuit_components.circuit_helpers import RunType
from python_testing.utils.onnx_helpers import dims_prod, extract_shape_dict, parse_attributes
from python_testing.utils.onnx_op_quantizer import ONNXOpQuantizer
from python_testing.utils.pytorch_partial_models import QuantizedConv2d, QuantizedLinear
from python_testing.utils.model_converter import ZKModelBase, ModelConverter


import model_analyzer
# @dataclass
# class ONNXLayerConstants:
#     input_index: int
#     name: str
#     value: Dict[str, List[int]]

@dataclass
class ONNXLayer:
    id: int
    name: str
    op_type: str #This is the operation type. eg. "Conv" for convolution layers
    inputs: List[str] #This will be a list of other input layers. the str inside the list, can either be the name or id. Unsure of the best way to tackle this
    outputs: List[str]
    shape: Dict[str, List[int]] # This will be a hashmap where the key is the output layer and the List[int] is the shape of that output layer
    tensor: Optional[List] # This will be empty for layers, and will contain the weights or biases, for Const nodes
    params: Optional[Dict] # Most layers will have params attached to them. For eg, params for conv would be - dilation, kernel_shape, pad, strides, group,...
    opset_version_number: int # This is the version number of the operation used. So far this is not used in rust, but I think for infrastructure purposes we can include

class ONNXConverter(ModelConverter):
    def __init__(self):
        super().__init__()
        self.op_quantizer = ONNXOpQuantizer()

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
        # https://github.com/sonos/tract/blob/main/api/py/docs/index.md
        # Can use tract to run model instead

    # def expand_padding(self, padding_2):
    #     if len(padding_2) != 2:
    #         raise(ValueError("Expand padding requires initial padding of dimension 2"))
    #     pad_h, pad_w = padding_2
    #     return (pad_w, pad_w, pad_h, pad_h)
    
    def analyze_layers_json(self, model):
        layers = model_analyzer.analyze_model_json("./models_onnx/doom.onnx")
        layers = json.loads(layers)
        # layers = [ONNXLayer(**l) for l in layers]

        # # sort layers
        # layers = sorted(layers, key=lambda ONNXLayer: ONNXLayer.id) 
        # print([(l.name, l.id, l.kind, l.inputs, l.outputs, l.shape) for l in layers])

        # for l in layers:
        #     if l.kind == {'type': 'Const'}:
        #         print(l.tensor)
        #         break

    def analyze_layers(self, path):
        # model = tract.onnx().model_for_path("./mobilenetv2-7.onnx").into_optimized().into_runnable()
        # tract_model = tract.onnx().model_for_path("./models_onnx/doom.onnx")
        # layers = model_analyzer.analyze_model("./models_onnx/doom.onnx")
        path  ="./models_onnx/doom.onnx"
        id_count = 0
        model = onnx.load(path)
        # Fix, can remove this next line 
        onnx.checker.check_model(model)

        # Check the model and print Y"s shape information
        onnx.checker.check_model(model)
        print(f"Before shape inference, the shape info of Y is:\n{model.graph.value_info}")

        # To be used if I need batch size
        # for input_tensor in model.graph.input:
            # if input_tensor.name == "input":  # replace with your input name
            # input_tensor.type.tensor_type.shape.dim[0].dim_value = getattr(self, "batch_size", 1)  # Set batch size to 1

        # Apply shape inference on the model
        inferred_model = shape_inference.infer_shapes(model)

        # Check the model and print Y"s shape information
        onnx.checker.check_model(inferred_model)
        # print(f"After shape inference, the shape info of Y is:\n{inferred_model.graph.value_info}")
        

        domain_to_version = {opset.domain: opset.version for opset in model.opset_import}
        print(inferred_model.graph.value_info[0].name)
        
        inferred_model = shape_inference.infer_shapes(model)
        output_name_to_shape = extract_shape_dict(inferred_model)
        id_count = 0
        architecture = self.get_model_architecture(model, output_name_to_shape, id_count, domain_to_version)
        w_and_b = self.get_model_w_and_b(model, output_name_to_shape, id_count, domain_to_version)

        new_model = self.quantize_model(model, 2, 21)
        onnx.checker.check_model(new_model)

        with open("model.onnx", "wb") as f:
            f.write(new_model.SerializeToString())

        model = onnx.load("model.onnx")
        onnx.checker.check_model(model)  # This throws a descriptive error

        inputs = torch.rand([1,4,28,28])
        self.run_model_onnx_runtime(path, inputs)

        self.run_model_onnx_runtime("model.onnx", inputs)


        return (architecture, w_and_b)
    
    
    
    def run_model_onnx_runtime(self, path: str, input: torch.Tensor):
        onnx_model = onnx.load(path)
        # Fix, can remove this next line 
        onnx.checker.check_model(onnx_model)
        ort_sess =  ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        input_name = ort_sess.get_inputs()[0].name
        output_name = ort_sess.get_outputs()[0].name
        outputs = ort_sess.run([output_name], {input_name: input.numpy()})
        print(outputs)
        # This can help:
        # for constant in onnx_model.graph.initializer:
        #     constant_dtype = constant.data_type
        #     np_data = onnx.numpy_helper.to_array(constant, constant_dtype)
        #     np_data
        #     print(constant.name, np_data.shape)
        # for layer in onnx_model.graph.node:
        #     print(layer.input, layer.op_type, layer.name)
        # self.get_model_architecture(onnx_model)
        return outputs

    
    def get_model_architecture(self, model: onnx.ModelProto, output_name_to_shape: Dict[str, List[int]], id_count: int = 0, domain_to_version: dict[str, int] = None):
        layers = []
        # Check the model and print Y"s shape information
        for (idx, node) in enumerate(model.graph.node):
            layer = self.analyze_layer(node, output_name_to_shape, id_count, domain_to_version )
            layers.append(layer)
            id_count += 1
        return layers
    
    def get_model_w_and_b(self, model: onnx.ModelProto, output_name_to_shape: Dict[str, List[int]], id_count: int = 0, domain_to_version: dict[str, int] = None):
        layers = []
        # Check the model and print Y"s shape information
        for (idx, node) in enumerate(model.graph.initializer):
            layer = self.analyze_constant(node, output_name_to_shape, id_count, domain_to_version )
            layers.append(layer)
            id_count += 1
        return layers
        

    def analyze_layer(self, node: NodeProto, output_name_to_shape: Dict[str, List[int]], id_count: int = -1, domain_to_version: dict[str, int] = None) -> List[ONNXLayer]:
        name = node.name
        id = id_count
        id_count += 1
        op_type = node.op_type
        inputs = node.input
        outputs = node.output
        domain = node.domain if node.domain else "ai.onnx"
        opset_version = domain_to_version.get(node.domain, "unknown") if domain_to_version else -1
        params = parse_attributes(node.attribute)

        # 💡 Extract output shapes
        output_shapes = {
                out_name: output_name_to_shape.get(out_name, []) for out_name in outputs
            }
        layer = ONNXLayer(
                id = id, 
                name = name,
                op_type = op_type,
                inputs = inputs,
                outputs = outputs,
                shape = output_shapes,
                params = params,
                opset_version_number = opset_version,
                tensor = None,
            )
        return layer
    
    def analyze_constant(self, node: TensorProto, output_name_to_shape: Dict[str, List[int]], id_count: int = -1, domain_to_version: dict[str, int] = None) -> List[ONNXLayer]:
        name = node.name
        id = id_count
        id_count += 1
        op_type = "Const"
        inputs = []
        outputs = []
        domain = "ai.onnx"
        opset_version = -1
        params = {}
        constant_dtype = node.data_type
        np_data = onnx.numpy_helper.to_array(node, constant_dtype)
            # 💡 Extract output shapes
        output_shapes = {
                out_name: output_name_to_shape.get(out_name, []) for out_name in outputs
            }
        layer = ONNXLayer(
                id = id, 
                name = name,
                op_type = op_type,
                inputs = inputs,
                outputs = outputs,
                shape = output_shapes,
                params = params,
                opset_version_number = opset_version,
                tensor = np_data,
            )
        return layer

    def get_used_layers(self, model=None, input_shape=None):
        # return super().get_used_layers(model, input_shape)
        pass

    



    def get_input_and_output_shapes_by_layer(self, model, input_shape):
        pass
    
    def quantize_model(self, unscaled_model: onnx.ModelProto, scale_base: int,  scale: int, rescale_config: dict = None):
        '''
        1. Read in the model and layers + analyze
        2. Look for layers that need quantizing 
        3. Convert layer to quantized version
        4. insert quantized version back into the model
        '''
        model = unscaled_model
        new_nodes = []
        initializer_map = {init.name: init for init in model.graph.initializer}
        for node in model.graph.node:
            rescale = rescale_config.get(node.name, False) if rescale_config else True
            quant_nodes = self.quantize_layer(node, rescale, model, scale, scale_base, initializer_map)
            if isinstance(quant_nodes, list):
                new_nodes.extend(quant_nodes)
            else:
                new_nodes.append(quant_nodes)

        model.graph.ClearField("node")
        model.graph.node.extend(new_nodes)

        used_initializer_names = set()
        for node in model.graph.node:
            used_initializer_names.update(node.input)

        # Keep only initializers actually used
        kept_initializers = [
            tensor for tensor in model.graph.initializer
            if tensor.name in used_initializer_names
        ]

        model.graph.ClearField("initializer")
        model.graph.initializer.extend(kept_initializers)

        # for (idx, initializer) in enumerate(model.graph.initializer):
        #     layer = self.quantize_constant(initializer, scale_base, scale)

        model.graph.initializer.extend(self.op_quantizer.new_initializers)

        self.op_quantizer.new_initializers = []
        for layer in model.graph.node:
            print(layer.name, layer.op_type, layer.input, layer.output)
            

        for layer in model.graph.initializer:
            print(layer.name)
        
        return model
        

    def quantize_layer(self, node: onnx.NodeProto, rescale: bool, model: onnx.ModelProto, scale: int, scale_base: int, initializer_map: dict[str, onnx.TensorProto]) -> onnx.NodeProto:
        quant_nodes = self.op_quantizer.quantize(node, rescale, model.graph, scale, scale_base, initializer_map)
        return quant_nodes

    
    # def quantize_constant(self, initializer: TensorProto, scale_base: int, scale: int):
    #     # if initializer.data_type == onnx.TensorProto.DataType.FLOAT:
    #     #     for i in range(dims_prod(initializer.dims)):
    #     #         initializer.float_data[i] = initializer.float_data[i]*(scale_base**scale)

    #     if initializer.data_type == onnx.TensorProto.FLOAT:
    #         factor = scale_base ** scale

    #         # Read full tensor into numpy
    #         arr = to_array(initializer).astype(np.float32)

    #         # Apply quantization (scaling)
    #         arr *= factor

    #         # Overwrite initializer with updated values
    #         new_tensor = from_array(arr, name=initializer.name)

    #         # Copy updated fields back (in-place mutation)
    #         initializer.ClearField('float_data')
    #         initializer.ClearField('raw_data')
    #         initializer.raw_data = new_tensor.raw_data
            
        
    

    # TODO JG suggestion - can maybe make the layers into a factory here, similar to how its done in Rust? Can refactor to this later imo.
    def get_weights(self, flatten = False):
        '''
        1. Analyze the model for architecture + w & b
        2. Put arch into format to be read by ECC circuit builder
        3. Put w + b into format to be read by ECC circuit builder
        '''
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

    # def run_quantized_model_inference_tract(self, path, input):
    #     outputs = model_analyzer.run_model_from_f32(path)
    #     print(outputs)
    #     return outputs
    
    # def run_model_inference_tract(self, path: str, input: torch.Tensor):
    #     outputs = model_analyzer.run_model_from_f32(path,input.flatten().tolist(), input.shape)
    #     print(outputs)
    #     return outputs
    
    # def get_used_layers_tract(self, model):
    #     architecture = model_analyzer.get_architecture(model)
    #     # # sort layers
    #     layers = sorted(architecture, key=lambda ONNXLayer: ONNXLayer.id) 
    #     # print([(l.name, l.id, l.kind, l.inputs, l.outputs, f"Tensor length {len(l.tensor) if l.tensor else None}", l.shape, json.loads(l.params.to_dict()) if l.params else None) for l in layers])
    #     return layers
    
    
    # def get_w_and_b_tract(self, model):
    #     w_and_b = model_analyzer.get_w_and_b(model)
    #     # # sort layers
    #     layers = sorted(w_and_b, key=lambda ONNXLayer: ONNXLayer.id) 
    #     # print([(l.name, l.id, l.kind, l.inputs, l.outputs, f"Tensor length {l.tensor.shape if l.tensor else None}", l.shape) for l in layers])
    #     return layers

    def get_model_and_quantize(self):
        pass

    def test_accuracy(self):
        inputs = torch.rand(self.input_shape)*2 - 1
        session = ort.InferenceSession(self.model_file_name)
        outputs = session.run(None, {"input": inputs})
        print(outputs)

        q_inputs = inputs*(self.scale_base**self.scaling)

        session = ort.InferenceSession(self.model_file_name)
        outputs = session.run(None, {"input": q_inputs})

        print(outputs/(self.scale_base**(2*self.scaling)))

    def get_outputs(self, inputs):
        input_name = self.ort_sess.get_inputs()[0].name
        output_name = self.ort_sess.get_outputs()[0].name
        return self.ort_sess.run([output_name], {input_name: inputs})