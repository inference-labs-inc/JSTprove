import copy
from dataclasses import asdict, dataclass, fields, is_dataclass
import inspect
import json
import sys
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import onnx
from onnx import NodeProto, TensorProto, shape_inference, helper, numpy_helper

import onnxruntime as ort
import os


from python_testing.circuit_components.circuit_helpers import RunType
from python_testing.utils.onnx_helpers import dims_prod, extract_shape_dict, get_input_shapes, parse_attributes
from python_testing.utils.onnx_op_quantizer import ONNXOpQuantizer
from python_testing.utils.pytorch_partial_models import QuantizedConv2d, QuantizedLinear
from python_testing.utils.model_converter import ZKModelBase, ModelConverter

from onnxruntime import InferenceSession, SessionOptions
from onnxruntime_extensions import get_library_path, OrtPyFunction
# from python_testing.utils.onnx_custom_ops import conv

# from python_testing.utils.onnx_custom_ops import *

# from python_testing.utils.onnx_custom_ops.conv import int64_conv
# from python_testing.utils.onnx_custom_ops.gemm import int64_gemm7
# from python_testing.utils.onnx_custom_ops.maxpool import int64_maxpool
# from python_testing.utils.onnx_custom_ops.relu import int64_relu

import python_testing.utils.onnx_custom_ops






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

@dataclass
class ONNXIO:
    name: str
    elem_type: int
    shape: List[int]


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

        self.required_keys = [input.name for input in onnx_model.graph.input]
        self.input_shape = get_input_shapes(onnx_model)
        return self.model
    

    def save_quantized_model(self, file_path: str):
        onnx.save(self.quantized_model, file_path)

    # Not sure this is ideal
    def load_quantized_model(self, file_path: str):
        # May be able to remove next few lines...
        print(file_path)
        onnx_model = onnx.load(file_path)
        custom_domain = onnx.helper.make_operatorsetid(domain="ai.onnx.contrib", version=1)
        onnx_model.opset_import.append(custom_domain)
        # Fix, can remove this next line 
        onnx.checker.check_model(onnx_model)
        self.quantized_model = onnx_model
        opts = SessionOptions()
        opts.register_custom_ops_library(get_library_path())
        self.ort_sess =  ort.InferenceSession(file_path, opts, providers=["CPUExecutionProvider"])
        self.required_keys = [input.name for input in onnx_model.graph.input]
        self.input_shape = get_input_shapes(onnx_model)

        self.quantized_model_path = file_path
    
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

    def analyze_layers(self, output_name_to_shape = None):
        # model = tract.onnx().model_for_path("./mobilenetv2-7.onnx").into_optimized().into_runnable()
        # tract_model = tract.onnx().model_for_path("./models_onnx/doom.onnx")
        # layers = model_analyzer.analyze_model("./models_onnx/doom.onnx")
        # path  ="./models_onnx/doom.onnx"
        id_count = 0

        

        # We may want to add our own model checker here, in order to confirm that the model layers meet our specs - layer types etc.

        # To be used if I need batch size
        # for input_tensor in model.graph.input:
            # if input_tensor.name == "input":  # replace with your input name
            # input_tensor.type.tensor_type.shape.dim[0].dim_value = getattr(self, "batch_size", 1)  # Set batch size to 1

        # Apply shape inference on the model
        if not output_name_to_shape:
            inferred_model = shape_inference.infer_shapes(self.model) 

            # Check the model and print Y"s shape information
            onnx.checker.check_model(inferred_model)
            output_name_to_shape = extract_shape_dict(inferred_model)

        # print(f"After shape inference, the shape info of Y is:\n{inferred_model.graph.value_info}")
        

        domain_to_version = {opset.domain: opset.version for opset in self.model.opset_import}
        
        id_count = 0
        architecture = self.get_model_architecture(self.model, output_name_to_shape, id_count, domain_to_version)
        w_and_b = self.get_model_w_and_b(self.model, output_name_to_shape, id_count, domain_to_version)
        return (architecture, w_and_b)
    
    
    
    def run_model_onnx_runtime(self, path: str, input: torch.Tensor):
        onnx_model = onnx.load(path)
        # Fix, can remove this next line 
        onnx.checker.check_model(onnx_model)
        

        if not hasattr(self, "ort_sess"):
            opts = SessionOptions()
            opts.register_custom_ops_library(get_library_path())
            ort_sess =  ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        else:
            # ort_sess = self.ort_sess
            opts = SessionOptions()
            opts.register_custom_ops_library(get_library_path())
            ort_sess =  ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        input_name = ort_sess.get_inputs()[0].name
        output_name = ort_sess.get_outputs()[0].name
        if ort_sess.get_inputs()[0].type == "tensor(double)":
            outputs = ort_sess.run([output_name], {input_name: np.asarray(input).astype(np.float64)})
        else:
            outputs = ort_sess.run([output_name], {input_name: np.asarray(input)})



        # intermediate_names = ["conv1.weight_scaled_cast", "conv1.bias_scaled_cast"]

        # results = ort_sess.run(intermediate_names, {input_name: np.asarray(input)})

        # with open('debug_data.json', 'w') as f:
        #     json.dump(results, f)

        # sys.exit()
        
        return outputs

        # This can help:
        # for constant in onnx_model.graph.initializer:
        #     constant_dtype = constant.data_type
        #     np_data = onnx.numpy_helper.to_array(constant, constant_dtype)
        #     np_data
        #     print(constant.name, np_data.shape)
        # for layer in onnx_model.graph.node:
        #     print(layer.input, layer.op_type, layer.name)
        # self.get_model_architecture(onnx_model)

    
    def get_model_architecture(self, model: onnx.ModelProto, output_name_to_shape: Dict[str, List[int]], id_count: int = 0, domain_to_version: dict[str, int] = None):
        layers = []
        constant_values = {}
        # First pass: collect constant nodes
        for node in model.graph.node:
            if node.op_type == "Constant":
                print(node)
                for attr in node.attribute:
                    if attr.name == "value":
                        tensor = attr.t
                        const_value = numpy_helper.to_array(tensor)
                        constant_values[node.output[0]] = const_value

        # Map output name to shape (assumed provided from previous analysis)
        layers = []
        id_count = 0

        # Second pass: analyze layers
        for (idx, node) in enumerate(model.graph.node):
            if node.op_type == "Constant":
                continue  # Already processed

            layer = self.analyze_layer(node, output_name_to_shape, id_count, domain_to_version)
            print(layer.shape)

            # Attach constant inputs as parameters
            for input_name in node.input:
                if input_name in constant_values:
                    print(layer.params)
                    if not hasattr(layer, 'params'):
                        layer.params = {}
                    result = constant_values[input_name]
                    if isinstance(result, np.ndarray) or isinstance(result, torch.Tensor):
                        layer.params[input_name] = result.tolist()
                    else:
                        layer.params[input_name] = constant_values[input_name]
                    print(layer.params)

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
                inputs = list(inputs),
                outputs = list(outputs),
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
        # Can do this step in rust potentially to keep file sizes low if needed
        np_data = onnx.numpy_helper.to_array(node, constant_dtype)
            # 💡 Extract output shapes
        output_shapes = {
                out_name: output_name_to_shape.get(out_name, []) for out_name in outputs
            }
        layer = ONNXLayer(
                id = id, 
                name = name,
                op_type = op_type,
                inputs = list(inputs),
                outputs = list(outputs),
                shape = output_shapes,
                params = params,
                opset_version_number = opset_version,
                tensor = np_data.tolist(),
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
        
        model = copy.deepcopy(unscaled_model)
        initializer_map = {init.name: init for init in model.graph.initializer}
        input_names = [inp.name for inp in unscaled_model.graph.input]



        new_nodes = []
        for i, name in enumerate(input_names):
            output_name, mul_node, floor_node, cast_to_int64 = self.quantize_input(name, self.op_quantizer, scale_base, scale)
            new_nodes.append(mul_node)
            # new_nodes.append(floor_node)
            new_nodes.append(cast_to_int64)
            for node in model.graph.node:
                for idx, inp in enumerate(node.input):
                    if inp == name:
                        node.input[idx] = output_name
        for input_tensor in model.graph.input:
            tensor_type = input_tensor.type.tensor_type
            # Only change float32 (type = 1)
            if tensor_type.elem_type == TensorProto.FLOAT:
                tensor_type.elem_type = TensorProto.DOUBLE  # float64 is enum 11
        



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
        # kept_initializers = [
        #     tensor for tensor in model.graph.initializer
        #     if tensor.name in used_initializer_names
        # ]
        # Keep and convert to float64 only used initializers
        kept_initializers = []
        for name in used_initializer_names:
            if name in initializer_map:
                orig_init = initializer_map[name]
                np_array = numpy_helper.to_array(orig_init)

                if np_array.dtype == np.float32:
                    # Convert to float64
                    np_array = np_array.astype(np.float64)
                    new_init = numpy_helper.from_array(np_array, name=name)
                    kept_initializers.append(new_init)
                else:
                    # Keep as-is
                    kept_initializers.append(orig_init)

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

        for out in model.graph.output:
            # if out.name == "output":
                # out.name = "output_int"  # match Cast output
                out.type.tensor_type.elem_type = onnx.TensorProto.INT64

        
        # # For debugging attributes
        # for node in model.graph.node:
        #     if node.op_type == "Int64Gemm":
        #         for attr in node.attribute:
        #             print(f"{attr.name}: {attr.type}")
        # model.opset_import[0].version = 17



        # a = np.ones((2, 3), dtype=np.int64)
        # b = np.ones((3, 4), dtype=np.int64)
        # c = np.zeros((2, 4), dtype=np.int64)
        # dummy_op = OrtPyFunction.from_customop("Int64Gemm",int64_gemm7)


        # out = dummy_op(a, b, c, alpha=1.0, beta=1.0, transA=False, transB=False)
        # print("Output:", out)
        # for node in model.graph.node:
        #     print(f"Node: {node.name}, OpType: {node.op_type}")
        #     for attr in node.attribute:
        #         print(f"  Attr: {attr.name}, Type: {attr.type}")
        # TODO This has not been extensively tested. May need to somehow include this when quantizing layers individually (Concern is that some layers shouldnt be converted into this type...)
        # Such as multiplying up scalers etc.
        for vi in model.graph.value_info:
            vi.type.tensor_type.elem_type = TensorProto.INT64
        # TODO remove
        # !!! MaxPool
        custom_domain = helper.make_operatorsetid(domain="ai.onnx.contrib",version=1)
        domains = [op.domain for op in model.opset_import]
        if "ai.onnx.contrib" not in domains:
            model.opset_import.append(custom_domain)
        onnx.checker.check_model(model)
        onnx.save(model, "debug_test.onnx")
        return model
        

    def quantize_layer(self, node: onnx.NodeProto, rescale: bool, model: onnx.ModelProto, scale: int, scale_base: int, initializer_map: dict[str, onnx.TensorProto]) -> onnx.NodeProto:
        quant_nodes = self.op_quantizer.quantize(node, rescale, model.graph, scale, scale_base, initializer_map)
        return quant_nodes
    
    def quantize_input(self, input_name, op_quantizer: ONNXOpQuantizer, scale_base, scale):
        scale_value = scale_base ** scale
        original_output = input_name

        # === Create scale constant ===
        scale_const_name = input_name + "_scale"
        scale_tensor = numpy_helper.from_array(
            np.array([scale_value], dtype=np.float64), name=scale_const_name
        )
        op_quantizer.new_initializers.append(scale_tensor)

        # === Add Mul node ===
        scaled_output_name = f"{input_name}_scaled"
        mul_node = helper.make_node(
            "Mul",
            inputs=[input_name, scale_const_name],
            outputs=[scaled_output_name],
            name=f"{input_name}_mul",
        )
        # graph.node.append(mul_node)
        # replace_input_references(graph, original_output, mul_node.output[0])

        # === Floor node (simulate rounding) ===
        rounded_output_name = f"{input_name}_scaled_floor"
        floor_node = helper.make_node(
            "Floor",
            inputs=[scaled_output_name],
            outputs=[rounded_output_name],
            name=f"{scaled_output_name}",
        )
        output_name = f"{rounded_output_name}_int"
        cast_to_int64 = helper.make_node(
            "Cast",
            # inputs=[rounded_output_name],
            inputs=[scaled_output_name],
            outputs=[output_name],
            to=onnx.TensorProto.INT64,
            name = rounded_output_name
        )
        return output_name, mul_node, floor_node, cast_to_int64


    
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
        print(self.model.graph.node)
        inferred_model = shape_inference.infer_shapes(self.model) 

        # Check the model and print Y"s shape information
        onnx.checker.check_model(inferred_model)
        output_name_to_shape = extract_shape_dict(inferred_model)
        (architecture, w_and_b) = self.analyze_layers(output_name_to_shape)
        for w in w_and_b:
            w_and_b_array = np.asarray(w.tensor)
            # VERY VERY TEMPORARY FIX
            if "bias" in w.name:
                w_and_b_scaled = w_and_b_array * (getattr(self, "scale_base", 2)**(getattr(self,"scaling", 18)*2))
            else:
                w_and_b_scaled = w_and_b_array * (getattr(self, "scale_base", 2)**getattr(self,"scaling", 18))
            w_and_b_out = w_and_b_scaled.astype(np.int64).tolist()
            w.tensor = w_and_b_out
            
        
        
        inputs = []
        outputs = []
        for input in self.model.graph.input:
            shape =  output_name_to_shape.get(input.name, [])
            elem_type = getattr(input, "elem_type", -1)
            inputs.append(ONNXIO(input.name, elem_type, shape))

        for output in self.model.graph.output:
            shape =  output_name_to_shape.get(output.name, [])
            elem_type = getattr(output, "elem_type", -1)
            outputs.append(ONNXIO(output.name, elem_type, shape))
        
        architecture = {
            "inputs": [asdict(i) for i in inputs],
            "outputs": [asdict(o) for o in outputs],
            "architecture": [asdict(a) for a in architecture],
        }
        weights = {
            "w_and_b": [asdict(w_b) for w_b in w_and_b]
        }
        circuit_params = {
            "scale_base": getattr(self, "scale_base", 2),
            "scaling": getattr(self,"scaling", 18),
            "rescale_config": getattr(self, "rescale_config", {})
        }
        self.save_quantized_model("test.onnx")
        return architecture, weights, circuit_params

    def get_model_and_quantize(self):
        
        if hasattr(self, 'model_file_name'):
            self.load_model(self.model_file_name)
        else:
            raise FileNotFoundError("An ONNX model is required at the specified path")
        
        # self.model = model
        self.quantized_model = self.quantize_model(self.model, getattr(self,"scale_base", 2), getattr(self,"scaling", 18), rescale_config=getattr(self,"rescale_config", {}))
        
        # sys.exit()

    def test_accuracy(self, inputs = None):
        # model = onnx.load()
        model = self.model
        input_shape = []
        for input in model.graph.input:
            for d in input.type.tensor_type.shape.dim:
                if (d.HasField("dim_value")):
                    val = d.dim_value  # known dimension
                elif (d.HasField("dim_param")):
                    if "batch_size" not in d.dim_param:
                        raise ValueError("Unknown dimension")
                    val = 1
                    # print (d.dim_param, end=", ")  # unknown dimension with symbolic name
                else:
                    # print ("?", end=", ")  # unknown dimension with no name
                    raise ValueError("Unknown dimension")

                input_shape.append(val)
        # inputs = torch.rand(input_shape)*2 - 1
        new_model = self.quantize_model(model, getattr(self,"scale_base", 2), getattr(self,"scaling", 18))
        custom_domain = onnx.helper.make_operatorsetid(domain="ai.onnx.contrib", version=1)
        new_model.opset_import.append(custom_domain)
        onnx.checker.check_model(new_model)

        with open(self.quantized_model_file_name, "wb") as f:
            f.write(new_model.SerializeToString())

        model = onnx.load(self.quantized_model_file_name)
        onnx.checker.check_model(model)  # This throws a descriptive error
        if inputs == None:
            inputs = torch.rand([1,4,28,28])*2 - 1
        outputs_true = self.run_model_onnx_runtime(self.model_file_name, inputs)[0][0].tolist()

        outputs_quant = self.run_model_onnx_runtime(self.quantized_model_file_name, inputs)[0][0].tolist()

        
        formatter = np.vectorize(lambda x: float(f"{x:.5f}"))
        print("ONNXRuntime true model output : ",formatter(outputs_true))


        scale = getattr(self, "scale_base", 2) ** getattr(self, "scaling", 18)
        formatter = np.vectorize(lambda o: float(f"{o / scale:.5f}"))
        print("ONNXRuntime quant model output: ",formatter(outputs_quant))
        # print([[o/(2**21) for o in outputs_quant]])


    def get_outputs(self, inputs):
        input_name = self.ort_sess.get_inputs()[0].name
        output_name = self.ort_sess.get_outputs()[0].name
        # if inputs.dtype in (torch.float16, torch.float32, torch.float64):
        #     print("Tensor is a float type")

        # TODO This may cause some rounding errors at some point but works for now. Should be checked at some point
        inputs = torch.as_tensor(inputs)
        if inputs.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            inputs = inputs.double()
            inputs = inputs / (self.scale_base**self.scaling)
        # TODO add for all inputs (we should be able to account for multiple inputs...)
        # TODO this is not optimal or robust
        if self.ort_sess.get_inputs()[0].type == "tensor(double)":
            outputs = self.ort_sess.run([output_name], {input_name: np.asarray(inputs).astype(np.float64)})
        else:
            outputs = self.ort_sess.run([output_name], {input_name: np.asarray(inputs)})
        return outputs


# def find_non_serializable_fields(obj, path="root"):
#     """
#     Recursively finds fields that are not JSON serializable.
#     """
#     if is_dataclass(obj):
#         obj = asdict(obj)

#     if isinstance(obj, dict):
#         for k, v in obj.items():
#             find_non_serializable_fields(v, f"{path}.{k}")
#     elif isinstance(obj, list):
#         for i, item in enumerate(obj):
#             find_non_serializable_fields(item, f"{path}[{i}]")
#     else:
#         try:
#             json.dumps(obj)
#         except TypeError:
#             # print(obj.tolist())
#             print(f"❌ Non-serializable field found at: {path} (type: {type(obj).__name__})")
#             # sys.exit()

class ZKONNXModel(ONNXConverter, ZKModelBase):
    def __init__(self):
        raise NotImplementedError("Must implement __init__")


if __name__ == "__main__":
    # path  ="./models_onnx/doom.onnx"
    path  = "./models_onnx/test_doom_cut.onnx"


    converter = ONNXConverter()
    converter.model_file_name, converter.quantized_model_file_name = path, "quantized_doom.onnx"
    converter.scale_base, converter.scaling = 2,18

    # converter.model_file_name = path
    converter.load_model(path)
    converter.get_model_and_quantize()

    # converter.model = create_dummy_model()
    converter.test_accuracy()
    # weights = converter.get_weights()
    # print(weights[1].keys())
    # print(weights[1]["w_and_b"][0].keys())

    # with open('onnx_weights.json', 'w') as fp:
    #     json.dump(weights,fp)
    # with open('onnx_arch.json', 'w') as fp:
    #     json.dump(arch,fp)

    # converter.analyze_layers("")
