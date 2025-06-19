from typing import Any, Dict, Iterable, List, Tuple
import numpy as np
from onnx import AttributeProto, numpy_helper
import onnx
from onnx.numpy_helper import to_array, from_array

def parse_attribute(attr: AttributeProto):
    dispatch = {
        AttributeProto.FLOAT: lambda a: a.f,
        AttributeProto.INT: lambda a: a.i,
        AttributeProto.STRING: lambda a: a.s.decode('utf-8'),
        AttributeProto.FLOATS: lambda a: list(a.floats),
        AttributeProto.INTS: lambda a: list(a.ints),
        AttributeProto.STRINGS: lambda a: [s.decode('utf-8') for s in a.strings],
        AttributeProto.TENSOR: lambda a: numpy_helper.to_array(a.t).tolist(),
        AttributeProto.TENSORS: lambda a: [numpy_helper.to_array(t).tolist() for t in a.tensors],
        # AttributeProto.GRAPH: lambda a: a.g,              # GraphProto
        # AttributeProto.GRAPHS: lambda a: list(a.graphs),  # List[GraphProto]
    }

    return dispatch.get(attr.type, lambda a: f"<Unsupported type {attr.type}>")(attr)


def parse_attributes(attrs: List[AttributeProto]) -> Dict[str, Any]:
    return {attr.name: parse_attribute(attr) for attr in attrs}

def extract_shape_dict(inferred_model) -> Dict[str, List[int]]:
        value_info = {}
        graph = inferred_model.graph
        all_info = list(graph.value_info) + list(graph.output) + list(graph.input)
        for vi in all_info:
            if vi.type.HasField("tensor_type"):
                shape = [
                    # TODO figure out how to deal with bad value
                    # d.dim_value if d.HasField("dim_value") else -1
                    d.dim_value if d.HasField("dim_value") else 1
                    for d in vi.type.tensor_type.shape.dim
                ]
                value_info[vi.name] = shape
        return value_info

def dims_prod(dims: Iterable) -> int:
    prod = 1
    for dim in dims:
        prod *= dim

    return prod

def replace_input_references(graph: onnx.GraphProto, old_output: str, new_output: str):
    for node in graph.node:
        for i, input_name in enumerate(node.input):
            if input_name == old_output:
                node.input[i] = new_output


def create_quantized_initializer(orig_tensor: onnx.TensorProto, scale_exponent: int, scale: int, scale_base: int) -> Tuple[onnx.TensorProto, str]:
    factor = scale_base ** (scale * scale_exponent)
    arr = to_array(orig_tensor).astype(np.float32) * factor
    arr = arr.astype(np.int64)
    new_name = orig_tensor.name + f"_q{scale_exponent}"
    new_tensor = from_array(arr, name=new_name)
    return new_tensor, new_name

def extract_attributes(node: onnx.NodeProto) -> dict:
    attrs = {}
    for attr in node.attribute:
        name = attr.name
        val = onnx.helper.get_attribute_value(attr)

        if attr.type == AttributeProto.FLOAT:
            attrs[name] = float(val)
        elif attr.type == AttributeProto.INT:
            attrs[name] = int(val)
        elif attr.type == AttributeProto.FLOATS:
            attrs[name] = [float(x) for x in val]  # ← you want to ensure these are int if your op expects it
        elif attr.type == AttributeProto.INTS:
            # attrs[name] = [int(x) for x in val]  # ← you want to ensure these are int if your op expects it
            attrs[name] =",".join(str(v) for v in val)
        elif attr.type == AttributeProto.STRING:
            attrs[name] = val.decode("utf-8") if isinstance(val, bytes) else val
        elif attr.type == AttributeProto.BOOL:
            attrs[name] = bool(val)
        else:
            raise ValueError(f"Unsupported attribute type: {attr.name} (type={attr.type})")
    

    for k, v in attrs.items():
        # print(type(k))
        pass
    #     if isinstance(v, float):
    #         print(f"⚠️  Attribute {k} is float: {v} → will be coerced?")
    #     if isinstance(v, list) and any(isinstance(i, float) for i in v):
    #         print(f"⚠️  Attribute {k} contains float values: {v}")
    return attrs

def get_input_shapes(onnx_model: onnx.ModelProto):
        input_shapes = {}
        for input in onnx_model.graph.input:
            input_name = input.name
            # Get the shape from the input's type information
            shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
            input_shapes[input_name] = shape
        return input_shapes

# def extract_attributes(node: onnx.NodeProto) -> dict:
#     return {
#         attr.name: onnx.helper.get_attribute_value(attr)
#         for attr in node.attribute
#     }