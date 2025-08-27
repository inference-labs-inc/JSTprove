import math
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple
import numpy as np
from onnx import AttributeProto, numpy_helper
import onnx
from onnx.numpy_helper import to_array, from_array

ATTRIBUTE_PARSERS = {
    AttributeProto.FLOAT: lambda a: a.f,
    AttributeProto.INT: lambda a: a.i,
    AttributeProto.STRING: lambda a: a.s.decode('utf-8', errors='replace'),
    AttributeProto.FLOATS: lambda a: list(a.floats),
    AttributeProto.INTS: lambda a: list(a.ints),
    AttributeProto.STRINGS: lambda a: [s.decode('utf-8', errors='replace') for s in a.strings],
    AttributeProto.TENSOR: lambda a: numpy_helper.to_array(a.t).tolist(),
    AttributeProto.TENSORS: lambda a: [numpy_helper.to_array(t).tolist() for t in a.tensors],
}

def parse_attribute(attr: AttributeProto) -> Any:
    """Parse ONNX attributes into a Python-native type.

    Args:
        attr (AttributeProto): The ONNX attribute to parse.

    Raises:
        ValueError: If the attribute type is unsupported.

    Returns:
        Any: The parsed attribute value as a Python type.
    """
    parser = ATTRIBUTE_PARSERS.get(attr.type)
    if parser is None:
        raise ValueError(f"Unsupported attribute type: {attr.type}")
    return parser(attr)


def parse_attributes(attrs: List[AttributeProto]) -> Dict[str, Any]:
    """Parse multiple ONNX attributes into a dictionary.

    Args:
        attrs (List[AttributeProto]): List of ONNX attributes.

    Returns:
        Dict[str, Any]: Mapping of attribute names to their parsed values.
    """
    return {attr.name: parse_attribute(attr) for attr in attrs}

def extract_shape_dict(inferred_model: onnx.GraphProto) -> Dict[str, List[int]]:
    """Extract shape information from an ONNX model's graph.

    Args:
        inferred_model (onnx.GraphProto): The inferred ONNX model graph.

    Returns:
        Dict[str, List[int]]: Mapping from tensor names to their shape dimensions.
                              Unknown dimensions are returned as 1.
    """
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
    """Compute the product of dimensions, for flattened length.

    Args:
        dims (Iterable): Iterable of integer dimensions.

    Returns:
        int: The product of all dimensions.
    """
    return math.prod(dims)

def replace_input_references(graph: onnx.GraphProto, old_output: str, new_output: str):
    """Replace all references to an input tensor in an ONNX graph.

    Args:
        graph (onnx.GraphProto): The ONNX graph to modify.
        old_output (str): The original tensor name to replace.
        new_output (str): The new tensor name.
    """   
    for node in graph.node:
        for i, input_name in enumerate(node.input):
            if input_name == old_output:
                node.input[i] = new_output


def create_quantized_initializer(orig_tensor: onnx.TensorProto, scale_exponent: int, scale: int, scale_base: int) -> Tuple[onnx.TensorProto, str]:
    """Create a quantized ONNX tensor initializer from a floating-point tensor.

    Args:
        orig_tensor (onnx.TensorProto): The original tensor.
        scale_exponent (int): Exponent for scaling (e.g., 18 would lead to a scale factor 2**18).    
        scale (int): The scale multiplier (eg. bias must often be multiplied twice).
        scale_base (int): Base for fixed-point scaling (e.g., 2).

    Returns:
        Tuple[onnx.TensorProto, str]: A tuple containing the quantized tensor and its new name.
    """
    factor = scale_base ** (scale * scale_exponent)
    arr = to_array(orig_tensor).astype(np.float64) * factor
    arr = arr.astype(np.int64)
    new_name = f"{orig_tensor.name}_q{scale_exponent}"
    return from_array(arr, name=new_name), new_name

def extract_attributes(node: onnx.NodeProto) -> dict:
    """Extract all attributes from an ONNX node into a Python dictionary.

    Args:
        node (onnx.NodeProto): The ONNX node to extract attributes from.

    Raises:
        ValueError: If an attribute type is unsupported.

    Returns:
        dict: Mapping of attribute names to Python-native values.
    """    
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
    return attrs

def get_input_shapes(onnx_model: onnx.ModelProto) -> dict:
    """Get the input tensor shapes from an ONNX model.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model.

    Returns:
        dict: Mapping from input tensor names to their shape dimensions.
    """   
    input_shapes = {}
    for input in onnx_model.graph.input:
        input_name = input.name
        # Get the shape from the input's type information
        shape = [dim.dim_value for dim in input.type.tensor_type.shape.dim]
        input_shapes[input_name] = shape
    return input_shapes

def rescale_to_int(rescale: Any) -> int:
    """Convert a value to an integer for rescaling purposes.

    Args:
        rescale (Any): The value to convert.

    Returns:
        int: The integer value.
    """
    return int(rescale)


def get_model_op_types(self, model: onnx.ModelProto) -> Set[str]:
    """Retrieve the set of operation types used in an ONNX model.

    Args:
        model (onnx.ModelProto): The ONNX model.

    Returns:
        Set[str]: A set of unique op types.
    """
    return {node.op_type for node in model.graph.node}

def check_model_compatibility(model: onnx.ModelProto, registry: Dict[str, Callable]) -> Tuple[bool, Set[str]]:
    """Check whether an ONNX model's operations are supported by a registry.

    Args:
        model (onnx.ModelProto): The ONNX model.
        registry (Dict[str, Callable]): Mapping of supported operation names to handlers.

    Returns:
        Tuple[bool, Set[str]]: A tuple containing:
            - bool: True if all operations are supported, False otherwise.
            - Set[str]: The set of unsupported operations.
    """   
    model_ops = get_model_op_types(model)
    unsupported_ops = model_ops - set(registry)
    return (len(unsupported_ops) == 0, unsupported_ops)

def get_attribute_ints(node: onnx.NodeProto, name: str, default: list[int] = None) -> list[int]:
    """Retrieve a list of integer values from an ONNX node's attribute.

    Args:
        node (onnx.NodeProto): The ONNX node.
        name (str): Name of the attribute to retrieve.
        default (list[int], optional): Default list to return if the attribute is missing. Defaults to None.

    Returns:
        list[int]: List of integers from the attribute, or the default if not found.
    """
    for attr in node.attribute:
        if attr.name == name and attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
    return default if default is not None else []
