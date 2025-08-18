import numpy as np
import onnx
from onnx import helper, numpy_helper
from typing import Callable, Dict, List, Optional, Union

from python.testing.core.utils.onnx_helpers import create_quantized_initializer, extract_attributes, get_attribute_ints, replace_input_references
from onnx.numpy_helper import to_array, from_array

from python.testing.core.utils.onnx_quantizer.exceptions import InvalidParamError
from python.testing.core.utils.onnx_quantizer.layers.base import BaseOpQuantizer

class MaxpoolQuantizer(BaseOpQuantizer):
    def __init__(self, new_initializer = None):
        super().__init__()
        
    def quantize(
        self,
        node: onnx.NodeProto,
        rescale: bool,
        graph: onnx.GraphProto,
        scale: int,
        scale_base: int,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> List[onnx.NodeProto]:
        attrs = {a.name: helper.get_attribute_value(a) for a in node.attribute}
        attr_str = {k: ",".join(map(str, v)) if isinstance(v, list) else str(v) for k, v in attrs.items()}
        return helper.make_node(
            "Int64MaxPool",
            inputs=node.input,
            outputs=node.output,
            name=node.name,
            domain="ai.onnx.contrib",
            **attr_str
    )

    def check_supported(self, node: onnx.NodeProto, initializer_map: dict[str, onnx.TensorProto]):
        self.check_all_params_exist(node)
        self.check_params_size(node)

    def check_all_params_exist(self, node: onnx.NodeProto):
        strides = next((attr.f for attr in node.attribute if attr.name == "strides"), "N/A")
        kernel_shape = next((attr.f for attr in node.attribute if attr.name == "kernel_shape"), "N/A")
        dilations = next((attr.f for attr in node.attribute if attr.name == "dilations"), "N/A")
        pads = next((attr.f for attr in node.attribute if attr.name == "pads"), "N/A")
        

        if strides == "N/A":
            raise InvalidParamError(node.name, node.op_type, f"Missing strides parameter", "strides")
        if kernel_shape == "N/A":
            raise InvalidParamError(node.name, node.op_type, f"Missing kernel_shape parameter", "kernel_shape")
        if dilations == "N/A":
            raise InvalidParamError(node.name, node.op_type, f"Missing dilations parameter", "dilations")
        if pads == "N/A":
            raise InvalidParamError(node.name, node.op_type, f"Missing pads parameter", "pads")
        
    def check_params_size(self, node: onnx.NodeProto):
        strides = get_attribute_ints(node, "strides", default="N/A")
        kernel_shape = get_attribute_ints(node, "kernel_shape", default="N/A")
        dilations = get_attribute_ints(node, "dilations", default="N/A")
        pads = get_attribute_ints(node, "pads", default="N/A")


        if len(kernel_shape) != 2:
            raise InvalidParamError(node.name, node.op_type, f"Currently only maxpool2d is supported. Found {len(kernel_shape)}D")
