import numpy as np
import onnx
from onnx import helper, numpy_helper
from typing import Callable, Dict, List, Optional, Union

from python.core.model_processing.onnx_custom_ops.onnx_helpers import create_quantized_initializer, extract_attributes, get_attribute_ints, replace_input_references
from onnx.numpy_helper import to_array, from_array

from python.core.model_processing.onnx_quantizer.exceptions import InvalidParamError
from python.core.model_processing.onnx_quantizer.layers.base import BaseOpQuantizer

class MaxpoolQuantizer(BaseOpQuantizer):
    """
    Quantizer for ONNX MaxPool layers.

    - Replaces standard MaxPool with Int64MaxPool from the `ai.onnx.contrib` domain and makes relevant additional changes to the graph.
    - Validates that all required MaxPool parameters are present.
    """
    def __init__(self, new_initializer = None):
        super().__init__()
        
    def quantize(
        self,
        node: onnx.NodeProto,
        rescale: bool,
        graph: onnx.GraphProto,
        scale_exponent: int,
        scale_base: int,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> List[onnx.NodeProto]:
        """
        Quantize a node by converting the node to Int64 version

        Args:
            node (onnx.NodeProto): The node to quantize.
            rescale (bool): Whether rescaling is enabled (Doesnt have an affect on this op type)
            graph (onnx.GraphProto): The ONNX graph.
            scale_exponent (int): Scale exponent.
            scale_base (int): The base of scaling.
            initializer_map (dict[str, onnx.TensorProto]): Map of initializer names to tensor data.

        Returns:
            List[onnx.NodeProto]: A list of ONNX nodes (quantized MaxPool and any auxiliary nodes).
        """
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
        """
        Perform high-level validation to ensure that this node
        can be quantized safely.

        Args:
            node (onnx.NodeProto): ONNX node to be checked
            initializer_map (dict[str, onnx.TensorProto]): Initializer map (name of weight or bias and tensor)

        Raises:
            InvalidParamError: If any requirement is not met.
        """
        self.check_all_params_exist(node)
        self.check_params_size(node)

    def check_all_params_exist(self, node: onnx.NodeProto):
        """Checks all parameters that are needed, do exist

        Args:
            node (onnx.NodeProto): ONNX node to check

        Raises:
            InvalidParamError: If shape requirement is not met.
        """        
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
        """Checks dimension of the layer and ensures that it is supported

        Args:
            node (onnx.NodeProto): ONNX node to check

        Raises:
            InvalidParamError: If shape requirement is not met.
        """        
        strides = get_attribute_ints(node, "strides", default="N/A")
        kernel_shape = get_attribute_ints(node, "kernel_shape", default="N/A")
        dilations = get_attribute_ints(node, "dilations", default="N/A")
        pads = get_attribute_ints(node, "pads", default="N/A")


        if len(kernel_shape) != 2:
            raise InvalidParamError(node.name, node.op_type, f"Currently only maxpool2d is supported. Found {len(kernel_shape)}D")
