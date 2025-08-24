import numpy as np
import onnx
from onnx import helper, numpy_helper
from typing import Callable, Dict, List, Optional, Union

from python.core.model_processing.onnx_custom_ops.onnx_helpers import create_quantized_initializer, extract_attributes, replace_input_references
from onnx.numpy_helper import to_array, from_array

from python.core.model_processing.onnx_quantizer.layers.base import BaseOpQuantizer
from python.core.model_processing.onnx_quantizer.exceptions import InvalidParamError

class GemmQuantizer(BaseOpQuantizer):
    """
    Quantizer for ONNX Gemm layers.

    - Replaces standard Gemm with Int64Gemm from the `ai.onnx.contrib` domain and makes relevant additional changes to the graph.
    - Validates that all required Gemm parameters are present.
    """
    def __init__(self, new_initializers):
        self.new_initializers = new_initializers

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
        Quantize a Gemm node by:
        1. Quantizing its weights and bias.
        2. Adding a scale constant.
        3. Replacing it with an Int64Gemm node.

        Args:
            node (onnx.NodeProto): The node to quantize.
            rescale (bool): Whether rescaling is enabled (Doesnt have an affect on this op type)
            graph (onnx.GraphProto): The ONNX graph.
            scale_exponent (int): Scale exponent.
            scale_base (int): The base of scaling.
            initializer_map (dict[str, onnx.TensorProto]): Map of initializer names to tensor data.

        Returns:
            List[onnx.NodeProto]: A list of ONNX nodes (quantized and any auxiliary nodes).
        """
        nodes = []
        output_name = f"{node.name}_int"

        nodes, node.input[:] = self.add_nodes_w_and_b(node = node, scale_exponent = scale_exponent, scale_base = scale_base, initializer_map = initializer_map, graph = graph)

        attrs = extract_attributes(node)
        attrs.setdefault("transA", 0)
        attrs.setdefault("transB", 0) 
        attrs["rescale"] = int(rescale)
        for attr in node.attribute:
            print(f"{attr.name}: type={attr.type} ({onnx.AttributeProto.AttributeType.Name(attr.type)})")

        scale_value = scale_base ** scale_exponent
        
        # TODO make this constant to all layers
        # === Create scale constant ===
        scale_const_name = f"{output_name}_scaler"
        scale_tensor = numpy_helper.from_array(
            np.array([scale_value], dtype=np.int64), name=scale_const_name
        )
        self.new_initializers.append(scale_tensor)
        node.input.append(scale_const_name)
        int64_gemm = onnx.helper.make_node(
                                        "Int64Gemm",
                                        inputs=node.input,
                                        outputs=node.output,  # preserve original output name
                                        name=output_name,
                                        domain="ai.onnx.contrib",
                                        **attrs
                                    )
        nodes.append(int64_gemm)
        return nodes
    
    
    def check_supported(self, node: onnx.NodeProto, initializer_map: dict[str, onnx.TensorProto] = None) -> None:
        """
        Perform high-level validation to ensure that this node
        can be quantized safely.

        Args:
            node (onnx.NodeProto): ONNX node to be checked
            initializer_map (dict[str, onnx.TensorProto]): Initializer map (name of weight or bias and tensor)

        Raises:
            InvalidParamError: If any requirement is not met.
        """
        if len(node.input) < 2:
            raise InvalidParamError(node.name, node.op_type, f"Expected at least 2 inputs (input, weights), got {len(node.input)}")
        
        # TODO currently requires bias layer
        if len(node.input) < 3:
            raise InvalidParamError(node.name, node.op_type, f"Expected at least 3 inputs (input, weights, bias), got {len(node.input)}")

        alpha = next((attr.f for attr in node.attribute if attr.name == "alpha"), 1.0)
        beta = next((attr.f for attr in node.attribute if attr.name == "beta"), 1.0)
        transA = next((attr.i for attr in node.attribute if attr.name == "transA"), 0)
        transB = next((attr.i for attr in node.attribute if attr.name == "transB"), 1)

        if alpha != 1.0:
            raise InvalidParamError(node.name, node.op_type, f"alpha value of {alpha} not supported", "alpha", "1.0")
        if beta != 1.0:
            raise InvalidParamError(node.name, node.op_type, f"beta value of {beta} not supported", "beta", "1.0")
        
        if not transA in [0,1]:
            raise InvalidParamError(node.name, node.op_type, f"transA value of {transA} not supported", "transA", "(0,1)")
        if not transB in [0,1]:
            raise InvalidParamError(node.name, node.op_type, f"transB value of {transB} not supported", "transB", "(0,1)")
