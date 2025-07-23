import numpy as np
import onnx
from onnx import helper, numpy_helper
from typing import Callable, Dict, List, Optional, Union

from python.testing.core.utils.onnx_helpers import create_quantized_initializer, extract_attributes, replace_input_references
from onnx.numpy_helper import to_array, from_array

from python.testing.core.utils.onnx_quantizer.layers.base import BaseOpQuantizer
from python.testing.core.utils.onnx_quantizer.exceptions import InvalidParamError

class GemmQuantizer(BaseOpQuantizer):
    def __init__(self, new_initializers):
        self.new_initializers = new_initializers

    def quantize(
        self,
        node: onnx.NodeProto,
        rescale: bool,
        graph: onnx.GraphProto,
        scale: int,
        scale_base: int,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> List[onnx.NodeProto]:
        nodes = []
        output_name = f"{node.name}_int"

        nodes, node.input[:] = self.add_nodes_w_and_b(node, scale, scale_base, initializer_map, graph)

        attrs = extract_attributes(node)
        attrs.setdefault("transA", 0)
        attrs.setdefault("transB", 0) 
        attrs["rescale"] = int(rescale)
        for attr in node.attribute:
            print(f"{attr.name}: type={attr.type} ({onnx.AttributeProto.AttributeType.Name(attr.type)})")

        scale_value = scale_base ** scale
        
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
    
    
    def check_supported(self, node: onnx.NodeProto, initializer_map: dict[str, onnx.TensorProto] = None):
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