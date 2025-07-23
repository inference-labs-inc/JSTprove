import numpy as np
import onnx
from onnx import helper, numpy_helper
from typing import Callable, Dict, List, Optional, Union

from python.testing.core.utils.onnx_helpers import create_quantized_initializer, extract_attributes, replace_input_references
from onnx.numpy_helper import to_array, from_array

from python.testing.core.utils.onnx_quantizer.layers.base import BaseOpQuantizer

class ConstantQuantizer(BaseOpQuantizer):
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
        output_name = node.output[0]

        data_ops = {"Add", "Mul", "Conv", "MatMul", "Sub", "Div", "Gemm"}  # ops that consume numeric constants
        is_data_constant = any(
            output_name in n.input and n.op_type in data_ops
            for n in graph.node
        )

        if not is_data_constant:
            return node  # ✅ Used for shape or index — don't quantize

        # Safe to quantize: numeric constant used in computation
        for attr in node.attribute:
            if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                arr = numpy_helper.to_array(attr.t).astype(np.float64)
                arr *= scale_base ** scale
                attr.t.CopyFrom(numpy_helper.from_array(arr, name=""))

        node.name = node.name + "_quant"
        return node