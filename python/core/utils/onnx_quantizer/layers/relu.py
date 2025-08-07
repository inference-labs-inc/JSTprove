import numpy as np
import onnx
from onnx import helper, numpy_helper
from typing import Callable, Dict, List, Optional, Union

from python.core.utils.onnx_helpers import create_quantized_initializer, extract_attributes, replace_input_references
from onnx.numpy_helper import to_array, from_array

from python.core.utils.onnx_quantizer.layers.base import BaseOpQuantizer

class ReluQuantizer(BaseOpQuantizer):
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
        return onnx.helper.make_node(
            "Int64Relu",
            inputs=node.input,
            outputs=node.output,  # preserve original output name
            # outputs=output_intermediate,  # preserve original output name
            name=node.name,
            domain="ai.onnx.contrib",
        )