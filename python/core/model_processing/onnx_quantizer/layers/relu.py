import numpy as np
import onnx
from onnx import helper, numpy_helper
from typing import Callable, Dict, List, Optional, Union

from python.core.model_processing.onnx_custom_ops.onnx_helpers import create_quantized_initializer, extract_attributes, replace_input_references
from onnx.numpy_helper import to_array, from_array

from python.core.model_processing.onnx_quantizer.layers.base import BaseOpQuantizer

class ReluQuantizer(BaseOpQuantizer): 
    """
    Quantizer for ONNX ReLU layers.

    - Replaces standard ReLU with Int64ReLU from the `ai.onnx.contrib` domain and makes relevant additional changes to the graph.
    - Validates that all required ReLU parameters are present.
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
            List[onnx.NodeProto]: The quantized ONNX node.
        """
        return onnx.helper.make_node(
            "Int64Relu",
            inputs=node.input,
            outputs=node.output,  # preserve original output name
            # outputs=output_intermediate,  # preserve original output name
            name=node.name,
            domain="ai.onnx.contrib",
        )
    
    def check_supported(self, node: onnx.NodeProto, initializer_map: dict[str, onnx.TensorProto] = None) -> None:    
        """
        Perform high-level validation to ensure that this node
        can be quantized safely.

        Args:
            node (onnx.NodeProto): ONNX node to be checked
            initializer_map (dict[str, onnx.TensorProto]): Initializer map (name of weight or bias and tensor)
        """ 
        return None
