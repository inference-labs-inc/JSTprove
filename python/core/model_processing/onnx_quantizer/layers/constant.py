import numpy as np
import onnx
from onnx import numpy_helper
from typing import List, Optional

from python.core.model_processing.onnx_quantizer.layers.base import BaseOpQuantizer

class ConstantQuantizer(BaseOpQuantizer):
    """
    Quantizer for ONNX Constant node.

    This quantizer only modifies constants that are:
    - Numeric tensors
    - Used directly in computation
    
    Constants used for shape, indexing, or other non-numeric roles are left unchanged.
    """
    DATA_OPS = {"Add", "Mul", "Conv", "MatMul", "Sub", "Div", "Gemm"}  # ops that consume numeric constants

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
        """Apply quantization scaling to a constant if it is used in numeric computation.

        Args:
            node (onnx.NodeProto): The Constant node to quantize.
            rescale (bool): Whether rescaling is enabled (Doesnt have an affect on this op type)
            graph (onnx.GraphProto): The ONNX graph.
            scale_exponent (int): Scale exponent.
            scale_base (int): The base of scaling
            initializer_map (dict[str, onnx.TensorProto]): Map of initializer names to tensor data.

        Returns:
            List[onnx.NodeProto]: The modified node (possibly unchanged).
        """        
        output_name = node.output[0]

        
        is_data_constant = any(
            output_name in n.input and n.op_type in self.DATA_OPS
            for n in graph.node
        )

        if not is_data_constant:
            # Skip quantization for non-numeric constants
            return node  

        # Safe to quantize: numeric constant used in computation
        for attr in node.attribute:
            if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                arr = numpy_helper.to_array(attr.t).astype(np.float64)
                arr *= scale_base ** scale_exponent
                attr.t.CopyFrom(numpy_helper.from_array(arr, name=""))
        node.name += "_quant"
        return node
    
    def check_supported(self, node: onnx.NodeProto, initializer_map: dict[str, onnx.TensorProto] = None) -> None:
        """All Constant nodes are supported... For now.

        Args:
            node (onnx.NodeProto): Node to be checked
            initializer_map (dict[str, onnx.TensorProto], optional): Map of initializer names to tensor data. Defaults to None.
        """        
        return None
