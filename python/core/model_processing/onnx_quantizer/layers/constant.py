from __future__ import annotations

from typing import ClassVar

import numpy as np
import onnx
from onnx import numpy_helper

from python.core.model_processing.onnx_quantizer.exceptions import (
    HandlerImplementationError,
)
from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    ScaleConfig,
)


class ConstantQuantizer(BaseOpQuantizer):
    """
    Quantizer for ONNX Constant node.

    This quantizer only modifies constants that are:
    - Numeric tensors
    - Used directly in computation

    Constants used for shape, indexing, or other non-numeric roles are left unchanged.
    """

    DATA_OPS: ClassVar = {
        "Add",
        "Mul",
        "Conv",
        "MatMul",
        "Sub",
        "Div",
        "Gemm",
    }  # ops that consume numeric constants

    def __init__(
        self: ConstantQuantizer,
        new_initializer: list[onnx.TensorProto] | None = None,
    ) -> None:
        super().__init__()
        _ = new_initializer

    def quantize(
        self: ConstantQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        """Apply quantization scaling to a constant if it is used in
            numeric computation.

        Args:
            node (onnx.NodeProto): The Constant node to quantize.
            rescale (bool): Whether rescaling is enabled
                (Doesnt have an affect on this op type in some cases)
            graph (onnx.GraphProto): The ONNX graph.
            scale_exponent (int): Scale exponent.
            scale_base (int): The base of scaling
            initializer_map (dict[str, onnx.TensorProto]):
                Map of initializer names to tensor data.

        Returns:
            list[onnx.NodeProto]: The modified node (possibly unchanged).

        Raises:
            HandlerImplementationError: If tensor is unreadable
        """
        _ = initializer_map
        self.validate_node_has_output(node)

        output_name = node.output[0]

        is_data_constant = any(
            output_name in n.input and n.op_type in self.DATA_OPS for n in graph.node
        )

        if not is_data_constant:
            # Skip quantization for non-numeric constants
            return [node]

        # Safe to quantize: numeric constant used in computation
        for attr in node.attribute:
            if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                try:
                    arr = numpy_helper.to_array(attr.t).astype(np.float64)
                except (ValueError, Exception) as e:
                    raise HandlerImplementationError(
                        op_type="Constant",
                        message="Failed to read tensor from Constant node"
                        f" '{node.name}': {e}",
                    ) from e

                arr *= self.get_scaling(
                    scale_config.base,
                    scale_config.exponent,
                )
                attr.t.CopyFrom(numpy_helper.from_array(arr, name=""))

        node.name += "_quant"
        return [node]

    def check_supported(
        self: ConstantQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        """All Constant nodes are supported... For now.

        Args:
            node (onnx.NodeProto): Node to be checked
            initializer_map (dict[str, onnx.TensorProto], optional):
                Map of initializer names to tensor data. Defaults to None.
        """
        _ = node, initializer_map
