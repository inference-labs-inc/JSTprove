from __future__ import annotations

import numpy as np
import onnx
from onnx import numpy_helper

from python.core.model_processing.onnx_custom_ops.onnx_helpers import extract_attributes
from python.core.model_processing.onnx_quantizer.exceptions import InvalidParamError
from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    ScaleConfig,
)


class GemmQuantizer(BaseOpQuantizer):
    """
    Quantizer for ONNX Gemm layers.

    - Replaces standard Gemm with Int64Gemm from the `ai.onnx.contrib`
        domain and makes relevant additional changes to the graph.
    - Validates that all required Gemm parameters are present.
    """

    def __init__(
        self: BaseOpQuantizer,
        new_initializers: dict[str, onnx.TensorProto],
    ) -> None:
        self.new_initializers = new_initializers

    def quantize(
        self: BaseOpQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        """
        Quantize a Gemm node by:
        1. Quantizing its weights and bias.
        2. Adding a scale constant.
        3. Replacing it with an Int64Gemm node.

        Args:
            node (onnx.NodeProto): The node to quantize.
            rescale (bool): Whether rescaling is enabled
            graph (onnx.GraphProto): The ONNX graph.
            scale_exponent (int): Scale exponent.
            scale_base (int): The base of scaling.
            initializer_map (dict[str, onnx.TensorProto]):
                Map of initializer names to tensor data.

        Returns:
            List[onnx.NodeProto]: A list of ONNX nodes
                (quantized and any auxiliary nodes).
        """
        _ = graph
        nodes = []
        output_name = f"{node.name}_int"

        nodes, new_inputs = self.add_nodes_w_and_b(
            node=node,
            scale_exponent=scale_config.exponent,
            scale_base=scale_config.base,
            initializer_map=initializer_map,
        )
        node.input[:] = new_inputs

        attrs = extract_attributes(node)
        attrs.setdefault("transA", 0)
        attrs.setdefault("transB", 0)
        attrs["rescale"] = int(scale_config.rescale)

        scale_value = self.get_scaling(
            scale_config.base,
            scale_config.exponent,
        )

        # === Create scale constant ===
        scale_const_name = f"{output_name}_scaler"
        scale_tensor = numpy_helper.from_array(
            np.array([scale_value], dtype=np.int64),
            name=scale_const_name,
        )
        self.new_initializers.append(scale_tensor)
        node.input.append(scale_const_name)
        int64_gemm = onnx.helper.make_node(
            "Int64Gemm",
            inputs=node.input,
            outputs=node.output,  # preserve original output name
            name=output_name,
            domain="ai.onnx.contrib",
            **attrs,
        )
        nodes.append(int64_gemm)
        return nodes

    def check_supported(
        self: BaseOpQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        """
        Perform high-level validation to ensure that this node
        can be quantized safely.

        Args:
            node (onnx.NodeProto): ONNX node to be checked
            initializer_map (dict[str, onnx.TensorProto]):
                Initializer map (name of weight or bias and tensor)

        Raises:
            InvalidParamError: If any requirement is not met.
        """
        _ = initializer_map
        num_valid_inputs = 2
        # Ensure inputs exist
        if len(node.input) < num_valid_inputs:
            raise InvalidParamError(
                node.name,
                node.op_type,
                f"Expected at least 2 inputs (input, weights), got {len(node.input)}",
            )
        num_valid_inputs = 3

        if len(node.input) < num_valid_inputs:
            raise InvalidParamError(
                node.name,
                node.op_type,
                "Expected at least 3 inputs (input, weights, bias)"
                f", got {len(node.input)}",
            )

        # Validate attributes with defaults
        attrs = {attr.name: attr for attr in node.attribute}
        alpha = getattr(attrs.get("alpha"), "f", 1.0)
        beta = getattr(attrs.get("beta"), "f", 1.0)
        trans_a = getattr(attrs.get("transA"), "i", 0)
        trans_b = getattr(attrs.get("transB"), "i", 1)

        if alpha != 1.0:
            raise InvalidParamError(
                node.name,
                node.op_type,
                f"alpha value of {alpha} not supported",
                "alpha",
                "1.0",
            )
        if beta != 1.0:
            raise InvalidParamError(
                node.name,
                node.op_type,
                f"beta value of {beta} not supported",
                "beta",
                "1.0",
            )
        if trans_a not in [0, 1]:
            raise InvalidParamError(
                node.name,
                node.op_type,
                f"transA value of {trans_a} not supported",
                "transA",
                "(0,1)",
            )
        if trans_b not in [0, 1]:
            raise InvalidParamError(
                node.name,
                node.op_type,
                f"transB value of {trans_b} not supported",
                "transB",
                "(0,1)",
            )
