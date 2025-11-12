from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import onnx

from python.core.model_processing.onnx_quantizer.exceptions import InvalidParamError
from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    QuantizerBase,
    ScaleConfig,
)


class QuantizeGemm(QuantizerBase):
    OP_TYPE = "Int64Gemm"
    USE_WB = True
    USE_SCALING = True
    DEFAULT_ATTRS: ClassVar = {"transA": 0, "transB": 0}
    SCALE_PLAN: ClassVar = {1: 1, 2: 2}


class GemmQuantizer(BaseOpQuantizer, QuantizeGemm):
    """
    Quantizer for ONNX Gemm layers.

    - Replaces standard Gemm with Int64Gemm from the `ai.onnx.contrib`
        domain and makes relevant additional changes to the graph.
    - Validates that all required Gemm parameters are present.
    """

    def __init__(
        self: GemmQuantizer,
        new_initializers: list[onnx.TensorProto] | None = None,
    ) -> None:
        super().__init__()
        # Only replace if caller provided something
        if new_initializers is not None:
            self.new_initializers = new_initializers

    def quantize(
        self: GemmQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        return QuantizeGemm.quantize(self, node, graph, scale_config, initializer_map)

    def check_supported(
        self: GemmQuantizer,
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
