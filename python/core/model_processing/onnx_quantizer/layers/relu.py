from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from onnx import GraphProto, NodeProto, TensorProto

from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    QuantizerBase,
    ScaleConfig,
)


class QuantizeRelu(QuantizerBase):
    OP_TYPE = "Int64Relu"
    USE_WB = False
    USE_SCALING = False


class ReluQuantizer(BaseOpQuantizer, QuantizeRelu):
    """
    Quantizer for ONNX ReLU layers.

    - Replaces standard ReLU with Int64ReLU from the `ai.onnx.contrib` domain
        and makes relevant additional changes to the graph.
    - Validates that all required ReLU parameters are present.
    """

    def __init__(
        self: ReluQuantizer,
        new_initializer: list[TensorProto] | None = None,
    ) -> None:
        super().__init__()
        _ = new_initializer

    def quantize(
        self: ReluQuantizer,
        node: NodeProto,
        graph: GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, TensorProto],
    ) -> list[NodeProto]:
        return QuantizeRelu.quantize(self, node, graph, scale_config, initializer_map)

    def check_supported(
        self: ReluQuantizer,
        node: NodeProto,
        initializer_map: dict[str, TensorProto] | None = None,
    ) -> None:
        """
        Perform high-level validation to ensure that this node
        can be quantized safely.

        Args:
            node (onnx.NodeProto): ONNX node to be checked
            initializer_map (dict[str, onnx.TensorProto]):
                Initializer map (name of weight or bias and tensor)
        """
        _ = node
        _ = initializer_map
