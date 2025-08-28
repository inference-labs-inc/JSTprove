from __future__ import annotations

import onnx

from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    ScaleConfig,
)


class ReluQuantizer(BaseOpQuantizer):
    """
    Quantizer for ONNX ReLU layers.

    - Replaces standard ReLU with Int64ReLU from the `ai.onnx.contrib` domain
        and makes relevant additional changes to the graph.
    - Validates that all required ReLU parameters are present.
    """

    def __init__(
        self: BaseOpQuantizer,
        new_initializer: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        super().__init__()
        _ = new_initializer

    def quantize(
        self: BaseOpQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        """
        Quantize a node by converting the node to Int64 version

        Args:
            node (onnx.NodeProto): The node to quantize.
            rescale (bool): Whether rescaling is enabled
                (Doesnt have an affect on this op type)
            graph (onnx.GraphProto): The ONNX graph.
            scale_exponent (int): Scale exponent.
            scale_base (int): The base of scaling.
            initializer_map (dict[str, onnx.TensorProto]):
                Map of initializer names to tensor data.

        Returns:
            List[onnx.NodeProto]: The quantized ONNX node.
        """
        _ = graph, scale_config, initializer_map
        return [
            onnx.helper.make_node(
                "Int64Relu",
                inputs=node.input,
                outputs=node.output,  # preserve original output name
                name=node.name,
                domain="ai.onnx.contrib",
            ),
        ]

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
        """
        _ = node
        _ = initializer_map
