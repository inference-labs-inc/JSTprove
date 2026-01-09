from __future__ import annotations

from typing import TYPE_CHECKING

from onnx import helper

if TYPE_CHECKING:
    import onnx

from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    ScaleConfig,
)


class SoftmaxQuantizer(BaseOpQuantizer):
    """
    Quantizer for ONNX Softmax layers.

    Replaces standard Softmax with Int64Softmax from the `ai.onnx.contrib` domain.
    The Int64Softmax op:
    1. Unscales the int64 input to float
    2. Clamps to [-8, 8) to match Rust circuit behavior
    3. Computes exp and normalizes
    4. Rescales back to int64
    """

    def __init__(
        self: SoftmaxQuantizer,
        new_initializers: list[onnx.TensorProto] | None = None,
    ) -> None:
        super().__init__()
        if new_initializers is not None:
            self.new_initializers = new_initializers

    def quantize(
        self: SoftmaxQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        _ = graph, initializer_map

        scale_value = self.get_scaling(scale_config.base, scale_config.exponent)

        axis = -1
        for attr in node.attribute:
            if attr.name == "axis":
                axis = attr.i
                break

        quantized_node = helper.make_node(
            "Int64Softmax",
            inputs=list(node.input),
            outputs=list(node.output),
            name=node.name,
            domain="ai.onnx.contrib",
            scale=scale_value,
            axis=axis,
        )

        return [quantized_node]

    def check_supported(
        self: SoftmaxQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        _ = node, initializer_map
