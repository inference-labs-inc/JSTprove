from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from onnx import helper, numpy_helper

if TYPE_CHECKING:
    import onnx

from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    ScaleConfig,
)


class SigmoidQuantizer(BaseOpQuantizer):
    """
    Quantizer for ONNX Sigmoid layers.

    Replaces standard Sigmoid with Int64Sigmoid from the `ai.onnx.contrib` domain.
    The Int64Sigmoid op:
    1. Unscales the int64 input to float
    2. Clamps to [-8, 8) to match Rust circuit behavior
    3. Applies sigmoid
    4. Rescales back to int64
    """

    def __init__(
        self: SigmoidQuantizer,
        new_initializers: list[onnx.TensorProto] | None = None,
    ) -> None:
        super().__init__()
        if new_initializers is not None:
            self.new_initializers = new_initializers

    def quantize(
        self: SigmoidQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        _ = graph, initializer_map

        scale_value = self.get_scaling(scale_config.base, scale_config.exponent)

        quantized_node = helper.make_node(
            "Int64Sigmoid",
            inputs=list(node.input),
            outputs=list(node.output),
            name=node.name,
            domain="ai.onnx.contrib",
            scale=scale_value,
        )

        return [quantized_node]

    def check_supported(
        self: SigmoidQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        _ = node, initializer_map
