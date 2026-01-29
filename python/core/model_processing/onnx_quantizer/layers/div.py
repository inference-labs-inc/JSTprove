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


class DivQuantizer(BaseOpQuantizer):
    """
    Quantizer for ONNX Div layers.

    Div requires casting initializer inputs to int64 but does NOT scale them,
    since scaling would change the division semantics.
    """

    def __init__(
        self: DivQuantizer,
        new_initializers: list[onnx.TensorProto] | None = None,
    ) -> None:
        super().__init__()
        if new_initializers is not None:
            self.new_initializers = new_initializers

    def quantize(
        self: DivQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        _ = graph, scale_config
        nodes = []
        new_inputs = list(node.input)

        for idx, input_name in enumerate(node.input):
            if input_name in initializer_map:
                tensor = initializer_map[input_name]
                arr = numpy_helper.to_array(tensor).astype(np.int64)
                cast_name = f"{input_name}_int64"
                cast_tensor = numpy_helper.from_array(arr, name=cast_name)
                self.new_initializers.append(cast_tensor)
                new_inputs[idx] = cast_name

        quantized_node = helper.make_node(
            "Div",
            inputs=new_inputs,
            outputs=node.output,
            name=node.name,
        )
        nodes.append(quantized_node)
        return nodes

    def check_supported(
        self: DivQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        pass
