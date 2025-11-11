# python/core/model_processing/onnx_quantizer/layers/max.py

from __future__ import annotations
from typing import List
import onnx
from onnx import helper

from python.core.model_processing.onnx_quantizer.exceptions import InvalidParamError
from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    ScaleConfig,
)


class MaxQuantizer(BaseOpQuantizer):
    """
    Passthrough INT64 quantizer for ONNX Max (elementwise).
    Converts all inputs to INT64 and applies a regular ONNX Max.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()

    def quantize(
        self,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> List[onnx.NodeProto]:
        """
        Simple passthrough quantization:
          - Cast each input to INT64
          - Apply standard Max on those casts
        """
        _ = graph, scale_config, initializer_map
        new_nodes: List[onnx.NodeProto] = []
        casted_inputs: List[str] = []

        for idx, inp_name in enumerate(node.input):
            cast_out = f"{node.name}_in{idx}_i64"
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[inp_name],
                    outputs=[cast_out],
                    to=onnx.TensorProto.INT64,
                    name=f"{node.name}_cast_in{idx}",
                )
            )
            casted_inputs.append(cast_out)

        new_nodes.append(
            helper.make_node(
                "Max",
                inputs=casted_inputs,
                outputs=list(node.output),
                name=node.name,
            )
        )
        return new_nodes

    def check_supported(
        self,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        """
        Basic safety checks for Max node.
        """
        if len(node.input) < 1:
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message="Max requires at least one input.",
            )

        if node.attribute:
            unexpected = ", ".join(a.name for a in node.attribute)
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message=f"Unexpected attributes for Max: {unexpected}",
            )
