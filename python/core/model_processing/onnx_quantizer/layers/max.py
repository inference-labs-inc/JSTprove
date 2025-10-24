from __future__ import annotations

from typing import Tuple

import onnx
from onnx import TensorProto, helper

from python.core.model_processing.onnx_quantizer.exceptions import (
    HandlerImplementationError,
    InvalidParamError,
)
from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    ScaleConfig,
)


def _tensor_shape(t: onnx.TensorProto) -> Tuple[int, ...]:
    """Return the static dims of an initializer as a Python tuple."""
    return tuple(int(d) for d in t.dims)


class MaxQuantizer(BaseOpQuantizer):
    """
    Quantizer for ONNX Max (elementwise) op.

    v1 policy:
      - Cast all inputs to INT64 so ORT sees a uniform dtype.
      - No broadcasting: if ≥2 initializer-backed inputs are visible, shapes must match.
      - ONNX Max has no attributes; any present attribute is an error.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()

    # -------------------- Quantization --------------------
    def quantize(
        self,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        _ = graph, scale_config, initializer_map
        if not isinstance(node, onnx.NodeProto):
            raise HandlerImplementationError(
                op_type="Max", message="quantize() expected an ONNX NodeProto"
            )

        # ORT requires all Max inputs to share the same dtype.
        # Simplest robust policy: cast every input to INT64 unconditionally.
        inputs = list(node.input)
        if not inputs:
            return [node]

        cast_nodes: list[onnx.NodeProto] = []
        new_inputs: list[str] = []

        for i, name in enumerate(inputs):
            cast_out = f"{name}__i64"
            cast_nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[name],
                    outputs=[cast_out],
                    name=f"{node.name or 'Max'}_cast_{i}",
                    to=TensorProto.INT64,
                )
            )
            new_inputs.append(cast_out)

        new_max = helper.make_node(
            "Max",
            inputs=new_inputs,
            outputs=list(node.output),
            name=node.name,
        )
        return [*cast_nodes, new_max]

    # -------------------- Model checker --------------------
    def check_supported(
        self,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        """
        Enforce constraints we support now:
          - at least one input (variadic >= 1)
          - no attributes at all for Max
          - no broadcasting: if ≥2 initializer-backed inputs are visible, shapes must match
        """
        if len(node.input) < 1:
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message="Max requires at least 1 input",
            )

        if node.attribute and len(node.attribute) > 0:
            unexpected = ", ".join(a.name for a in node.attribute)
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message=f"Unexpected attributes for Max: {unexpected}",
            )

        if not initializer_map:
            return

        shapes: list[Tuple[int, ...]] = []
        for name in node.input:
            t = initializer_map.get(name)
            if t is not None and t.dims:
                shapes.append(_tensor_shape(t))

        if len(shapes) < 2:
            return

        first = shapes[0]
        if any(s != first for s in shapes[1:]):
            pretty = ", ".join(str(s) for s in shapes)
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message=(
                    "Broadcasting is not supported for Max. "
                    f"All inputs must have identical shapes; got {pretty}"
                ),
            )
