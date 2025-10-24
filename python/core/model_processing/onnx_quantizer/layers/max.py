# python/core/model_processing/onnx_quantizer/layers/max.py
from __future__ import annotations

from typing import Tuple, List

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from python.core.model_processing.onnx_quantizer.exceptions import (
    HandlerImplementationError,
    InvalidParamError,
)
from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    ScaleConfig,
)


def _tensor_shape(t: TensorProto) -> Tuple[int, ...]:
    """Return the static dims of an initializer as a Python tuple."""
    return tuple(int(d) for d in t.dims)


class MaxQuantizer(BaseOpQuantizer):
    """
    Quantizer for ONNX Max (elementwise) op.

    Policy:
      - Scale *all* inputs by S = base^exponent, then Cast to INT64, then run Max.
      - No broadcasting (yet): if we can see >1 initializer-backed inputs, their shapes must match.
      - Allow ≥1 inputs (variadic). If only one input is provided, it's a passthrough after scale+cast.
      - ONNX Max has no attributes.
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
    ) -> List[onnx.NodeProto]:
        """
        For each input:
          input --Mul(S)--> scaled --Cast(INT64)--> casted_input
        Then build a new Max over the casted inputs.

        We do not walk graph.node (keeps tests' graph-mocks happy).
        """
        _ = graph, initializer_map  # not needed here; avoid iterating graph.node
        if not isinstance(node, onnx.NodeProto):
            raise HandlerImplementationError(
                op_type="Max", message="quantize() expected an ONNX NodeProto"
            )

        # Make a per-node scale initializer S = base^exponent (float64 so Mul matches float inputs)
        S = self.get_scaling(scale_config.base, scale_config.exponent)
        scale_name = f"{node.name}_scale_const"
        scale_tensor = numpy_helper.from_array(
            np.array([S], dtype=np.float64), name=scale_name
        )
        self.new_initializers.append(scale_tensor)

        new_nodes: List[onnx.NodeProto] = []
        casted_inputs: List[str] = []

        # Rewire each input: Mul(input, S) -> Cast(INT64)
        for idx, inp in enumerate(node.input):
            mul_out = f"{node.name}_in{idx}_scaled"
            cast_out = f"{mul_out}_i64"

            mul_node = helper.make_node(
                "Mul",
                inputs=[inp, scale_name],
                outputs=[mul_out],
                name=f"{node.name}_scale_in{idx}",
            )
            cast_node = helper.make_node(
                "Cast",
                inputs=[mul_out],
                outputs=[cast_out],
                to=onnx.TensorProto.INT64,
                name=f"{node.name}_cast_in{idx}",
            )

            new_nodes.append(mul_node)
            new_nodes.append(cast_node)
            casted_inputs.append(cast_out)

        # Final Max over the int64, equally-scaled inputs
        max_node = helper.make_node(
            "Max",
            inputs=casted_inputs,
            outputs=list(node.output),  # preserve original output names
            name=node.name,
        )
        new_nodes.append(max_node)

        return new_nodes

    # -------------------- Model checker --------------------
    def check_supported(
        self,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        """
        Constraints we support now:
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
            return  # nothing more we can check statically

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
