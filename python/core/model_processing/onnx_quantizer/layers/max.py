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
    return tuple(int(d) for d in t.dims)


class MaxQuantizer(BaseOpQuantizer):
    """
    Quantizer for ONNX Max (elementwise).

    Policy:
      - Scale *all* inputs by S = base^exponent, picking a float or int scale
        to match the input dtype (avoid Mul type mismatches), then Cast to INT64.
      - No broadcasting (yet): initializer-backed inputs must have identical shapes.
      - Variadic ≥ 1 input. ONNX Max has no attributes.
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
        _ = graph
        if not isinstance(node, onnx.NodeProto):
            raise HandlerImplementationError(
                op_type="Max", message="quantize() expected an ONNX NodeProto"
            )

        # Prepare scale constants: float64 and int64 forms
        S = self.get_scaling(scale_config.base, scale_config.exponent)

        scale_f_name = f"{node.name}_scale_f64"
        scale_i_name = f"{node.name}_scale_i64"

        scale_f = numpy_helper.from_array(
            np.array([S], dtype=np.float64), name=scale_f_name
        )
        scale_i = numpy_helper.from_array(
            np.array([S], dtype=np.int64), name=scale_i_name
        )

        # register both; only the one we use will be referenced
        self.new_initializers.append(scale_f)
        self.new_initializers.append(scale_i)

        new_nodes: List[onnx.NodeProto] = []
        casted_inputs: List[str] = []

        for idx, inp in enumerate(node.input):
            # Choose the scale to match input dtype when we can see it.
            use_int_scale = False
            init_t = initializer_map.get(inp)
            if init_t is not None:
                use_int_scale = init_t.data_type == onnx.TensorProto.INT64

            chosen_scale = scale_i_name if use_int_scale else scale_f_name

            mul_out = f"{node.name}_in{idx}_scaled"
            cast_out = f"{mul_out}_i64"

            mul_node = helper.make_node(
                "Mul",
                inputs=[inp, chosen_scale],
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

        max_node = helper.make_node(
            "Max",
            inputs=casted_inputs,
            outputs=list(node.output),
            name=node.name,
        )
        new_nodes.append(max_node)
        return new_nodes

    def check_supported(
        self,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
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

        if len(shapes) >= 2:
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
