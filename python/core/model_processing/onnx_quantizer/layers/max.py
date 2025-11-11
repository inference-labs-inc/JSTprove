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
    Quantizer for ONNX Max (variadic elementwise).

    Policy:
      - Multiply *each* input by S = base**exponent using a scale tensor
        that matches the input’s numeric family (float vs int), then Cast to INT64.
      - No broadcasting (for now): if multiple inputs are *initializers*, their shapes must match.
      - ONNX Max has no attributes.
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

        # Prepare scale = base**exponent in both float64 and int64 forms
        S = self.get_scaling(scale_config.base, scale_config.exponent)

        scale_f_name = f"{node.name}_scale_f64"
        scale_i_name = f"{node.name}_scale_i64"

        scale_f = numpy_helper.from_array(
            np.array([S], dtype=np.float64), name=scale_f_name
        )
        scale_i = numpy_helper.from_array(
            np.array([S], dtype=np.int64), name=scale_i_name
        )

        self.new_initializers.extend([scale_f, scale_i])

        new_nodes: List[onnx.NodeProto] = []
        casted_inputs: List[str] = []

        for idx, inp in enumerate(node.input):
            # prefer int scale if the input is an INT initializer
            use_int_scale = False
            init_t = initializer_map.get(inp)
            if init_t is not None:
                use_int_scale = init_t.data_type == onnx.TensorProto.INT64

            chosen_scale = scale_i_name if use_int_scale else scale_f_name

            mul_out = f"{node.name}_in{idx}_scaled"
            cast_out = f"{mul_out}_i64"

            new_nodes.append(
                helper.make_node(
                    "Mul",
                    inputs=[inp, chosen_scale],
                    outputs=[mul_out],
                    name=f"{node.name}_scale_in{idx}",
                )
            )
            new_nodes.append(
                helper.make_node(
                    "Cast",
                    inputs=[mul_out],
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
        initializer_map: dict[str, onnx.TensorProto],
    ) -> None:
        # arity/attrs
        if len(node.input) < 1:
            raise InvalidParamError(
                node.name, node.op_type, "Max requires at least 1 input"
            )
        if node.attribute:
            unexpected = ", ".join(a.name for a in node.attribute)
            raise InvalidParamError(
                node.name, node.op_type, f"Unexpected attributes for Max: {unexpected}"
            )

        # broadcasting guard for initializer-backed inputs
        # (only enforce when >=2 inputs are initializers with known dims)
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
                    node.name,
                    node.op_type,
                    "Broadcasting is not supported for Max. "
                    f"All initializer inputs must have identical shapes; got {pretty}",
                )
