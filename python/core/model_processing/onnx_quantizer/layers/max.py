# python/core/model_processing/onnx_quantizer/layers/max.py
from __future__ import annotations

from typing import Tuple

import onnx
from onnx import TensorProto

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
    Quantizer for ONNX Max (elementwise) op.

    v1 policy:
      - Passthrough quantization (ONNX Max supports INT64).
      - No broadcasting yet: all inputs must share the same shape.
      - If an input is provided as an initializer, it must be INT64.
      - No attributes are supported for Max.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()

    # Quantization: passthrough
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
        return [node]

    # Model checker
    def check_supported(
        self,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        # 1) Arity
        if len(node.input) < 2:
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message="Max requires >= 2 inputs",
            )

        # 2) Attributes (Max has none)
        if node.attribute and len(node.attribute) > 0:
            # List the unexpected attributes to aid debugging
            names = ", ".join(a.name for a in node.attribute)
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message=f"Unexpected attributes for Max: {names}",
            )

        if not initializer_map:
            return  # nothing else we can validate statically

        # 3) Collect shapes & dtypes for any initializer-backed inputs
        init_shapes: list[Tuple[int, ...]] = []
        for name in node.input:
            t = initializer_map.get(name)
            if t is None:
                continue
            # dtype must be INT64 (fixed-point domain)
            if t.data_type != onnx.TensorProto.INT64:
                raise InvalidParamError(
                    node_name=node.name,
                    op_type=node.op_type,
                    message=(
                        f"Max expects INT64 initializers; got {onnx.TensorProto.DataType.Name(t.data_type)} "
                        f"for input '{name}'"
                    ),
                )
            if t.dims:
                init_shapes.append(_tensor_shape(t))

        # 4) All known initializer shapes must match (no broadcasting yet)
        if len(init_shapes) > 1:
            first = init_shapes[0]
            if any(s != first for s in init_shapes[1:]):
                pretty = ", ".join(str(s) for s in init_shapes)
                raise InvalidParamError(
                    node_name=node.name,
                    op_type=node.op_type,
                    message=(
                        "Broadcasting is not supported for Max. "
                        f"All inputs must have identical shapes; got {pretty}"
                    ),
                )
