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


def _has_any_attributes(node: onnx.NodeProto) -> bool:
    return bool(getattr(node, "attribute", None))


def _elem_type_is_int64(t: TensorProto) -> bool:
    return getattr(t, "data_type", None) == onnx.TensorProto.INT64


class MaxQuantizer(BaseOpQuantizer):
    """
    Elementwise Max (variadic). v1 policy:
      • passthrough quantization (no rewrite)
      • no broadcasting yet → all inputs must match shape
      • expect INT64 fixed-point domain for known initializers
    """

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()

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

    def check_supported(
        self,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        # 1) ONNX Max is variadic but requires >= 2 inputs
        if len(node.input) < 2:
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message="Max requires ≥ 2 inputs",
            )

        # 2) Max has no attributes; reject any we see so users get a clear error
        if _has_any_attributes(node):
            names = [a.name for a in node.attribute]
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message=f"Unexpected attributes for Max (none supported): {names}",
            )

        if not initializer_map:
            return  # nothing else we can check statically

        # 3) If any inputs are initializers, enforce:
        #    • identical shapes (no broadcasting yet)
        #    • (optional) INT64 element type
        shapes = []
        bad_dtypes = []
        for name in node.input:
            t = initializer_map.get(name)
            if t is None:
                continue
            if t.dims:
                shapes.append(_tensor_shape(t))
            if not _elem_type_is_int64(t):
                bad_dtypes.append(name)

        if len(shapes) > 1:
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

        if bad_dtypes:
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message=(
                    "Max expects INT64 initializers in the fixed-point domain; "
                    f"non-INT64 inputs found for: {bad_dtypes}"
                ),
            )
