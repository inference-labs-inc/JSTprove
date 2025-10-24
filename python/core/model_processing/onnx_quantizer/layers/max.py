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
    """Return the static dims of an initializer as a Python tuple."""
    return tuple(int(d) for d in t.dims)


def _dtype_of_value(name: str, graph: onnx.GraphProto) -> int | None:
    """
    Best-effort: find elem_type of a tensor value by scanning graph inputs/outputs/value_info,
    or infer from an initializer if present.
    Returns ONNX TensorProto data type enum or None if unknown.
    """

    # 1) look at value_info / graph inputs / outputs
    def _search(vinfos):
        for vi in vinfos:
            if vi.name == name:
                t = vi.type.tensor_type
                if t.elem_type:
                    return int(t.elem_type)
        return None

    dt = _search(graph.input) or _search(graph.value_info) or _search(graph.output)
    if dt is not None:
        return dt

    # 2) fall back to initializers
    for init in graph.initializer:
        if init.name == name:
            return int(init.data_type) if init.data_type else None

    return None


def _make_cast_node(inp: str, out: str, to_dtype: int, *, name: str) -> onnx.NodeProto:
    return onnx.helper.make_node(
        "Cast",
        inputs=[inp],
        outputs=[out],
        name=name,
        to=to_dtype,
    )


class MaxQuantizer(BaseOpQuantizer):
    """
    Quantizer for ONNX Max (elementwise) op.

    v1 policy:
      - No rewrite: ONNX Max accepts int64; outputs stay in the same fixed-point domain.
      - No broadcasting (yet): if we can see >1 initializer-backed inputs, their shapes must match.
      - Allow ≥1 inputs (variadic). If only one input is provided, it's a passthrough.
      - ONNX Max has no attributes; any present attribute is an error.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        super().__init__()

    # -------------------- Quantization (passthrough) --------------------
    def quantize(
        self,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        _ = scale_config, initializer_map
        if not isinstance(node, onnx.NodeProto):
            raise HandlerImplementationError(
                op_type="Max", message="quantize() expected an ONNX NodeProto"
            )

        # Ensure all inputs have the same dtype for ORT:
        # Pick the dtype of the first input we can resolve; cast others to match.
        inputs = list(node.input)
        if not inputs:
            return [node]

        target_dtype = None
        for name in inputs:
            target_dtype = _dtype_of_value(name, graph)
            if target_dtype is not None:
                break

        if target_dtype is None:
            # We couldn't determine any dtype statically; leave node unchanged.
            return [node]

        new_nodes: list[onnx.NodeProto] = []
        new_inputs: list[str] = []

        for idx, name in enumerate(inputs):
            dt = _dtype_of_value(name, graph)
            if dt is None or dt == target_dtype:
                new_inputs.append(name)
                continue

            cast_out = f"{name}__cast_to_{target_dtype}"
            cast_node = _make_cast_node(
                name, cast_out, target_dtype, name=f"{node.name or 'Max'}_cast_{idx}"
            )
            new_nodes.append(cast_node)
            new_inputs.append(cast_out)

        # Rewire Max inputs if we inserted any casts
        if new_nodes:
            node = onnx.helper.make_node(
                "Max",
                inputs=new_inputs,
                outputs=list(node.output),
                name=node.name,
            )
            return [*new_nodes, node]

        return [node]

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
        # 1) Arity: ONNX allows >=1 inputs
        if len(node.input) < 1:
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message="Max requires at least 1 input",
            )

        # 2) Attributes: ONNX Max has no attributes
        if node.attribute and len(node.attribute) > 0:
            unexpected = ", ".join(a.name for a in node.attribute)
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message=f"Unexpected attributes for Max: {unexpected}",
            )

        # 3) Shape check (no broadcasting): compare shapes of any initializer-backed inputs we can see
        if not initializer_map:
            return  # nothing more we can check statically

        shapes: list[Tuple[int, ...]] = []
        for name in node.input:
            t = initializer_map.get(name)
            if t is not None and t.dims:
                shapes.append(_tensor_shape(t))

        # Need at least two known shapes to detect a mismatch
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
