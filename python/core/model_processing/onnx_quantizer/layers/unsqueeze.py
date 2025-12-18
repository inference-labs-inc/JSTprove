from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from onnx import numpy_helper

from python.core.model_processing.onnx_quantizer.exceptions import (
    InvalidParamError,
)
from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    QuantizerBase,
    ScaleConfig,
)

if TYPE_CHECKING:
    import onnx


class QuantizeUnsqueeze(QuantizerBase):
    OP_TYPE = "Unsqueeze"
    DOMAIN = ""
    USE_WB = False
    USE_SCALING = False
    # Only the data input is relevant for scale-planning.
    SCALE_PLAN: ClassVar = {0: 1}


class UnsqueezeQuantizer(BaseOpQuantizer, QuantizeUnsqueeze):
    """
    Quantizer for ONNX Unsqueeze.

    Unsqueeze is scale-preserving (pure shape/view transform):
    - No arithmetic
    - No rescaling
    - No custom op

    Semantics:
    - Inserts new dimensions of size 1 at the specified axes positions.

    We support:
    - axes as an attribute (older opsets)
    - axes as a constant initializer input (opset >= 13 style)

    We do NOT support dynamic axes provided at runtime.
    """

    def __init__(
        self: UnsqueezeQuantizer,
        new_initializers: list[onnx.TensorProto] | None = None,
    ) -> None:
        super().__init__()
        if new_initializers is not None:
            self.new_initializers = new_initializers

    def quantize(
        self: UnsqueezeQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        # Pure passthrough; QuantizerBase handles standard bookkeeping.
        return QuantizeUnsqueeze.quantize(
            self,
            node,
            graph,
            scale_config,
            initializer_map,
        )

    _N_INPUTS_NO_AXES: ClassVar[int] = 1
    _N_INPUTS_WITH_AXES: ClassVar[int] = 2

    def _get_axes_from_attribute(self, node: onnx.NodeProto) -> list[int] | None:
        for attr in node.attribute:
            if attr.name == "axes":
                return list(attr.ints)
        return None

    def _get_axes_from_initializer_input(
        self,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[int]:
        axes_name = node.input[1]
        if axes_name not in initializer_map:
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message=(
                    "Unsqueeze expects either 1 input (axes as attribute) "
                    "or 2 inputs (axes as initializer), got {n_inputs}."
                ),
            )

        axes_tensor = initializer_map[axes_name]
        arr = numpy_helper.to_array(axes_tensor)

        if not np.issubdtype(arr.dtype, np.integer):
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message=f"Unsqueeze axes initializer must be integer, got {arr.dtype}.",
                attr_key="axes",
                expected="integer tensor (0-D or 1-D)",
            )

        if arr.ndim == 0:
            return [int(arr)]
        if arr.ndim == 1:
            return [int(x) for x in arr.tolist()]

        raise InvalidParamError(
            node_name=node.name,
            op_type=node.op_type,
            message=f"Unsqueeze axes initializer must be 0-D or 1-D, got {arr.ndim}-D.",
            attr_key="axes",
            expected="0-D scalar or 1-D list of axes",
        )

    def check_supported(
        self: UnsqueezeQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        self.validate_node_has_output(node)
        initializer_map = initializer_map or {}

        n_inputs = len(node.input)
        axes = self._get_axes_from_attribute(node)

        # ONNX Unsqueeze has two schema styles:
        #  - newer: Unsqueeze(data, axes)  -> 2 inputs, no attribute required
        #  - older: Unsqueeze(data) with axes attribute -> 1 input, attribute required
        if n_inputs == self._N_INPUTS_NO_AXES:
            if axes is None:
                raise InvalidParamError(
                    node_name=node.name,
                    op_type=node.op_type,
                    message=(
                        "Unsqueeze with 1 input is only supported when 'axes' is "
                        " provided as an attribute (older opsets)."
                    ),
                    attr_key="axes",
                    expected="axes attribute",
                )
        elif n_inputs == self._N_INPUTS_WITH_AXES:
            # If axes is provided as a second input, it must be a constant initializer.
            if axes is None:
                axes = self._get_axes_from_initializer_input(node, initializer_map)
        else:
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message=f"Unsqueeze expects 1 or 2 inputs, got {n_inputs}.",
            )

        # At this point, axes must be known.
        if len(set(axes)) != len(axes):
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message=f"axes must not contain duplicates: {axes}",
                attr_key="axes",
                expected="axes list with unique entries",
            )
