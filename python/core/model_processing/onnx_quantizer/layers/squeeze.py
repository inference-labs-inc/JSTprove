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


class QuantizeSqueeze(QuantizerBase):
    OP_TYPE = "Squeeze"
    DOMAIN = ""
    USE_WB = False
    USE_SCALING = False
    # Only the data input is relevant for scale-planning.
    SCALE_PLAN: ClassVar = {0: 1}


class SqueezeQuantizer(BaseOpQuantizer, QuantizeSqueeze):
    """
    Quantizer for ONNX Squeeze.

    Squeeze is scale-preserving (pure shape/view transform):
    - No arithmetic
    - No rescaling
    - No custom op

    We support:
    - axes as an attribute (older opsets)
    - axes as a constant initializer input (newer opsets)

    We do NOT support dynamic axes provided at runtime.
    """

    def __init__(
        self: SqueezeQuantizer,
        new_initializers: list[onnx.TensorProto] | None = None,
    ) -> None:
        super().__init__()
        if new_initializers is not None:
            self.new_initializers = new_initializers

    def quantize(
        self: SqueezeQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        # Pure passthrough; QuantizerBase handles standard bookkeeping.
        return QuantizeSqueeze.quantize(
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
                    "Dynamic axes input is not supported for Squeeze "
                    f"(expected axes '{axes_name}' to be an initializer)."
                ),
            )

        axes_tensor = initializer_map[axes_name]
        arr = numpy_helper.to_array(axes_tensor)

        if not np.issubdtype(arr.dtype, np.integer):
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message=f"Squeeze axes initializer must be integer, got {arr.dtype}.",
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
            message=f"Squeeze axes initializer must be 0-D or 1-D, got {arr.ndim}-D.",
            attr_key="axes",
            expected="0-D scalar or 1-D list of axes",
        )

    def check_supported(
        self: SqueezeQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        self.validate_node_has_output(node)
        initializer_map = initializer_map or {}

        if len(node.input) not in (self._N_INPUTS_NO_AXES, self._N_INPUTS_WITH_AXES):
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message=f"Squeeze expects 1 or 2 inputs, got {len(node.input)}.",
            )

        axes = self._get_axes_from_attribute(node)
        if axes is None and len(node.input) == self._N_INPUTS_WITH_AXES:
            axes = self._get_axes_from_initializer_input(node, initializer_map)

        # If axes omitted entirely: ONNX removes all dims of size 1.
        # We can't validate that here without shape info.
        if axes is None:
            return

        if len(set(axes)) != len(axes):
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message=f"axes must not contain duplicates: {axes}",
                attr_key="axes",
                expected="axes list with unique entries",
            )
