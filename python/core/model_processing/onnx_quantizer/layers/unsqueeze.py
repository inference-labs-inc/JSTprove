from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import numpy as np
from onnx import helper, numpy_helper

from python.core.model_processing.converters.base import ModelType
from python.core.model_processing.errors import LayerAnalysisError
from python.core.model_processing.onnx_custom_ops.onnx_helpers import parse_attributes
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

_N_UNSQUEEZE_INPUTS: int = 2


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

    def pre_analysis_transform(
        self: UnsqueezeQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        initializer_map: dict[str, onnx.TensorProto],
        scale_base: int,
        scale_exponent: int,
    ) -> None:
        _ = initializer_map, scale_base, scale_exponent
        model_type = ModelType.ONNX
        params = parse_attributes(node.attribute)
        if node.op_type != "Unsqueeze":
            return
        if params and "axes" in params:
            return
        axes = _extract_unsqueeze_axes_into_params(
            name=node.name,
            inputs=node.input,
            params=params,
            graph=graph,
            model_type=model_type,
            initializer_map=initializer_map,
        )
        attr = helper.make_attribute("axes", axes["axes"])
        node.attribute.append(attr)

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
                    f"Dynamic axes input is not supported for Unsqueeze "
                    f"(expected axes '{axes_name}' to be an initializer)."
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
        scale_base: int | None = 2,
        scale_exponent: int | None = 18,
    ) -> None:
        _, _ = scale_base, scale_exponent
        self.validate_node_has_output(node)
        initializer_map = initializer_map or {}

        n_inputs = len(node.input)
        if n_inputs not in (self._N_INPUTS_NO_AXES, self._N_INPUTS_WITH_AXES):
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message=(
                    "Unsqueeze expects either 1 input (axes as attribute) or 2 inputs "
                    f"(axes as initializer), got {n_inputs}."
                ),
            )

        axes = self._get_axes_from_attribute(node)

        # ONNX Unsqueeze has two schema styles:
        #  - newer: Unsqueeze(data, axes) -> 2 inputs, axes is initializer input
        #  - older: Unsqueeze(data) with axes attribute -> 1 input
        if n_inputs == self._N_INPUTS_NO_AXES:
            if axes is None:
                raise InvalidParamError(
                    node_name=node.name,
                    op_type=node.op_type,
                    message=(
                        "Unsqueeze with 1 input is only supported when 'axes' is "
                        "provided as an attribute (older opsets)."
                    ),
                    attr_key="axes",
                    expected="axes attribute",
                )
        elif axes is None:
            axes = self._get_axes_from_initializer_input(node, initializer_map)

        if axes is None:
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message="Unsqueeze requires 'axes' to be provided.",
                attr_key="axes",
                expected="axes attribute or initializer input",
            )

        if len(set(axes)) != len(axes):
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message=f"axes must not contain duplicates: {axes}",
                attr_key="axes",
                expected="axes list with unique entries",
            )


def _extract_unsqueeze_axes_into_params(  # noqa: PLR0913
    *,
    name: str,
    inputs: list[str] | tuple[str, ...],
    params: dict | None,
    graph: onnx.GraphProto,
    model_type: ModelType,
    initializer_map: dict[str, onnx.TensorProto] | None = None,
) -> dict:
    if len(inputs) != _N_UNSQUEEZE_INPUTS:
        msg = (
            f"Unsqueeze '{name}' is missing axes input. "
            f"Expected 2 inputs (data, axes), got {len(inputs)}: {list(inputs)}"
        )
        raise LayerAnalysisError(model_type=model_type, reason=msg)

    axes_name = inputs[1]

    axes_arr = _resolve_unsqueeze_axes_array(
        name=name,
        axes_name=axes_name,
        graph=graph,
        model_type=model_type,
        initializer_map=initializer_map,
    )

    _validate_unsqueeze_axes_are_integer(
        name=name,
        axes_arr=axes_arr,
        model_type=model_type,
    )

    out_params = params or {}
    out_params["axes"] = _axes_array_to_int_list(axes_arr)
    return out_params


def _resolve_unsqueeze_axes_array(
    *,
    name: str,
    axes_name: str,
    graph: onnx.GraphProto,
    model_type: ModelType,
    initializer_map: dict[str, onnx.TensorProto] | None = None,
) -> np.ndarray:
    if not initializer_map:
        initializer_map = {init.name: init for init in graph.initializer}

    if axes_name in initializer_map:
        return numpy_helper.to_array(initializer_map[axes_name])

    const_tensor = _find_constant_tensor_by_output_name(
        graph=graph,
        output_name=axes_name,
    )

    if const_tensor is not None:
        return numpy_helper.to_array(const_tensor)

    msg = (
        f"Unsqueeze '{name}' has dynamic axes input '{axes_name}'. "
        "Only constant initializer axes or Constant-node axes are supported."
    )
    raise LayerAnalysisError(model_type=model_type, reason=msg)


def _find_constant_tensor_by_output_name(
    *,
    graph: onnx.GraphProto,
    output_name: str,
) -> onnx.TensorProto | None:
    for n in graph.node:
        if n.op_type != "Constant" or not n.output:
            continue
        if n.output[0] != output_name:
            continue

        for attr in n.attribute:
            if attr.name == "value" and attr.t is not None:
                return attr.t

        # Constant node exists but doesn't have the expected tensor attribute.
        return None

    return None


def _validate_unsqueeze_axes_are_integer(
    *,
    name: str,
    axes_arr: np.ndarray,
    model_type: ModelType,
) -> None:
    if not np.issubdtype(axes_arr.dtype, np.integer):
        msg = f"Unsqueeze '{name}' axes must be integer, got dtype {axes_arr.dtype}."
        raise LayerAnalysisError(model_type=model_type, reason=msg)


def _axes_array_to_int_list(axes_arr: np.ndarray) -> list[int]:
    if axes_arr.ndim == 0:
        return [int(axes_arr)]
    return [int(x) for x in axes_arr.reshape(-1).tolist()]
