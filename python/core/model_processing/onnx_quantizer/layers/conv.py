from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    import onnx

from python.core.model_processing.onnx_quantizer.exceptions import InvalidParamError
from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    QuantizerBase,
    ScaleConfig,
)


class QuantizeConv(QuantizerBase):
    OP_TYPE = "Int64Conv"
    USE_WB = True
    USE_SCALING = True
    DEFAULT_ATTRS: ClassVar = {"group": 1, "auto_pad": "NOTSET"}
    SCALE_PLAN: ClassVar = {1: 1, 2: 2}  # weight = 1x scale, bias = 2x scale


class ConvQuantizer(BaseOpQuantizer, QuantizeConv):
    """
    Quantizer for ONNX Conv layers.

    - Replaces standard Conv with Int64Conv from the `ai.onnx.contrib` domain
        and makes relevant additional changes to the graph.
    - Validates that all required Conv parameters are present.
    """

    def __init__(
        self: ConvQuantizer,
        new_initializers: list[onnx.TensorProto] | None = None,
    ) -> None:
        super().__init__()
        # Only replace if caller provided something
        if new_initializers is not None:
            self.new_initializers = new_initializers

    def quantize(
        self: ConvQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        return QuantizeConv.quantize(self, node, graph, scale_config, initializer_map)

    def check_supported(
        self: ConvQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> None:
        """
        Perform high-level validation to ensure that this Conv node
        can be quantized safely.

        Args:
            node (onnx.NodeProto): ONNX node to be checked
            initializer_map (dict[str, onnx.TensorProto]):
                Initializer map (name of weight or bias and tensor)

        Raises:
            InvalidParamError: If any requirement is not met.
        """
        num_inputs = 2
        if len(node.input) < num_inputs:
            raise InvalidParamError(
                node.name,
                node.op_type,
                f"Expected at least 2 inputs (input, weights), got {len(node.input)}",
            )
        num_inputs = 3

        if len(node.input) < num_inputs:
            raise InvalidParamError(
                node.name,
                node.op_type,
                "Expected at least 3 inputs (input, weights, bias),"
                f" got {len(node.input)}",
            )

        self.check_supported_shape(node, initializer_map)
        self.check_all_params_exist(node)

    def check_all_params_exist(self: ConvQuantizer, node: onnx.NodeProto) -> None:
        """Verify that all required Conv attributes are present.

        Args:
            node (onnx.NodeProto): The Conv node being validated.

        Raises:
            InvalidParamError: If any required parameter is missing.
        """
        # May need: ["strides", "kernel_shape", "dilations", "pads"]
        required_attrs = ["strides", "kernel_shape", "dilations"]
        self.validate_required_attrs(node, required_attrs)

    def check_supported_shape(
        self: ConvQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> None:
        """Ensure that Conv weights are available and have the correct dimensionality.

        Args:
            node (onnx.NodeProto): The node being validated.
            initializer_map (dict[str, onnx.TensorProto]):
                Mapping of initializer tensor names to TensorProtos.

        Raises:
            InvalidParamError: If weights are missing or have an unsupported shape.
        """
        supported_size = [4]
        weight_name = node.input[1]
        initializer = initializer_map.get(weight_name)

        if initializer is None:
            raise InvalidParamError(
                node.name,
                node.op_type,
                f"Weight tensor '{weight_name}' not found in initializers",
            )

        weight_dims = list(initializer.dims)

        if len(weight_dims) not in supported_size:
            msg = f"Unsupported Conv weight dimensionality {len(weight_dims)}. "
            msg += f"Expected 4D weights for Conv2D, got shape {weight_dims}"
            raise InvalidParamError(node.name, node.op_type, msg)
