from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import ClassVar

    import onnx

from python.core.model_processing.onnx_custom_ops.onnx_helpers import (
    extract_attributes,
    get_attribute_ints,
)
from python.core.model_processing.onnx_quantizer.exceptions import InvalidParamError
from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    QuantizerBase,
    ScaleConfig,
)


class QuantizeMaxpool(QuantizerBase):
    OP_TYPE = "Int64MaxPool"
    USE_WB = False
    USE_SCALING = False

    DEFAULT_ATTRS: ClassVar = {
        "dilations": "1",
        "pads": "0",
        "strides": "1",
    }


class MaxpoolQuantizer(BaseOpQuantizer, QuantizeMaxpool):
    """
    Quantizer for ONNX MaxPool layers.

    - Replaces standard MaxPool with Int64MaxPool from the `ai.onnx.contrib`
        domain and makes relevant additional changes to the graph.
    - Validates that all required MaxPool parameters are present.
    """

    def __init__(
        self: MaxpoolQuantizer,
        new_initializer: list[onnx.TensorProto] | None = None,
    ) -> None:
        super().__init__()
        self.accepted_kernel_shapes = [2]
        _ = new_initializer

    def quantize(
        self: MaxpoolQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        return QuantizeMaxpool.quantize(
            self,
            node,
            graph,
            scale_config,
            initializer_map,
        )

    def check_supported(
        self: MaxpoolQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> None:
        """
        Perform high-level validation to ensure that this node
        can be quantized safely.

        Args:
            node (onnx.NodeProto): ONNX node to be checked
            initializer_map (dict[str, onnx.TensorProto]):
                Initializer map (name of weight or bias and tensor)

        Raises:
            InvalidParamError: If any requirement is not met.
        """
        _ = initializer_map
        attributes = extract_attributes(node)
        ceil_mode = attributes.get("ceil_mode", None)
        auto_pad = attributes.get("auto_pad", None)
        storage_order = attributes.get("storage_order", None)

        if ceil_mode != 0 and ceil_mode is not None:
            raise InvalidParamError(
                node.name,
                node.op_type,
                "ceil_mode must be 0",
                "ceil_mode",
                "0",
            )
        if auto_pad != "NOTSET" and auto_pad is not None:
            raise InvalidParamError(
                node.name,
                node.op_type,
                "auto_pad must be NOTSET",
                "auto_pad",
                "0",
            )
        if storage_order != 0 and storage_order is not None:
            raise InvalidParamError(
                node.name,
                node.op_type,
                "storage_order must be 0",
                "storage_order",
                "0",
            )
        self.check_all_params_exist(node)
        self.check_params_size(node)
        self.check_pool_pads(node)

    def check_all_params_exist(self: MaxpoolQuantizer, node: onnx.NodeProto) -> None:
        """Checks all parameters that are needed, do exist

        Args:
            node (onnx.NodeProto): ONNX node to check

        Raises:
            InvalidParamError: If shape requirement is not met.
        """
        # May need: ["strides", "kernel_shape", "pads", "dilations"]
        required_attrs = ["strides", "kernel_shape"]
        self.validate_required_attrs(node, required_attrs)

        # Check dimension of kernel
        kernel_shape = get_attribute_ints(node, "kernel_shape", default=[])
        if len(kernel_shape) not in self.accepted_kernel_shapes:
            raise InvalidParamError(
                node.name,
                node.op_type,
                "Currently only MaxPool2D is supported."
                f"Found {len(kernel_shape)}D kernel",
                "kernel_shape",
                "2D",
            )

    def check_params_size(self: MaxpoolQuantizer, node: onnx.NodeProto) -> None:
        """Checks dimension of the layer and ensures that it is supported

        Args:
            node (onnx.NodeProto): ONNX node to check

        Raises:
            InvalidParamError: If shape requirement is not met.
        """

        kernel_shape = get_attribute_ints(node, "kernel_shape", default=[])
        if len(kernel_shape) not in self.accepted_kernel_shapes:
            raise InvalidParamError(
                node.name,
                node.op_type,
                f"Currently only maxpool2d is supported. Found {len(kernel_shape)}D",
            )

    def check_pool_pads(self: MaxpoolQuantizer, node: onnx.NodeProto) -> None:
        kernel_shape = get_attribute_ints(node, "kernel_shape", default=[])
        pads = get_attribute_ints(node, "pads", default=None)
        if pads is None:
            return
        num_dims = len(kernel_shape)
        if len(pads) != num_dims * 2:
            raise InvalidParamError(
                node.name,
                node.op_type,
                f"Expected {num_dims * 2} pads, got {len(pads)}",
            )

        for dim in range(num_dims):
            pad_before = pads[dim]
            pad_after = pads[dim + num_dims]
            kernel = kernel_shape[dim]
            if pad_before >= kernel:
                raise InvalidParamError(
                    node.name,
                    node.op_type,
                    f"pads[{dim}]={pad_before} >= kernel[{dim}]={kernel}",
                )
            if pad_after >= kernel:
                raise InvalidParamError(
                    node.name,
                    node.op_type,
                    f"pads[{dim + num_dims}]={pad_after} >= kernel[{dim}]={kernel}",
                )
