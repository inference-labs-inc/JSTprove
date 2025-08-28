from __future__ import annotations

import onnx
from onnx import helper

from python.core.model_processing.onnx_custom_ops.onnx_helpers import (
    extract_attributes,
    get_attribute_ints,
)
from python.core.model_processing.onnx_quantizer.exceptions import InvalidParamError
from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    ScaleConfig,
)


class MaxpoolQuantizer(BaseOpQuantizer):
    """
    Quantizer for ONNX MaxPool layers.

    - Replaces standard MaxPool with Int64MaxPool from the `ai.onnx.contrib`
        domain and makes relevant additional changes to the graph.
    - Validates that all required MaxPool parameters are present.
    """

    def __init__(
        self: BaseOpQuantizer,
        new_initializer: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        super().__init__()
        self.accepted_kernel_shapes = [2]
        _ = new_initializer

    def quantize(
        self: BaseOpQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        """
        Quantize a node by converting the node to Int64 version

        Args:
            node (onnx.NodeProto): The node to quantize.
            rescale (bool): Whether rescaling is enabled
                (Doesnt have an affect on this op type)
            graph (onnx.GraphProto): The ONNX graph.
            scale_exponent (int): Scale exponent.
            scale_base (int): The base of scaling.
            initializer_map (dict[str, onnx.TensorProto]):
                Map of initializer names to tensor data.

        Returns:
            List[onnx.NodeProto]: A list of ONNX nodes
                (quantized MaxPool and any auxiliary nodes).
        """
        _ = initializer_map, graph

        attrs = extract_attributes(node)
        attrs["rescale"] = int(scale_config.rescale)

        attr_str = {
            k: ",".join(map(str, v)) if isinstance(v, list) else str(v)
            for k, v in attrs.items()
        }
        return [
            helper.make_node(
                "Int64MaxPool",
                inputs=node.input,
                outputs=node.output,
                name=node.name,
                domain="ai.onnx.contrib",
                **attr_str,
            ),
        ]

    def check_supported(
        self: BaseOpQuantizer,
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
        self.check_all_params_exist(node)
        self.check_params_size(node)

    def check_all_params_exist(self: BaseOpQuantizer, node: onnx.NodeProto) -> None:
        """Checks all parameters that are needed, do exist

        Args:
            node (onnx.NodeProto): ONNX node to check

        Raises:
            InvalidParamError: If shape requirement is not met.
        """
        required_attrs = ["strides", "kernel_shape", "pads", "dilations"]
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

    def check_params_size(self: BaseOpQuantizer, node: onnx.NodeProto) -> None:
        """Checks dimension of the layer and ensures that it is supported

        Args:
            node (onnx.NodeProto): ONNX node to check

        Raises:
            InvalidParamError: If shape requirement is not met.
        """

        kernel_shape = get_attribute_ints(node, "kernel_shape", default="N/A")
        if len(kernel_shape) not in self.accepted_kernel_shapes:
            raise InvalidParamError(
                node.name,
                node.op_type,
                f"Currently only maxpool2d is supported. Found {len(kernel_shape)}D",
            )
