from __future__ import annotations

import numpy as np
import onnx
from onnx import numpy_helper

from python.core.model_processing.onnx_custom_ops.onnx_helpers import extract_attributes
from python.core.model_processing.onnx_quantizer.exceptions import InvalidParamError
from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    ScaleConfig,
)


class ConvQuantizer(BaseOpQuantizer):
    """
    Quantizer for ONNX Conv layers.

    - Replaces standard Conv with Int64Conv from the `ai.onnx.contrib` domain
        and makes relevant additional changes to the graph.
    - Validates that all required Conv parameters are present.
    """

    def __init__(
        self: BaseOpQuantizer,
        new_initializers: dict[str, onnx.TensorProto],
    ) -> None:
        self.new_initializers = new_initializers

    def quantize(
        self: BaseOpQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        """
        Quantize a Conv node by:
        1. Quantizing its weights and bias.
        2. Adding a scale constant.
        3. Replacing it with an Int64Conv node.

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
            list[onnx.NodeProto]: A list of ONNX nodes
                (quantized and any auxiliary nodes).
        """
        _ = graph

        nodes = []
        output_name = f"{node.name}_int"

        nodes, node.input[:] = self.add_nodes_w_and_b(
            node=node,
            scale_exponent=scale_config.exponent,
            scale_base=scale_config.base,
            initializer_map=initializer_map,
        )
        attrs = extract_attributes(node)
        attrs.setdefault("group", 1)
        attrs.setdefault("auto_pad", "NOTSET")

        attrs["rescale"] = int(scale_config.rescale)

        scale_value = self.get_scaling(
            scale_config.base,
            scale_config.exponent,
        )

        # Create scale constant
        scale_const_name = f"{output_name}_scaler"
        scale_tensor = numpy_helper.from_array(
            np.array([scale_value], dtype=np.int64),
            name=scale_const_name,
        )
        self.new_initializers.append(scale_tensor)
        node.input.append(scale_const_name)
        int64_conv_node = onnx.helper.make_node(
            "Int64Conv",
            inputs=node.input,
            outputs=node.output,  # preserve original output name
            name=node.name,
            domain="ai.onnx.contrib",
            **attrs,
        )

        nodes.append(int64_conv_node)
        return nodes

    def check_supported(
        self: BaseOpQuantizer,
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

    def check_all_params_exist(self: BaseOpQuantizer, node: onnx.NodeProto) -> None:
        """Verify that all required Conv attributes are present.

        Args:
            node (onnx.NodeProto): The Conv node being validated.

        Raises:
            InvalidParamError: If any required parameter is missing.
        """
        required_attrs = ["strides", "kernel_shape", "dilations", "pads"]
        self.validate_required_attrs(node, required_attrs)

    def check_supported_shape(
        self: BaseOpQuantizer,
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
