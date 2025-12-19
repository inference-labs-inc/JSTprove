from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
import onnx
from onnx import helper, numpy_helper

from python.core.model_processing.onnx_custom_ops.onnx_helpers import (
    extract_attributes,
    replace_input_references,
)
from python.core.model_processing.onnx_quantizer.exceptions import (
    HandlerImplementationError,
    InitializerNotFoundError,
    InvalidConfigError,
    InvalidParamError,
)


@dataclass
class ScaleConfig:
    exponent: int
    base: int
    rescale: bool


class BaseOpQuantizer:
    """
    Abstract base class for ONNX operator quantizers.

    Subclasses must implement:
        - `quantize`: Apply quantization logic to an ONNX node.
        - `check_supported`: Checks if the layer and param specs are supported.

    Attributes:
        new_initializers (list[onnx.TensorProto]):
            A list of initializers created during quantization.
            These should be added to the graph after processing.
    """

    def __init__(self: BaseOpQuantizer) -> None:
        self.new_initializers: list[onnx.TensorProto] = []

    @staticmethod
    def get_scaling(scale_base: int, scale_exponent: int) -> int:
        """Validate and compute the scaling factor.

        Args:
            scale_base (int): Base for the scaling exponent.
            scale_exponent (int): Scaling exponent.

        Returns:
            int: The computed scaling factor (scale_base ** scale_exponent).

        Raises:
            InvalidConfigError: If parameters are invalid.
        """
        if scale_base <= 0:
            key = "scale_base"
            raise InvalidConfigError(key, scale_base, expected="> 0")
        if scale_exponent < 0:
            key = "scale_exponent"
            raise InvalidConfigError(key, scale_exponent, expected=">= 0")

        try:
            return scale_base**scale_exponent
        except (TypeError, OverflowError, ValueError, Exception) as e:
            key = "scaling"
            raise InvalidConfigError(
                key,
                f"{scale_base}^{scale_exponent}",
                str(e),
            ) from e

    @staticmethod
    def validate_node_has_output(node: onnx.NodeProto) -> None:
        """Ensure a node has at least one output.

        Args:
            node (onnx.NodeProto): The node to validate.
            op_type (str): Name of the operator type for error reporting.

        Raises:
            HandlerImplementationError: If the node has no outputs.
        """
        if not node.output or len(node.output) == 0:
            raise HandlerImplementationError(
                op_type=node.op_type,
                message=f"Node '{node.name or '<unnamed>'}' of type '{node.op_type}'"
                " has no outputs.",
            )

    @staticmethod
    def validate_required_attrs(
        node: onnx.NodeProto,
        required_attrs: list[str],
    ) -> None:
        """
        Ensure that a node contains all required attributes.

        Args:
            node (onnx.NodeProto): The ONNX node to validate.
            required_attrs (list[str]): list of attribute names that must exist.
            op_type (str): Name of the operator type for error reporting.

        Raises:
            InvalidParamError: If any required attribute is missing.
        """
        missing_attrs = []
        for attr_name in required_attrs:
            found = any(attr.name == attr_name for attr in node.attribute)
            if not found:
                missing_attrs.append(attr_name)

        if missing_attrs:
            missing_str = ", ".join(missing_attrs)
            raise InvalidParamError(
                node_name=node.name,
                op_type=node.op_type,
                message=f"Missing required attributes: {missing_str}",
            )

    def quantize(
        self: BaseOpQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        """
        Quantize the given node.

        Must be implemented by subclasses.

        Raises:
            HandlerImplementationError: If subclass does not implement quantize
        """
        _ = node, graph, scale_config, initializer_map
        raise HandlerImplementationError(
            op_type=self.__class__.__name__,
            message="quantize() not implemented in subclass.",
        )

    def check_supported(
        self: BaseOpQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> str | None:
        """
        Check if the node is supported by the quantizer.

        Must be overridden by subclasses to validate parameters.

        Raises:
            HandlerImplementationError: If called on BaseOpQuantizer directly.
        """
        _ = node, initializer_map
        raise HandlerImplementationError(
            op_type=self.__class__.__name__,
            message="check_supported() not implemented in subclass.",
        )

    def rescale_layer(
        self: BaseOpQuantizer,
        node: onnx.NodeProto,
        scale_base: int,
        scale_exponent: int,
        graph: onnx.GraphProto,
    ) -> list[onnx.NodeProto]:
        """
        Helper function for any quantizer.
        Used to add a rescaling step after the given node.

        This replaces the node's output with a scaled version using a Div op.
        This function incorporates the logic to insert and restructure the graph.

        Args:
            node (onnx.NodeProto): Node to rescale.
            scale_base (int): Base for the scaling exponent.
            scale_exponent (int): Scaling exponent.
            graph (onnx.GraphProto): The ONNX graph.

        Returns:
            list[onnx.NodeProto]: Original node and the inserted Div node.

        Raises:
            HandlerImplementationError if there are no outputs to be rescaled
        """
        self.validate_node_has_output(node)

        original_output = node.output[0]
        quantized_output = original_output + "_raw"
        node.output[0] = quantized_output

        # Create scale constant initializer
        scale_const_name = node.name + "_scale"

        scale_value = self.get_scaling(scale_base, scale_exponent)
        scale_tensor = numpy_helper.from_array(
            np.array([scale_value], dtype=np.int64),
            name=scale_const_name,
        )
        self.new_initializers.append(scale_tensor)

        # Create Div node for rescaling output
        div_node = helper.make_node(
            "Div",
            inputs=[quantized_output, scale_const_name],
            outputs=[original_output],  # restore original output name
            name=node.name + "_rescale",
        )

        # Rewire consumers to point to the new output
        replace_input_references(
            graph=graph,
            old_output=original_output,
            new_output=div_node.output[0],
        )

        return [node, div_node]

    def add_nodes_w_and_b(
        self: BaseOpQuantizer,
        node: onnx.NodeProto,
        scale_exponent: int,
        scale_base: int,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> tuple[list[onnx.NodeProto], list[str]]:
        """Insert scaling and casting nodes for weight and bias,
            to convert from float to scaled int64 values.

        Args:
            node (onnx.NodeProto): Node to find used weights and biases.
            scale_exponent (int): Scaling exponent.
            scale_base (int): Base for the scaling exponent.
            initializer_map (dict[str, onnx.TensorProto]): The initializer map.
            graph (onnx.GraphProto): ONNX Graph

        Returns:
            tuple[list[onnx.NodeProto], list[str]]:
                list of new nodes added, updated input names for nodes.

        Raises:
            InitializerNotFoundError: If weights or biases are missing from the graph.
            HandlerImplementationError:
                If there are no weights or biases to add to the graph.
        """
        weights_input_length = 2
        if len(node.input) < weights_input_length:
            raise HandlerImplementationError(
                op_type=node.op_type,
                message=f"Node '{node.name}'"
                " has fewer than 2 inputs (weights missing).",
            )
        # Quantize weight
        weight_name = node.input[1]
        if not weight_name or weight_name not in initializer_map:
            raise InitializerNotFoundError(node.name, weight_name or "<missing>")

        weight_tensor = initializer_map[weight_name]
        if not weight_tensor.name:
            raise HandlerImplementationError(
                op_type=node.op_type,
                message=f"Weight tensor for node '{node.name}' is missing a name.",
            )

        quant_weight_name, mul_node, cast_node = self.insert_scale_node(
            tensor=weight_tensor,
            scale_base=scale_base,
            scale_exponent=scale_exponent,
        )

        # Quantize bias if present
        new_inputs = [node.input[0], quant_weight_name]
        nodes = [mul_node, cast_node]

        bias_inputs_length = 3

        if len(node.input) >= bias_inputs_length:
            bias_name = node.input[2]
            if bias_name not in initializer_map:
                raise InitializerNotFoundError(node.name, bias_name)

            bias_tensor = initializer_map[bias_name]
            quant_bias_name, mul_node_2, cast_node_2 = self.insert_scale_node(
                tensor=bias_tensor,
                scale_base=scale_base,
                scale_exponent=(scale_exponent * 2),
            )
            new_inputs.append(quant_bias_name)
            nodes.append(mul_node_2)
            nodes.append(cast_node_2)

        # === Mutate the original node ===
        return nodes, new_inputs

    def add_scaled_initializer_inputs(
        self: BaseOpQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto],
        scale_base: int,
        scale_exponent: int,
        scale_plan: dict[int, int],
    ) -> tuple[list[onnx.NodeProto], list[str]]:
        """
        Scale and cast specific initializer inputs
        of a node according to a scaling plan.

        Handles optional inputs gracefully (e.g. missing bias).
        """
        new_nodes: list[onnx.NodeProto] = []
        new_inputs = list(node.input)

        for input_idx, scale_mult in scale_plan.items():
            # Skip if node doesn't have that many inputs (e.g. missing bias)
            if input_idx >= len(node.input):
                # Just ignore — optional input not provided
                continue

            input_name = node.input[input_idx]
            if not input_name:
                # Empty input name → optional input not present
                continue

            if input_name not in initializer_map:
                # Optional inputs may be missing from initializers (e.g., dynamic bias)
                continue

            tensor = initializer_map[input_name]
            if not tensor.name:
                raise HandlerImplementationError(
                    op_type=node.op_type,
                    message=f"Initializer tensor for '{input_name}' on node "
                    f"'{node.name}' is missing a name.",
                )

            # Scale according to plan (e.g., scale_exponent * 2 for bias)
            quant_name, mul_node, cast_node = self.insert_scale_node(
                tensor=tensor,
                scale_base=scale_base,
                scale_exponent=(scale_exponent * scale_mult),
            )

            # Update node input to point to scaled version
            new_inputs[input_idx] = quant_name

            # Record new scaling/cast nodes
            new_nodes.extend([mul_node, cast_node])

        return new_nodes, new_inputs

    def insert_scale_node(
        self: BaseOpQuantizer,
        tensor: onnx.TensorProto,
        scale_base: int,
        scale_exponent: int,
    ) -> tuple[str, onnx.NodeProto, onnx.NodeProto]:
        """Insert Mul and Cast nodes to apply scaling to a tensor.

        Args:
            tensor (onnx.TensorProto): Tensor to scale.
            scale_base (int): Base for scaling exponent.
            scale_exponent (int): Scaling exponent.
            graph (onnx.GraphProto): ONNX graph.

        Returns:
            tuple[str, onnx.NodeProto, onnx.NodeProto]:
                New tensor name, Mul node, Cast node.

        Raises:
            HandlerImplementationError:
                If tensor does not exist, incorrectly formatted or not named
        """
        if not tensor or not isinstance(tensor, onnx.TensorProto):
            raise HandlerImplementationError(
                op_type="insert_scale_node",
                message="Expected a valid onnx.TensorProto, got None or wrong type.",
            )

        if not tensor.name:
            raise HandlerImplementationError(
                op_type="insert_scale_node",
                message="Tensor is missing a name.",
            )

        scale_value = self.get_scaling(scale_base, scale_exponent)

        # Create scale constant
        scale_const_name = tensor.name + "_scale"
        scale_tensor = numpy_helper.from_array(
            np.array([scale_value], dtype=np.float64),
            name=scale_const_name,
        )
        self.new_initializers.append(scale_tensor)

        # Add Mul node
        scaled_output_name = f"{tensor.name}_scaled"
        mul_node = helper.make_node(
            "Mul",
            inputs=[tensor.name, scale_const_name],
            outputs=[scaled_output_name],
            name=f"{tensor.name}_mul",
        )

        # Add cast node
        output_name = f"{scaled_output_name}_cast"
        rounded_output_name = scaled_output_name
        cast_to_int64 = helper.make_node(
            "Cast",
            inputs=[scaled_output_name],
            outputs=[output_name],
            to=onnx.TensorProto.INT64,
            name=rounded_output_name,
        )
        return output_name, mul_node, cast_to_int64


class QuantizerBase:
    """
    Shared mixin implementing the generic INT64 quantization pipeline.

    IMPORTANT:
        QuantizerBase is *not* a standalone quantizer. It must always be
        combined with BaseOpQuantizer via multiple inheritance:

            class FooQuantizer(BaseOpQuantizer, QuantizeFoo):
                ...

        BaseOpQuantizer supplies required methods and attributes that
        QuantizerBase relies on:
            - add_scaled_initializer_inputs
            - insert_scale_node
            - get_scaling
            - new_initializers  (initializer buffer shared with converter)

        If a subclass inherits QuantizerBase without BaseOpQuantizer,
        QuantizerBase.quantize() will raise attribute errors at runtime.

    This mixin centralizes:
        - attribute extraction/merging
        - optional initializer scaling (USE_WB + SCALE_PLAN)
        - optional rescaling of outputs (USE_SCALING)
        - creation of the final quantized NodeProto

    The Quantize<Op> mixins should define:
        - OP_TYPE
        - DOMAIN
        - USE_WB (bool)
        - USE_SCALING (bool)
        - SCALE_PLAN (dict[int,int]) if initializer scaling is enabled
    """

    OP_TYPE = None
    DOMAIN = "ai.onnx.contrib"
    DEFAULT_ATTRS: ClassVar = {}
    USE_WB = False
    USE_SCALING = False

    def quantize(
        self,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        """Generic quantization template for most Int64 ops."""
        _ = graph
        nodes = []

        # (1) Quantize weights/bias if applicable
        if self.USE_WB:
            # Each subclass defines its scaling plan for which inputs get scaled and how
            scale_plan = getattr(self, "SCALE_PLAN", {1: 1, 2: 2})  # default for W & B
            nodes, new_inputs = self.add_scaled_initializer_inputs(
                node=node,
                initializer_map=initializer_map,
                scale_base=scale_config.base,
                scale_exponent=scale_config.exponent,
                scale_plan=scale_plan,
            )
            node.input[:] = new_inputs

        # (2) Collect & merge attributes
        self.apply_default_attrs(node)
        attrs = extract_attributes(node)
        if self.USE_SCALING:
            attrs["rescale"] = int(scale_config.rescale)

        attrs = self._serialize_quantized_attrs(attrs)

        # (3) Add scaling constant if needed
        if self.USE_SCALING:
            scale_value = self.get_scaling(scale_config.base, scale_config.exponent)
            scale_name = f"{node.name}_int_scaler"
            scale_tensor = numpy_helper.from_array(
                np.array([scale_value], dtype=np.int64),
                name=scale_name,
            )
            self.new_initializers.append(scale_tensor)
            node.input.append(scale_name)

        # (4) Create quantized node
        quantized_node = onnx.helper.make_node(
            self.OP_TYPE,
            inputs=node.input,
            outputs=node.output,
            name=node.name,
            domain=self.DOMAIN,
            **attrs,
        )

        nodes.append(quantized_node)
        return nodes

    def pre_analysis_transform(
        self: QuantizerBase,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        initializer_map: dict[str, onnx.TensorProto],
        scale_base: int,
        scale_exponent: int,
    ) -> None:
        """
        pre_analysis_transform aims to transform the given layer along the
        same lines as it would be transformed for the quantized model, but
        for the weights and biases file instead, to be sent to the backend

        Default pre-analysis behavior:

        - If the subclass uses weights/bias (`USE_WB=True`), apply the SAME
        scaling rules as quantization, but directly mutate the initializers.

        - Subclasses can override this to implement more complex rewrites
        (e.g., BatchNorm → Mul/Add).

        Args:
            node (onnx.NodeProto): Node to transform.
            graph (onnx.GraphProto): Rest of the Onnx graph for initializers.
            initializer_map (dict[str, onnx.TensorProto]): The initializer map.

            scale_base (int): Scaling base.
            scale_exponent (int): Scaling exponent.

        NOTE
         - The resulting model will not make accurate prediction and should be
         used solely for analysis and keeping track of w_and_b
        """
        self.apply_default_attrs(node)
        # If subclass does not want auto-scaling, do nothing
        if not getattr(self, "USE_WB", False):
            return

        # Each quantizer defines which inputs to scale (Weight:1x, Bias:2x etc.)
        scale_plan = getattr(self, "SCALE_PLAN", {})

        # Perform the same scaling as quantization, but directly modify initializers
        for input_idx, scale_mult in scale_plan.items():
            if input_idx >= len(node.input):
                continue

            name = node.input[input_idx]
            if name not in initializer_map:
                continue  # optional input missing

            tensor = initializer_map[name]
            arr = numpy_helper.to_array(tensor).astype(np.float64)

            scale = scale_base ** (scale_exponent * scale_mult)
            new_arr = arr * scale

            # Replace initializer directly
            new_tensor = numpy_helper.from_array(new_arr, name=tensor.name)

            # Modify graph initializer in place
            for j in range(len(graph.initializer)):
                if graph.initializer[j].name == tensor.name:
                    del graph.initializer[j]
                    break
            graph.initializer.append(new_tensor)

            initializer_map[tensor.name] = new_tensor

    def apply_default_attrs(self, node: onnx.NodeProto) -> None:
        """
        Ensure DEFAULT_ATTRS are explicitly present on the node.
        Does not overwrite existing attributes.
        """
        if not getattr(self, "DEFAULT_ATTRS", None):
            return

        existing = {attr.name for attr in node.attribute}

        for name, value in self.DEFAULT_ATTRS.items():
            if name in existing:
                continue

            try:
                attr = onnx.helper.make_attribute(name, value)
            except Exception as e:
                raise HandlerImplementationError(
                    op_type=node.op_type,
                    message=f"Failed to create default attribute '{name}': {e}",
                ) from e

            node.attribute.append(attr)

    def _serialize_quantized_attrs(self, attrs: dict) -> dict:
        """
        Convert logical attribute values into the serialized form expected
        by quantized custom ops.

        Lists are converted to comma-separated strings.
        """
        serialized = {}

        for name, value in attrs.items():
            if isinstance(value, list):
                serialized[name] = ", ".join(str(v) for v in value)
            else:
                serialized[name] = value

        return serialized


class PassthroughQuantizer(BaseOpQuantizer):
    """
    Quantizer that leaves the node unchanged.
    Useful for operators that do not require quantization, such as shaping operations.
    """

    def __init__(
        self: BaseOpQuantizer,
        new_initializer: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        _ = new_initializer
        super().__init__()

    def quantize(
        self: BaseOpQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        _ = graph, scale_config, initializer_map
        if not isinstance(node, onnx.NodeProto):
            raise HandlerImplementationError(
                op_type="PassthroughQuantizer",
                message="quantize() expected a NodeProto",
            )
        return [node]

    def check_supported(
        self: BaseOpQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        _ = node, initializer_map
