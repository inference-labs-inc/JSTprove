from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from python.core.circuits.errors import CircuitConfigurationError

if TYPE_CHECKING:
    import onnx

import numpy as np
from onnx import helper, numpy_helper

from python.core.model_processing.onnx_custom_ops.onnx_helpers import extract_attributes
from python.core.model_processing.onnx_quantizer.exceptions import InvalidParamError
from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    QuantizerBase,
    ScaleConfig,
)


class QuantizeBatchnorm(QuantizerBase):
    OP_TYPE = "Int64BatchNorm"
    USE_WB = True
    USE_SCALING = False
    SCALE_PLAN: ClassVar = {}


class BatchnormQuantizer(BaseOpQuantizer, QuantizeBatchnorm):
    """
    Quantizer for ONNX Batchnorm layers.

    - Uses standard ONNX Batchnorm layer in standard domain, and
      makes relevant additional changes to the graph.
    """

    def __init__(
        self: BatchnormQuantizer,
        new_initializers: list[onnx.TensorProto] | None = None,
    ) -> None:
        super().__init__()
        # Only replace if caller provided something
        if new_initializers is not None:
            self.new_initializers = new_initializers

    def _compute_mul_add(
        self: BatchnormQuantizer,
        initializer_map: dict[str, onnx.TensorProto],
        node: onnx.NodeProto,
        scale_base: int,
        scale_exponent: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the 'mul' and 'add' tensors for BatchNorm folding.
        """
        self._validate_inputs(node=node)
        # ONNX BatchNorm inputs: [X, scale, bias, mean, var]
        scale_factor = scale_base**scale_exponent
        scale = numpy_helper.to_array(initializer_map[node.input[1]]).astype(np.float32)
        bias = numpy_helper.to_array(initializer_map[node.input[2]]).astype(np.float32)
        mean = numpy_helper.to_array(initializer_map[node.input[3]]).astype(np.float32)
        var = numpy_helper.to_array(initializer_map[node.input[4]]).astype(np.float32)

        # Find epsilon attribute
        epsilon_attr = next((a for a in node.attribute if a.name == "epsilon"), None)
        epsilon = float(epsilon_attr.f) if epsilon_attr else 1e-5

        mul = scale / np.sqrt(var + epsilon)
        add = bias - mean * mul
        scaled_add = add * (scale_factor**2)
        scaled_mul = scale_factor * mul
        return scaled_mul, scaled_add

    def pre_analysis_transform(
        self: BatchnormQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        initializer_map: dict[str, onnx.TensorProto],
        scale_base: int,
        scale_exponent: int,
    ) -> None:
        # Compute linearized BN tensors
        mul, add = self._compute_mul_add(
            initializer_map,
            node,
            scale_base=scale_base,
            scale_exponent=scale_exponent,
        )

        # Name base
        node_name = node.name if node.name else node.input[0]
        mul_name = f"{node_name}_mul"
        add_name = f"{node_name}_add"

        # Create ONNX tensors
        mul_tensor = numpy_helper.from_array(mul.astype(np.int64), name=mul_name)
        add_tensor = numpy_helper.from_array(add.astype(np.int64), name=add_name)

        # Insert them into the graph
        graph.initializer.extend([mul_tensor, add_tensor])
        initializer_map[mul_name] = mul_tensor
        initializer_map[add_name] = add_tensor
        self.new_initializers.extend([mul_tensor, add_tensor])

        node.input[:] = [node.input[0], mul_name, add_name]

        del node.attribute[:]

    def quantize(
        self,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        _ = graph

        nodes: list[onnx.NodeProto] = []

        # 1. Compute unscaled float mul/add coefficients
        mul, add = self._compute_mul_add(
            initializer_map,
            node,
            scale_base=1,
            scale_exponent=1,
        )

        node_name = node.name if node.name else node.input[0]
        mul_name = f"{node_name}_mul"
        add_name = f"{node_name}_add"

        # 2. Store unscaled mul and add initializers (as floats)
        scale_value = self.get_scaling(scale_config.base, scale_config.exponent)
        scale_name = f"{node.name}_int_scaler"
        scale_tensor = numpy_helper.from_array(
            np.array([scale_value], dtype=np.int64),
            name=scale_name,
        )
        self.new_initializers.append(scale_tensor)

        mul_tensor = numpy_helper.from_array(mul.astype(np.float32), name=mul_name)
        add_tensor = numpy_helper.from_array(add.astype(np.float32), name=add_name)

        initializer_map[mul_name] = mul_tensor
        initializer_map[add_name] = add_tensor

        # 3. Insert scale and cast for mul_tensor
        scaled_mul_name, mul_scale_node, mul_cast_node = self.insert_scale_node(
            tensor=mul_tensor,
            scale_base=scale_config.base,
            scale_exponent=scale_config.exponent,
        )

        # 4. Insert scale and cast for add_tensor
        scaled_add_name, add_scale_node, add_cast_node = self.insert_scale_node(
            tensor=add_tensor,
            scale_base=scale_config.base,
            scale_exponent=scale_config.exponent * 2,
        )
        # Note, order is important here
        nodes.extend(
            [
                mul_scale_node,
                mul_cast_node,
                add_scale_node,
                add_cast_node,
            ],
        )

        # 5. Build final Int64BatchNorm node
        attrs = extract_attributes(node)
        for k, v in getattr(self, "DEFAULT_ATTRS", {}).items():
            attrs.setdefault(k, v)
        attrs["rescale"] = 1

        quant_node = helper.make_node(
            self.OP_TYPE,  # Should be "Int64BatchNorm"
            inputs=[
                node.input[0],  # original X
                scaled_mul_name,  # scaled mul
                scaled_add_name,  # scaled add
                scale_name,  # scaling factor
            ],
            outputs=node.output,
            name=node.name,
            domain=self.DOMAIN,
            **attrs,
        )

        nodes.append(quant_node)
        return nodes

    def check_supported(
        self: BatchnormQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
    ) -> None:
        """
        For our current implementation, all batchnorm inputs
        (scale, variance, mean, etc.)
        must be initializers to the circuit and not inputs from earlier in the graph.
        """

        if initializer_map is None:
            msg = "initializer_map is required for BatchNorm support check"
            raise CircuitConfigurationError(node.name, node.op_type, msg)

        self._validate_inputs(node=node)

        # First, check to make sure that each of the batchnorm inputs are initializers
        initializer_inputs = node.input[1:]
        if not all(i in initializer_map for i in initializer_inputs):
            msg = "Unsupported BatchNorm with normalization inputs not in initializers"
            raise InvalidParamError(node.name, node.op_type, msg)

    def _validate_inputs(self, node: onnx.NodeProto) -> None:
        """Validate BatchNorm has required inputs in initializer_map."""
        num_inputs = 5
        if len(node.input) < num_inputs:
            raise InvalidParamError(
                node.name,
                node.op_type,
                f"BatchNorm requires 5 inputs, got {len(node.input)}",
            )
