from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from onnx import helper, numpy_helper

if TYPE_CHECKING:
    import onnx

from python.core.model_processing.onnx_quantizer.exceptions import InvalidParamError
from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    ScaleConfig,
)


def is_integer_valued(arr: np.ndarray) -> bool:
    return np.all(np.equal(arr, np.floor(arr)))


class DivQuantizer(BaseOpQuantizer):
    """
    Quantizer for ONNX Div layers.

    Div requires casting initializer inputs to int64 but does NOT scale them,
    since scaling would change the division semantics.
    """

    def __init__(
        self: DivQuantizer,
        new_initializers: list[onnx.TensorProto] | None = None,
    ) -> None:
        super().__init__()
        if new_initializers is not None:
            self.new_initializers = new_initializers

    def quantize(
        self: DivQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        _ = graph
        attrs = {}
        attrs["mode"] = "default_error"
        if self.check_divisor_circuit_constant(node, initializer_map):
            divisors = numpy_helper.to_array(initializer_map[node.input[1]]).astype(
                np.float64,
            )
            if (
                self.check_divisor_positive(divisors)
                and self.check_divisor_integer(divisors)
                and self.check_divisor_sufficiently_small(
                    divisors,
                    scale_config.base,
                    scale_config.exponent,
                )
                and self.check_divisor_power_of_two(divisors)
            ):
                attrs["mode"] = "constant_pos_int_pow_two"

        if attrs["mode"] == "default_error":
            msg = "The div node is unsupported"
            raise InvalidParamError(node.name, node.op_type, msg)
        nodes = []
        new_inputs = list(node.input)

        for idx, input_name in enumerate(node.input):
            if input_name in initializer_map:
                tensor = initializer_map[input_name]
                arr = numpy_helper.to_array(tensor).astype(np.int64)
                cast_name = f"{input_name}_int64"
                cast_tensor = numpy_helper.from_array(arr, name=cast_name)
                self.new_initializers.append(cast_tensor)
                new_inputs[idx] = cast_name

        scale_value = self.get_scaling(scale_config.base, scale_config.exponent)
        scale_name = f"{node.name}_int_scaler"
        scale_tensor = numpy_helper.from_array(
            np.array([scale_value], dtype=np.int64),
            name=scale_name,
        )
        self.new_initializers.append(scale_tensor)
        new_inputs.append(scale_name)

        quantized_node = helper.make_node(
            "Int64Div",
            inputs=new_inputs,
            outputs=node.output,
            name=node.name,
            domain="ai.onnx.contrib",
            **attrs,
        )

        nodes.append(quantized_node)
        return nodes

    def check_supported(
        self: DivQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto] | None = None,
        scale_base: int | None = 2,
        scale_exponent: int | None = 18,
    ) -> None:
        # 1. Check that the divisor is a circuit constant
        # 2. Check that the divisor is an integer
        # 3. Check that the divisor is positive
        # 4. Check that the divisor is not zero
        # 5. Check that the divisor is sufficiently small
        # 6. Check that the divisor is a power of 2
        if initializer_map is None:
            msg = "The divisor must be an initializer"
            raise InvalidParamError(node.name, node.op_type, msg)
        num_inputs = 2

        if not self.check_divisor_num_inputs(node, num_inputs):
            msg = f"Div must have exactly {num_inputs} inputs"
            raise InvalidParamError(node.name, node.op_type, msg)

        # 1. Check that the divisor is a circuit constant
        if not self.check_divisor_circuit_constant(node, initializer_map):
            msg = "The divisor must be a circuit constant"
            raise InvalidParamError(node.name, node.op_type, msg)

        # 2. Check that the divisor is an integer
        divisors = numpy_helper.to_array(initializer_map[node.input[1]]).astype(
            np.float64,
        )
        if not self.check_divisor_integer(divisors):
            msg = "The divisors must be integers"
            raise InvalidParamError(node.name, node.op_type, msg)

        # 3. Check that the divisor is positive
        if not self.check_divisor_positive(divisors):
            msg = "The divisors must be positive"
            raise InvalidParamError(node.name, node.op_type, msg)

        if not self.check_divisor_sufficiently_small(
            divisors,
            scale_base,
            scale_exponent,
        ):
            msg = "The divisors must be sufficiently small"
            raise InvalidParamError(node.name, node.op_type, msg)

        if not self.check_divisor_power_of_two(divisors):
            msg = "The divisors must be powers of 2"
            raise InvalidParamError(node.name, node.op_type, msg)

    def check_divisor_integer(self: DivQuantizer, divisors: np.ndarray) -> bool:
        return is_integer_valued(divisors)

    def check_divisor_positive(self: DivQuantizer, divisors: np.ndarray) -> bool:
        return np.all(divisors > 0)

    def check_divisor_sufficiently_small(
        self: DivQuantizer,
        divisors: np.ndarray,
        scale_base: int,
        scale_exponent: int,
    ) -> bool:
        return np.all(divisors <= scale_base ** (scale_exponent // 2))

    def check_divisor_not_zero(self: DivQuantizer, divisors: np.ndarray) -> bool:
        return np.all(divisors != 0)

    def check_divisor_circuit_constant(
        self: DivQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> bool:
        return node.input[1] in initializer_map

    def check_divisor_num_inputs(
        self: DivQuantizer,
        node: onnx.NodeProto,
        num_inputs: int,
    ) -> bool:
        return len(node.input) == num_inputs

    def check_divisor_power_of_two(self: DivQuantizer, divisors: np.ndarray) -> bool:
        return np.all(np.log2(divisors) % 1 == 0)
