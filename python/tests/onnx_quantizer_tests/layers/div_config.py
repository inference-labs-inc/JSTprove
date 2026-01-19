from collections.abc import Generator

import numpy as np

from python.core.model_processing.onnx_quantizer.exceptions import InvalidParamError
from python.tests.onnx_quantizer_tests import TEST_RNG_SEED
from python.tests.onnx_quantizer_tests.layers.base import (
    BaseLayerConfigProvider,
    LayerTestConfig,
    LayerTestSpec,
    e2e_test,
    edge_case_test,
    error_test,
    valid_test,
)


def random_power_of_two(
    rng: Generator,
    shape: tuple[int] | list[int] | np.ndarray,
    max_exp: int = 4,
) -> np.ndarray:
    """
    Generates random powers of two: 2^0 .. 2^max_exp
    """
    exponents = rng.integers(0, max_exp + 1, size=shape)
    return (2**exponents).astype(np.float32)


class DivConfigProvider(BaseLayerConfigProvider):
    """
    Test configuration provider for Div layer.

    Note: In quantized ML, Div is typically used for division by constants
    (e.g., rescaling). When both inputs are scaled, (A*s)/(B*s)=A/B loses
    the scale factor. Tests focus on the realistic case: constant divisors.
    """

    @property
    def layer_name(self) -> str:
        return "Div"

    def get_config(self) -> LayerTestConfig:
        return LayerTestConfig(
            op_type="Div",
            valid_inputs=["A", "B"],
            valid_attributes={},
            required_initializers={"B": np.array([2.0], dtype=np.float32)},
            input_shapes={
                "A": [1, 3, 4, 4],
            },
            output_shapes={
                "div_output": [1, 3, 4, 4],
            },
        )

    def get_test_specs(self) -> list[LayerTestSpec]:
        rng = np.random.default_rng(TEST_RNG_SEED)
        return [
            valid_test("scalar_div")
            .description("Div tensor by scalar constant")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", np.array([2.0], dtype=np.float32))
            .tags("scalar", "elementwise", "div")
            .build(),
            valid_test("initializer_div")
            .description("Div by tensor constant initializer (power of two)")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer(
                "B",
                random_power_of_two(rng, (1, 3, 4, 4)),
            )
            .tags("initializer", "elementwise", "div")
            .build(),
            valid_test("broadcast_div")
            .description("Div with broadcasting (power-of-two divisor)")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer(
                "B",
                random_power_of_two(rng, (1, 3, 1, 1)),
            )
            .tags("broadcast", "elementwise", "div")
            .build(),
            edge_case_test("div_by_one")
            .description("Division by 1 (identity operation)")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", np.array([1.0], dtype=np.float32))
            .tags("edge", "identity", "div")
            .build(),
            edge_case_test("large_divisor")
            .description("Division by large power-of-two constant")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", np.array([1024.0], dtype=np.float32))
            .tags("edge", "large_divisor", "div")
            .build(),
            e2e_test("e2e_scalar_div")
            .description("E2E div tensor by scalar constant")
            .override_input_shapes(A=[1])
            .override_output_shapes(div_output=[1])
            .override_initializer("B", np.array([2.0], dtype=np.float32))
            .tags("scalar", "elementwise", "div", "e2e")
            .build(),
            e2e_test("e2e_scalar_div_matrix")
            .description("E2E div tensor by scalar constant matrix")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", np.array([2.0], dtype=np.float32))
            .tags("scalar", "elementwise", "div", "e2e")
            .build(),
            e2e_test("e2e_initializer_div")
            .description("E2E div by tensor constant initializer")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer(
                "B",
                random_power_of_two(rng, (1, 3, 4, 4)),
            )
            .tags("initializer", "elementwise", "div", "e2e")
            .build(),
            e2e_test("e2e_broadcast_div")
            .description("E2E div with broadcasting (constant divisor)")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer(
                "B",
                random_power_of_two(rng, (1, 3, 1, 1)),
            )
            .tags("broadcast", "elementwise", "div", "e2e")
            .build(),
            # ---- Error Tests ----
            error_test("non_integer_scalar_divisor")
            .description("Divisor must be integer-valued (scalar float)")
            .override_initializer("B", np.array([2.5], dtype=np.float32))
            .expects_error(
                InvalidParamError,
                "The divisors must be integers",
            )
            .tags("error", "non_integer", "scalar", "div")
            .build(),
            error_test("non_integer_tensor_divisor")
            .description("Divisor tensor contains non-integer values")
            .override_initializer(
                "B",
                np.array([[1.0, 2.0], [3.5, 4.0]], dtype=np.float32),
            )
            .expects_error(
                InvalidParamError,
                "The divisors must be integers",
            )
            .tags("error", "non_integer", "tensor", "div")
            .build(),
            error_test("zero_divisor_scalar")
            .description("Division by zero is not allowed")
            .override_initializer("B", np.array([0.0], dtype=np.float32))
            .expects_error(
                InvalidParamError,
                "The divisors must be positive",
            )
            .tags("error", "zero", "scalar", "div")
            .build(),
            error_test("zero_in_tensor_divisor")
            .description("Tensor divisor contains zero")
            .override_initializer(
                "B",
                np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32),
            )
            .expects_error(
                InvalidParamError,
                "The divisors must be positive",
            )
            .tags("error", "zero", "tensor", "div")
            .build(),
            error_test("negative_scalar_divisor")
            .description("Divisor must be positive")
            .override_initializer("B", np.array([-2.0], dtype=np.float32))
            .expects_error(
                InvalidParamError,
                "The divisors must be positive",
            )
            .tags("error", "negative", "scalar", "div")
            .build(),
            error_test("negative_tensor_divisor")
            .description("Tensor divisor contains negative values")
            .override_initializer(
                "B",
                np.array([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32),
            )
            .expects_error(
                InvalidParamError,
                "The divisors must be positive",
            )
            .tags("error", "negative", "tensor", "div")
            .build(),
            error_test("divisor_not_initializer")
            .description("Divisor must be provided as an initializer")
            .override_inputs("A", "C")  # B is not backed by initializer
            .override_input_shapes(A=(2, 2), C=(2, 2))
            .expects_error(
                InvalidParamError,
                "The divisor must be a circuit constant",
            )
            .tags("error", "initializer", "div")
            .build(),
            error_test("non_power_of_two_scalar")
            .description("Divisor must be a power of two (scalar)")
            .override_initializer("B", np.array([3.0], dtype=np.float32))
            .expects_error(
                InvalidParamError,
                "The divisors must be powers of 2",
            )
            .tags("error", "non_power_of_two", "scalar", "div")
            .build(),
            error_test("non_power_of_two_tensor")
            .description("Tensor divisor contains non-power-of-two values")
            .override_initializer(
                "B",
                np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            )
            .expects_error(
                InvalidParamError,
                "The divisors must be powers of 2",
            )
            .tags("error", "non_power_of_two", "tensor", "div")
            .build(),
        ]
