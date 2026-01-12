import numpy as np

from python.tests.onnx_quantizer_tests import TEST_RNG_SEED
from python.tests.onnx_quantizer_tests.layers.base import (
    BaseLayerConfigProvider,
    LayerTestConfig,
    LayerTestSpec,
    e2e_test,
    edge_case_test,
    valid_test,
)


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
            .description("Div by tensor constant initializer")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", rng.integers(1, 10, (1, 3, 4, 4)).astype(np.float32))
            .tags("initializer", "elementwise", "div")
            .build(),
            valid_test("broadcast_div")
            .description("Div with broadcasting (constant divisor)")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", rng.integers(1, 10, (1, 3, 1, 1)).astype(np.float32))
            .tags("broadcast", "elementwise", "div")
            .build(),
            edge_case_test("div_by_one")
            .description("Division by 1 (identity operation)")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", np.array([1.0], dtype=np.float32))
            .tags("edge", "identity", "div")
            .build(),
            edge_case_test("large_divisor")
            .description("Division by large constant")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", np.array([1000.0], dtype=np.float32))
            .tags("edge", "large_divisor", "div")
            .build(),
            e2e_test("e2e_scalar_div")
            .description("E2E div tensor by scalar constant")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", np.array([2.0], dtype=np.float32))
            .tags("scalar", "elementwise", "div", "e2e")
            .build(),
            e2e_test("e2e_initializer_div")
            .description("E2E div by tensor constant initializer")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", rng.integers(1, 10, (1, 3, 4, 4)).astype(np.float32))
            .tags("initializer", "elementwise", "div", "e2e")
            .build(),
            e2e_test("e2e_broadcast_div")
            .description("E2E div with broadcasting (constant divisor)")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", rng.integers(1, 10, (1, 3, 1, 1)).astype(np.float32))
            .tags("broadcast", "elementwise", "div", "e2e")
            .build(),
        ]
