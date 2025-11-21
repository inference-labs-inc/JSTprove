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


class MulConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for Mul layer"""

    @property
    def layer_name(self) -> str:
        return "Mul"

    def get_config(self) -> LayerTestConfig:
        return LayerTestConfig(
            op_type="Mul",
            valid_inputs=["A", "B"],
            valid_attributes={},  # Mul has no layer-specific attributes
            required_initializers={},
            input_shapes={
                "A": [1, 3, 4, 4],
                "B": [1, 3, 4, 4],
            },
            output_shapes={
                "mul_output": [1, 3, 4, 4],
            },
        )

    def get_test_specs(self) -> list[LayerTestSpec]:
        rng = np.random.default_rng(TEST_RNG_SEED)
        return [
            # --- VALID TESTS ---
            valid_test("basic")
            .description("Basic elementwise Mul of two same-shaped tensors")
            .override_input_shapes(A=[1, 3, 4, 4], B=[1, 3, 4, 4])
            .tags("basic", "elementwise", "Mul")
            .build(),
            valid_test("broadcast_mul")
            .description("mul with Numpy-style broadcasting along spatial dimensions")
            .override_input_shapes(A=[1, 3, 4, 4], B=[1, 3, 1, 1])
            .tags("broadcast", "elementwise", "mul", "onnx14")
            .build(),
            valid_test("initializer_mul")
            .description(
                "mul where second input (B) is a tensor initializer instead of input",
            )
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", rng.normal(0, 1, (1, 3, 4, 4)))
            .tags("initializer", "elementwise", "mul", "onnxruntime")
            .build(),
            valid_test("scalar_mul")
            .description("mul scalar (initializer) to tensor")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", np.array([2.0], dtype=np.float32))
            .tags("scalar", "elementwise", "mul")
            .build(),
            # # --- E2E TESTS ---
            e2e_test("e2e_mul")
            .description("End-to-end mul test with random inputs")
            .override_input_shapes(A=[1, 3, 4, 4], B=[1, 3, 4, 4])
            .override_output_shapes(mul_output=[1, 3, 4, 4])
            .tags("e2e", "mul", "2d")
            .build(),
            e2e_test("e2e_initializer_mul")
            .description(
                "mul where second input (B) is a tensor initializer instead of input",
            )
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", rng.normal(0, 1, (1, 3, 4, 4)))
            .tags("initializer", "elementwise", "mul", "onnxruntime")
            .build(),
            e2e_test("e2e_broadcast_mul")
            .description("mul with Numpy-style broadcasting along spatial dimensions")
            .override_input_shapes(A=[1, 3, 4, 4], B=[1, 3, 1, 1])
            .tags("broadcast", "elementwise", "mul", "onnx14")
            .build(),
            e2e_test("e2e_scalar_mul")
            .description("mul scalar (initializer) to tensor")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", np.array([2.0], dtype=np.float32))
            .tags("scalar", "elementwise", "mul")
            .build(),
            # # --- EDGE CASES ---
            edge_case_test("empty_tensor")
            .description("mul with empty tensor input (zero elements)")
            .override_input_shapes(A=[0], B=[0])
            .tags("edge", "empty", "mul")
            .build(),
            edge_case_test("large_tensor")
            .description("Large tensor mul performance/stress test")
            .override_input_shapes(A=[1, 64, 256, 256], B=[1, 64, 256, 256])
            .tags("large", "performance", "mul")
            .skip("Performance test, skipped by default")
            .build(),
        ]
