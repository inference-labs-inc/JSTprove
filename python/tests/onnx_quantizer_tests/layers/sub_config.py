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


class SubConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for Sub layer"""

    @property
    def layer_name(self) -> str:
        return "Sub"

    def get_config(self) -> LayerTestConfig:
        return LayerTestConfig(
            op_type="Sub",
            valid_inputs=["A", "B"],
            valid_attributes={},  # Sub has no layer-specific attributes
            required_initializers={},
            input_shapes={
                "A": [1, 3, 4, 4],
                "B": [1, 3, 4, 4],
            },
            output_shapes={
                "sub_output": [1, 3, 4, 4],
            },
        )

    def get_test_specs(self) -> list[LayerTestSpec]:
        rng = np.random.default_rng(TEST_RNG_SEED)
        return [
            # --- VALID TESTS ---
            valid_test("basic")
            .description("Basic elementwise Sub of two same-shaped tensors")
            .override_input_shapes(A=[1, 3, 4, 4], B=[1, 3, 4, 4])
            .tags("basic", "elementwise", "Sub")
            .build(),
            valid_test("broadcast_Sub")
            .description("Sub with Numpy-style broadcasting along spatial dimensions")
            .override_input_shapes(A=[1, 3, 4, 4], B=[1, 3, 1, 1])
            .tags("broadcast", "elementwise", "Sub", "onnx14")
            .build(),
            valid_test("initializer_Sub")
            .description(
                "Sub where second input (B) is a tensor initializer instead of input",
            )
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", rng.normal(0, 1, (1, 3, 4, 4)))
            .tags("initializer", "elementwise", "Sub", "onnxruntime")
            .build(),
            valid_test("scalar_Sub")
            .description("Sub scalar (initializer) to tensor")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", np.array([2.0], dtype=np.float32))
            .tags("scalar", "elementwise", "Sub")
            .build(),
            # --- E2E TESTS ---
            e2e_test("e2e_Sub")
            .description("End-to-end Sub test with random inputs")
            .override_input_shapes(A=[1, 3, 4, 4], B=[1, 3, 4, 4])
            .override_output_shapes(sub_output=[1, 3, 4, 4])
            .tags("e2e", "Sub", "2d")
            .build(),
            e2e_test("e2e_initializer_Sub")
            .description(
                "Sub where second input (B) is a tensor initializer instead of input",
            )
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", rng.normal(0, 1, (1, 3, 4, 4)))
            .tags("initializer", "elementwise", "Sub", "onnxruntime")
            .build(),
            e2e_test("e2e_broadcast_Sub")
            .description("Sub with Numpy-style broadcasting along spatial dimensions")
            .override_input_shapes(A=[1, 3, 4, 4], B=[1, 3, 1, 1])
            .tags("broadcast", "elementwise", "Sub", "onnx14")
            .build(),
            e2e_test("e2e_scalar_Sub")
            .description("Sub scalar (initializer) to tensor")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", np.array([2.0], dtype=np.float32))
            .tags("scalar", "elementwise", "Sub")
            .build(),
            # # --- EDGE CASES ---
            edge_case_test("empty_tensor")
            .description("Sub with empty tensor input (zero elements)")
            .override_input_shapes(A=[0], B=[0])
            .tags("edge", "empty", "Sub")
            .build(),
            edge_case_test("large_tensor")
            .description("Large tensor Sub performance/stress test")
            .override_input_shapes(A=[1, 64, 256, 256], B=[1, 64, 256, 256])
            .tags("large", "performance", "Sub")
            .skip("Performance test, skipped by default")
            .build(),
        ]
