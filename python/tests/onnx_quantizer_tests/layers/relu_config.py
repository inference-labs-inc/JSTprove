from python.tests.onnx_quantizer_tests.layers.base import (
    e2e_test,
    valid_test,
)
from python.tests.onnx_quantizer_tests.layers.factory import (
    BaseLayerConfigProvider,
    LayerTestConfig,
)


class ReluConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for Relu layers"""

    @property
    def layer_name(self) -> str:
        return "Relu"

    def get_config(self) -> LayerTestConfig:
        return LayerTestConfig(
            op_type="Relu",
            valid_inputs=["input"],
            valid_attributes={},
            required_initializers={},
        )

    def get_test_specs(self) -> list:
        return [
            # --- VALID TESTS ---
            valid_test("basic")
            .description("Basic ReLU activation")
            .tags("basic", "activation")
            .build(),
            valid_test("negative_inputs")
            .description("ReLU should zero out negative input values")
            .override_input_shapes(input=[1, 3, 4, 4])
            .tags("activation", "negative_values")
            .build(),
            valid_test("high_dimension_input")
            .description("ReLU applied to a 5D input tensor (NCHWT layout)")
            .override_input_shapes(input=[1, 3, 4, 4, 2])
            .tags("activation", "high_dim", "5d")
            .build(),
            valid_test("scalar_input")
            .description("ReLU with scalar input (edge case)")
            .override_input_shapes(input=[1])
            .tags("activation", "scalar")
            .build(),
            e2e_test("e2e_basic")
            .description("End-to-end test for ReLU activation")
            .override_input_shapes(input=[1, 3, 4, 4])
            .override_output_shapes(relu_output=[1, 3, 4, 4])
            .tags("e2e", "activation")
            .build(),
            # --- EDGE CASE / SKIPPED TEST ---
            valid_test("large_input")
            .description("Large input tensor for ReLU (performance/stress test)")
            .override_input_shapes(input=[1, 3, 512, 512])
            .tags("large", "performance", "activation")
            .skip("Performance test, skipped by default")
            .build(),
        ]
