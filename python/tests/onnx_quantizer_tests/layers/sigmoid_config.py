from python.tests.onnx_quantizer_tests.layers.base import (
    e2e_test,
    valid_test,
)
from python.tests.onnx_quantizer_tests.layers.factory import (
    BaseLayerConfigProvider,
    LayerTestConfig,
)


class SigmoidConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for Sigmoid layers"""

    @property
    def layer_name(self) -> str:
        return "Sigmoid"

    def get_config(self) -> LayerTestConfig:
        return LayerTestConfig(
            op_type="Sigmoid",
            valid_inputs=["input"],
            valid_attributes={},
            required_initializers={},
        )

    def get_test_specs(self) -> list:
        return [
            valid_test("basic")
            .description("Basic Sigmoid activation")
            .tags("basic", "activation")
            .build(),
            valid_test("negative_inputs")
            .description("Sigmoid with negative input values (should output near 0)")
            .override_input_shapes(input=[1, 3, 4, 4])
            .tags("activation", "negative_values")
            .build(),
            valid_test("positive_inputs")
            .description("Sigmoid with positive input values (should output near 1)")
            .override_input_shapes(input=[1, 3, 4, 4])
            .tags("activation", "positive_values")
            .build(),
            valid_test("high_dimension_input")
            .description("Sigmoid applied to a 5D input tensor")
            .override_input_shapes(input=[1, 3, 4, 4, 2])
            .tags("activation", "high_dim", "5d")
            .build(),
            valid_test("scalar_input")
            .description("Sigmoid with scalar input")
            .override_input_shapes(input=[1])
            .tags("activation", "scalar")
            .build(),
            e2e_test("e2e_basic")
            .description("End-to-end test for Sigmoid activation")
            .override_input_shapes(input=[1, 3, 4, 4])
            .override_output_shapes(sigmoid_output=[1, 3, 4, 4])
            .tags("e2e", "activation")
            .build(),
            valid_test("large_input")
            .description("Large input tensor for Sigmoid")
            .override_input_shapes(input=[1, 3, 64, 64])
            .tags("large", "activation")
            .skip("Performance test, skipped by default")
            .build(),
        ]
