from python.tests.onnx_quantizer_tests.layers.base import e2e_test, valid_test
from python.tests.onnx_quantizer_tests.layers.factory import (
    BaseLayerConfigProvider,
    LayerTestConfig,
)


class MaxPoolConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for MaxPool layers"""

    @property
    def layer_name(self) -> str:
        return "MaxPool"

    def get_config(self) -> LayerTestConfig:
        return LayerTestConfig(
            op_type="MaxPool",
            valid_inputs=["input"],
            valid_attributes={
                "kernel_shape": [2, 2],
                "strides": [2, 2],
                "dilations": [1, 1],
                "pads": [0, 0, 0, 0],
            },
            required_initializers={},
        )

    def get_test_specs(self) -> list:
        return [
            valid_test("basic")
            .description("Basic MaxPool with 2x2 kernel and stride 2")
            .tags("basic", "pool", "2d")
            .build(),
            e2e_test("e2e_basic")
            .description("End-to-end test for 2D MaxPool")
            .override_input_shapes(input=[1, 3, 4, 4])
            .override_output_shapes(maxpool_output=[1, 3, 2, 2])
            .tags("e2e", "pool", "2d")
            .build(),
        ]
