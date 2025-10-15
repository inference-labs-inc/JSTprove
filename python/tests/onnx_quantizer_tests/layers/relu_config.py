from python.tests.onnx_quantizer_tests.layers.base import e2e_test, valid_test
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
            valid_test("basic")
            .description("Basic ReLU activation")
            .tags("basic", "activation")
            .build(),
            e2e_test("e2e_basic")
            .description("End-to-end test for ReLU activation")
            .override_input_shapes(input=[1, 3, 4, 4])
            .override_output_shapes(relu_output=[1, 3, 4, 4])
            .tags("e2e", "activation")
            .build(),
        ]
