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
