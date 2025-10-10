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
