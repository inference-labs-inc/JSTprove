from python.tests.onnx_quantizer_tests.layers.factory import (
    BaseLayerConfigProvider,
    LayerTestConfig,
)


class FlattenConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for Flatten layers"""

    @property
    def layer_name(self) -> str:
        return "Flatten"

    def get_config(self) -> LayerTestConfig:
        return LayerTestConfig(
            op_type="Flatten",
            valid_inputs=["input"],
            valid_attributes={"axis": 1},
            required_initializers={},
        )
