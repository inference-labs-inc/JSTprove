import numpy as np

from python.tests.onnx_quantizer_tests.layers.factory import (
    BaseLayerConfigProvider,
    LayerTestConfig,
)


class ReshapeConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for Reshape layers"""

    @property
    def layer_name(self) -> str:
        return "Reshape"

    def get_config(self) -> LayerTestConfig:
        return LayerTestConfig(
            op_type="Reshape",
            valid_inputs=["input", "shape"],
            valid_attributes={},
            required_initializers={"shape": np.array([1, -1])},
        )
