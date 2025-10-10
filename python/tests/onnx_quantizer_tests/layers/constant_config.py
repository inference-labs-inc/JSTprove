import numpy as np
from onnx import numpy_helper

from python.tests.onnx_quantizer_tests.layers.factory import (
    BaseLayerConfigProvider,
    LayerTestConfig,
)


class ConstantConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for Constant layers"""

    @property
    def layer_name(self) -> str:
        return "Constant"

    def get_config(self) -> LayerTestConfig:
        return LayerTestConfig(
            op_type="Constant",
            valid_inputs=[],
            valid_attributes={
                "value": numpy_helper.from_array(np.array([1.0]), name="const_value"),
            },
            required_initializers={},
        )
