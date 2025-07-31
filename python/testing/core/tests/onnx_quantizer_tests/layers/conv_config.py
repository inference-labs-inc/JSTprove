import numpy as np
from python.testing.core.tests.onnx_quantizer_tests.layers.factory import BaseLayerConfigProvider, LayerTestConfig


class ConvConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for Conv layers"""
    
    @property
    def layer_name(self) -> str:
        return "Conv"
    
    def get_config(self) -> LayerTestConfig:
        return LayerTestConfig(
            op_type="Conv",
            valid_inputs=["input", "conv_weight", "conv_bias"],
            valid_attributes={
                "strides": [1, 1],
                "kernel_shape": [3, 3],
                "dilations": [1, 1],
                "pads": [1, 1, 1, 1]
            },
            required_initializers={
                "conv_weight": np.random.randn(32, 16, 3, 3),
                "conv_bias": np.random.randn(32)
            }
        )
    
    class ConvUnsupportedStrideConfigProvider(BaseLayerConfigProvider):
    @property
    def layer_name(self) -> str:
        return "Conv_UnsupportedStride"

    def get_config(self) -> LayerTestConfig:
        return LayerTestConfig(
            op_type="Conv_UnsupportedStride",  # must match layer_name to avoid factory test failure
            valid_inputs=["input", "conv_weight", "conv_bias"],
            valid_attributes={
                "strides": [2, 2],  # unsupported
                "kernel_shape": [3, 3],
                "dilations": [1, 1],
                "pads": [1, 1, 1, 1]
            },
            required_initializers={
                "conv_weight": np.random.randn(32, 16, 3, 3),
                "conv_bias": np.random.randn(32)
            }
        )
