from typing import List
import numpy as np
from python.testing.core.tests.onnx_quantizer_tests.layers.base import TestSpec, error_test, valid_test
from python.testing.core.tests.onnx_quantizer_tests.layers.factory import BaseLayerConfigProvider, LayerTestConfig
from python.testing.core.utils.onnx_quantizer.exceptions import InvalidParamError


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
    
    def get_test_specs(self) -> List[TestSpec]:
        """Return all test specifications for Conv layers"""
        return [
            # Valid variations
            valid_test("basic")
            .description("Basic 2D convolution")
            .tags("basic", "2d")
            .build(),
            
            
            
            valid_test("different_padding")
            .description("Convolution with different padding")
            .override_attrs(pads=[2, 2, 2, 2], kernel_shape=[5, 5])
            .override_initializer("conv_weight", np.random.randn(32, 16, 5, 5))
            .tags("padding", "5x5_kernel")
            .build(),
            
            # # # Error cases
            error_test("no_bias")
            .description("2D convolution without bias")
            .override_inputs("input", "conv_weight")
            .override_attrs(strides=[2, 2], kernel_shape=[5, 5])
            .override_initializer("conv_weight", np.random.randn(64, 16, 5, 5))
            .expects_error(InvalidParamError, "Expected at least 3 inputs (input, weights, bias), got 2")
            .tags("no_bias", "stride_2")
            .build(),
            error_test("conv3d_unsupported")
            .description("3D convolution should raise error")
            .override_attrs(
                kernel_shape=[3, 3, 3],
                strides=[1, 1, 1],
                dilations=[1, 1, 1],
                pads=[1, 1, 1, 1, 1, 1]
            )
            .override_initializer("conv_weight", np.random.randn(32, 16, 3, 3, 3))
            .expects_error(InvalidParamError, "3D Conv is not currently supported")
            .tags("3d", "unsupported")
            .build(),
            
            # error_test("invalid_stride")
            # .description("Invalid stride values")
            # .override_attrs(strides=[0, 1])
            # .override_inputs("input", "conv_weight")
            # .expects_error(InvalidParamError, "stride must be positive")
            # .tags("invalid_params")
            # .build(),
            
            # error_test("negative_dilation")
            # .description("Negative dilation values")
            # .override_attrs(dilations=[-1, 1])
            # .expects_error(InvalidParamError, "dilation must be positive")
            # .tags("invalid_params")
            # .build(),
        ]
