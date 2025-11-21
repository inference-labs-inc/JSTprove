import numpy as np

from python.core.model_processing.onnx_quantizer.exceptions import InvalidParamError
from python.tests.onnx_quantizer_tests import TEST_RNG_SEED
from python.tests.onnx_quantizer_tests.layers.base import (
    LayerTestSpec,
    e2e_test,
    error_test,
    valid_test,
)
from python.tests.onnx_quantizer_tests.layers.factory import (
    BaseLayerConfigProvider,
    LayerTestConfig,
)


class ConvConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for Conv layers"""

    @property
    def layer_name(self) -> str:
        return "Conv"

    def get_config(self) -> LayerTestConfig:
        rng = np.random.default_rng(TEST_RNG_SEED)
        return LayerTestConfig(
            op_type="Conv",
            valid_inputs=["input", "conv_weight", "conv_bias"],
            valid_attributes={
                "strides": [1, 1],
                "kernel_shape": [3, 3],
                "dilations": [1, 1],
                "pads": [1, 1, 1, 1],
            },
            required_initializers={
                "conv_weight": rng.normal(0, 1, (32, 16, 3, 3)),
                "conv_bias": rng.normal(0, 1, 32),
            },
        )

    def get_test_specs(self) -> list[LayerTestSpec]:
        """Return all test specifications for Conv layers"""
        rng = np.random.default_rng(TEST_RNG_SEED)
        return [
            # Valid variations
            valid_test("basic")
            .description("Basic 2D convolution")
            .tags("basic", "2d")
            .build(),
            valid_test("different_padding")
            .description("Convolution with different padding")
            .override_attrs(pads=[2, 2, 2, 2], kernel_shape=[5, 5])
            .override_initializer("conv_weight", rng.normal(0, 1, (32, 16, 5, 5)))
            .tags("padding", "5x5_kernel")
            .build(),
            # E2E test
            e2e_test("e2e_basic")
            .description("End-to-end test for basic 2D convolution")
            .override_input_shapes(input=[1, 3, 4, 4])
            .override_output_shapes(conv_output=[1, 8, 4, 4])
            .override_initializer("conv_weight", rng.normal(0, 1, (8, 3, 3, 3)))
            .override_initializer("conv_bias", rng.normal(0, 1, 8))
            .tags("e2e", "basic", "2d")
            .build(),
            # Error cases
            error_test("no_bias")
            .description("2D convolution without bias")
            .override_inputs("input", "conv_weight")
            .override_attrs(strides=[2, 2], kernel_shape=[5, 5])
            .override_initializer("conv_weight", rng.normal(0, 1, (64, 16, 5, 5)))
            .expects_error(
                InvalidParamError,
                "Expected at least 3 inputs (input, weights, bias), got 2",
            )
            .tags("no_bias", "stride_2")
            .build(),
            error_test("conv3d_unsupported")
            .description("3D convolution should raise error")
            .override_attrs(
                kernel_shape=[3, 3, 3],
                strides=[1, 1, 1],
                dilations=[1, 1, 1],
                pads=[1, 1, 1, 1, 1, 1],
            )
            .override_initializer(
                "conv_weight",
                rng.normal(0, 1, (32, 16, 3, 3, 3)),
            )
            .expects_error(
                InvalidParamError,
                "Unsupported Conv weight dimensionality 5",
            )
            .tags("3d", "unsupported")
            .build(),
            error_test("invalid_stride")
            .description("Invalid stride values")
            .override_attrs(strides=[0, 1])
            .override_inputs("input", "conv_weight")
            .expects_error(InvalidParamError, "stride must be positive")
            .tags("invalid_params")
            .skip("Not yet supported")
            .build(),
            error_test("negative_dilation")
            .description("Negative dilation values")
            .override_attrs(dilations=[-1, 1])
            .expects_error(InvalidParamError, "dilation must be positive")
            .tags("invalid_params")
            .skip("Not yet supported")
            .build(),
            error_test("invalid_kernel_shape_long")
            .description("kernel_shape too long (length 3)")
            .override_attrs(kernel_shape=[3, 3, 3])
            .override_initializer(
                "conv_weight",
                rng.normal(0, 1, (32, 16, 3, 3, 3)),
            )
            .expects_error(InvalidParamError, "kernel_shape")
            .tags("invalid_attr_length")
            .build(),
            # Missing required attributes
            error_test("missing_strides")
            .description("Conv node missing 'strides' attribute")
            .omit_attrs("strides")  # exclude strides
            .override_attrs(
                kernel_shape=[3, 3],
                dilations=[1, 1],
                pads=[1, 1, 1, 1],
            )  # supply others explicitly
            .expects_error(InvalidParamError, "strides")
            .tags("missing_attr")
            .build(),
            error_test("missing_kernel_shape")
            .description("Conv node missing 'kernel_shape' attribute")
            .omit_attrs("kernel_shape")  # exclude kernel_shape
            .override_attrs(
                strides=[3, 3],
                dilations=[1, 1],
                pads=[1, 1, 1, 1],
            )  # supply others explicitly
            .expects_error(InvalidParamError, "kernel_shape")
            .tags("missing_attr")
            .build(),
            error_test("missing_dilations")
            .description("Conv node missing 'dilations' attribute")
            .omit_attrs("dilations")  # exclude dilations
            .override_attrs(
                strides=[3, 3],
                kernel_shape=[3, 3],
                pads=[1, 1, 1, 1],
            )  # supply others explicitly
            .expects_error(InvalidParamError, "dilations")
            .tags("missing_attr")
            .build(),
        ]
