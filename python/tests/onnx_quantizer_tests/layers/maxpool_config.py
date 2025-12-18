from python.core.model_processing.onnx_quantizer.exceptions import InvalidParamError
from python.tests.onnx_quantizer_tests.layers.base import (
    e2e_test,
    error_test,
    valid_test,
)
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
            input_shapes={"input": [1, 3, 4, 4]},
            output_shapes={"maxpool_output": [1, 3, 2, 2]},
        )

    def get_test_specs(self) -> list:
        return [
            # --- VALID TESTS ---
            valid_test("basic")
            .description("Basic MaxPool with 2x2 kernel and stride 2")
            .tags("basic", "pool", "2d")
            .build(),
            valid_test("larger_kernel")
            .description("MaxPool with 3x3 kernel and stride 1")
            .override_attrs(kernel_shape=[3, 3], strides=[1, 1])
            .tags("kernel_3x3", "stride_1", "pool")
            .build(),
            valid_test("dilated_pool")
            .description("MaxPool with dilation > 1")
            .override_attrs(dilations=[2, 2])
            .tags("dilation", "pool")
            .build(),
            valid_test("stride_one")
            .description("MaxPool with stride 1 (overlapping windows)")
            .override_attrs(strides=[1, 1])
            .tags("stride_1", "pool", "overlap")
            .build(),
            valid_test("missing_dilations_attr")
            .description("MaxPool without dilations attribute should default to [1, 1]")
            .override_attrs(dilations=None)
            .tags("default_attr", "dilations", "pool")
            .build(),
            valid_test("non_default_dilations")
            .description("MaxPool with explicit non-default dilations")
            .override_attrs(dilations=[2, 2])
            .tags("dilations", "non_default", "pool")
            .override_output_shapes(maxpool_output=[1, 3, 1, 1])
            .build(),
            valid_test("missing_optional_attrs")
            .description(
                "MaxPool without pads/strides/dilations should use default values",
            )
            .override_attrs(
                pads=None,
                strides=None,
                dilations=None,
            )
            .tags("defaults", "optional_attrs", "pool")
            .build(),
            valid_test("non_default_pads")
            .description("MaxPool with explicit non-default pads")
            .override_attrs(pads=[1, 1, 1, 1])
            .override_output_shapes(maxpool_output=[1, 3, 3, 3])
            .tags("pads", "non_default", "pool")
            .build(),
            valid_test("non_default_strides")
            .description("MaxPool with explicit non-default strides [3, 3]")
            .override_attrs(strides=[3, 3])
            .override_output_shapes(maxpool_output=[1, 3, 1, 1])
            .tags("strides", "non_default", "pool")
            .build(),
            valid_test("rectangular_strides")
            .description("MaxPool with non-square strides [2, 1]")
            .override_attrs(strides=[2, 1])
            .override_output_shapes(maxpool_output=[1, 3, 2, 3])
            .tags("strides", "non_square", "pool")
            .build(),
            # --- E2E TESTS ---
            e2e_test("e2e_basic")
            .description("End-to-end test for 2D MaxPool")
            .override_input_shapes(input=[1, 3, 4, 4])
            .override_output_shapes(maxpool_output=[1, 3, 2, 2])
            .tags("e2e", "pool", "2d")
            .build(),
            e2e_test("missing_dilations_attr")
            .description("MaxPool without dilations attribute should default to [1, 1]")
            .override_attrs(dilations=None)
            .tags("default_attr", "dilations", "pool")
            .build(),
            e2e_test("non_default_dilations")
            .description("MaxPool with explicit non-default dilations")
            .override_attrs(dilations=[2, 2])
            .override_output_shapes(maxpool_output=[1, 3, 1, 1])
            .tags("dilations", "non_default", "pool")
            .build(),
            e2e_test("e2e_defaults_applied")
            .description("E2E MaxPool with default pads/strides/dilations applied")
            .override_attrs(
                pads=None,
                strides=None,
                dilations=None,
            )
            .override_input_shapes(input=[1, 1, 4, 4])
            .override_output_shapes(maxpool_output=[1, 1, 3, 3])
            .tags("e2e", "defaults", "pool")
            .build(),
            e2e_test("non_default_pads")
            .description("MaxPool with explicit non-default pads")
            .override_attrs(pads=[1, 1, 1, 1])
            .override_output_shapes(maxpool_output=[1, 3, 3, 3])
            .tags("pads", "non_default", "pool")
            .build(),
            e2e_test("non_default_strides")
            .description("E2E MaxPool with explicit non-default strides")
            .override_attrs(strides=[3, 3])
            .override_input_shapes(input=[1, 3, 4, 4])
            .override_output_shapes(maxpool_output=[1, 3, 1, 1])
            .tags("e2e", "strides", "non_default", "pool")
            .build(),
            e2e_test("missing_strides_attr_defaults_applied")
            .description(
                "E2E MaxPool without strides attribute should default to [1, 1]",
            )
            .override_attrs(strides=None)
            .override_input_shapes(input=[1, 1, 4, 4])
            .override_output_shapes(maxpool_output=[1, 1, 3, 3])
            .tags("e2e", "defaults", "strides", "pool")
            .build(),
            # # --- ERROR TESTS ---
            error_test("asymmetric_padding")
            .description("MaxPool with asymmetric padding")
            .override_attrs(pads=[1, 0, 2, 1])
            .expects_error(InvalidParamError, "pads[2]=2 >= kernel[0]=2")
            .tags("padding", "asymmetric", "pool")
            .build(),
            error_test("invalid_kernel_shape")
            .description("Invalid kernel shape length (3D instead of 2D)")
            .override_attrs(kernel_shape=[2, 2, 2])
            .expects_error(InvalidParamError, "Currently only MaxPool2D is supported")
            .tags("invalid_attr_length", "kernel_shape")
            .build(),
            error_test("auto_pad_not_supported")
            .description("MaxPool with auto_pad should be rejected")
            .override_attrs(auto_pad="SAME_UPPER")
            .expects_error(InvalidParamError, "auto_pad must be NOTSET")
            .tags("invalid", "auto_pad", "pool")
            .build(),
            error_test("ceil_mode_not_supported")
            .description("MaxPool with ceil_mode != 0 should be rejected")
            .override_attrs(ceil_mode=1)
            .expects_error(InvalidParamError, "ceil_mode must be 0")
            .tags("invalid", "ceil_mode", "pool")
            .build(),
            error_test("storage_order_not_supported")
            .description("MaxPool with storage_order != 0 should be rejected")
            .override_attrs(storage_order=1)
            .expects_error(InvalidParamError, "storage_order must be 0")
            .tags("invalid", "storage_order", "pool")
            .build(),
            # This can be removed if we start to support explicit None strides
            # --- EDGE CASE / SKIPPED TEST ---
            valid_test("large_input")
            .description("Large MaxPool input (performance/stress test)")
            .override_input_shapes(input=[1, 3, 64, 64])
            .override_attrs(kernel_shape=[3, 3], strides=[2, 2])
            .tags("large", "performance", "pool")
            .skip("Performance test, skipped by default")
            .build(),
        ]
