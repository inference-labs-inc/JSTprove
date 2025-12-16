from __future__ import annotations

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


class SqueezeConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for Squeeze"""

    @property
    def layer_name(self) -> str:
        return "Squeeze"

    def get_config(self) -> LayerTestConfig:
        # NOTE: We use attribute-based axes for now.
        # Testing axes-as-initializer requires extending create_initializers()
        # to allow int64 initializers for name="axes".
        return LayerTestConfig(
            op_type="Squeeze",
            valid_inputs=["A"],  # axes may be omitted or provided as attribute
            valid_attributes={},  # overridden per-test as needed
            required_initializers={},
            input_shapes={
                "A": [1, 3, 1, 5],
            },
            output_shapes={
                "squeeze_output": [3, 5],
            },
        )

    def get_test_specs(self) -> list:
        return [
            # --- VALID TESTS ---
            valid_test("axes_omitted")
            .description("Squeeze with no axes attribute: removes all dims of size 1")
            .override_input_shapes(A=[1, 3, 1, 5])
            .override_output_shapes(squeeze_output=[3, 5])
            .tags("basic", "squeeze", "axes_omitted")
            .build(),
            valid_test("axes_attr_basic")
            .description(
                "Squeeze with axes attribute [0, 2] on shape [1,3,1,5] -> [3,5]",
            )
            .override_attrs(axes=[0, 2])
            .override_input_shapes(A=[1, 3, 1, 5])
            .override_output_shapes(squeeze_output=[3, 5])
            .tags("basic", "squeeze", "axes_attr")
            .build(),
            valid_test("axes_attr_singleton_middle")
            .description("Squeeze with axes attribute [1] on shape [2,1,4] -> [2,4]")
            .override_attrs(axes=[1])
            .override_input_shapes(A=[2, 1, 4])
            .override_output_shapes(squeeze_output=[2, 4])
            .tags("squeeze", "axes_attr")
            .build(),
            valid_test("axes_attr_negative")
            .description("Squeeze with negative axis [-2] on shape [2,1,4] -> [2,4]")
            .override_attrs(axes=[-2])
            .override_input_shapes(A=[2, 1, 4])
            .override_output_shapes(squeeze_output=[2, 4])
            .tags("squeeze", "axes_attr", "negative_axis")
            .build(),
            # --- ERROR TESTS (quantizer-level validation) ---
            error_test("duplicate_axes_attr")
            .description("Duplicate axes in attribute should be rejected")
            .override_attrs(axes=[1, 1])
            .override_input_shapes(A=[2, 1, 4])
            .override_output_shapes(squeeze_output=[2, 4])
            .expects_error(InvalidParamError, match="axes must not contain duplicates")
            .tags("error", "squeeze", "axes_attr")
            .build(),
            error_test("dynamic_axes_input_not_supported")
            .description(
                "Squeeze with 2 inputs where axes is not an initializer "
                "should be rejected",
            )
            .override_inputs("A", "axes")  # axes as runtime input (unsupported)
            .override_input_shapes(
                A=[1, 3, 1, 5],
                axes=[2],
            )  # axes becomes a graph input
            .override_output_shapes(squeeze_output=[3, 5])
            .expects_error(
                InvalidParamError,
                match="Dynamic axes input is not supported",
            )
            .tags("error", "squeeze", "axes_input")
            .build(),
            # --- E2E TESTS ---
            e2e_test("e2e_axes_omitted")
            .description("End-to-end Squeeze test (axes omitted)")
            .override_input_shapes(A=[1, 3, 1, 5])
            .override_output_shapes(squeeze_output=[3, 5])
            .tags("e2e", "squeeze")
            .build(),
            e2e_test("e2e_axes_attr")
            .description("End-to-end Squeeze test (axes attribute)")
            .override_attrs(axes=[0, 2])
            .override_input_shapes(A=[1, 3, 1, 5])
            .override_output_shapes(squeeze_output=[3, 5])
            .tags("e2e", "squeeze", "axes_attr")
            .build(),
        ]
