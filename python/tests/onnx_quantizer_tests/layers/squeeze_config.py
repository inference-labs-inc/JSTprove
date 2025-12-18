from __future__ import annotations

import numpy as np

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
        # Test opset-newer form: Squeeze(data, axes) where axes is an int64 initializer.
        return LayerTestConfig(
            op_type="Squeeze",
            valid_inputs=["A", "axes"],
            valid_attributes={},  # no attribute-based axes
            required_initializers={},
            input_shapes={
                "A": [1, 3, 1, 5],
                # "axes" will be removed from graph inputs automatically when it is an initializer.
                "axes": [2],
            },
            output_shapes={
                "squeeze_output": [3, 5],
            },
        )

    def get_test_specs(self) -> list:

        return [
            # --- VALID TESTS ---
            valid_test("axes_omitted")
            .description("Squeeze with no axes input: removes all dims of size 1")
            .override_inputs("A")  # only data input
            .override_input_shapes(A=[1, 3, 1, 5])
            .override_output_shapes(squeeze_output=[3, 5])
            .tags("basic", "squeeze", "axes_omitted")
            .build(),
            valid_test("axes_init_basic")
            .description("Squeeze with axes initializer [0,2] on [1,3,1,5] -> [3,5]")
            .override_inputs("A", "axes")
            .override_initializer("axes", np.array([0, 2], dtype=np.int64))
            .override_input_shapes(A=[1, 3, 1, 5])
            .override_output_shapes(squeeze_output=[3, 5])
            .tags("basic", "squeeze", "axes_initializer")
            .build(),
            valid_test("axes_init_singleton_middle")
            .description("Squeeze with axes initializer [1] on [2,1,4] -> [2,4]")
            .override_inputs("A", "axes")
            .override_initializer("axes", np.array([1], dtype=np.int64))
            .override_input_shapes(A=[2, 1, 4])
            .override_output_shapes(squeeze_output=[2, 4])
            .tags("squeeze", "axes_initializer")
            .build(),
            valid_test("axes_init_negative")
            .description("Squeeze with axes initializer [-2] on [2,1,4] -> [2,4]")
            .override_inputs("A", "axes")
            .override_initializer("axes", np.array([-2], dtype=np.int64))
            .override_input_shapes(A=[2, 1, 4])
            .override_output_shapes(squeeze_output=[2, 4])
            .tags("squeeze", "axes_initializer", "negative_axis")
            .build(),
            # --- ERROR TESTS ---
            error_test("duplicate_axes_init")
            .description("Duplicate axes in initializer should be rejected")
            .override_inputs("A", "axes")
            .override_initializer("axes", np.array([1, 1], dtype=np.int64))
            .override_input_shapes(A=[2, 1, 4])
            .override_output_shapes(squeeze_output=[2, 4])
            .expects_error(InvalidParamError, match="axes must not contain duplicates")
            .tags("error", "squeeze", "axes_initializer")
            .build(),
            error_test("dynamic_axes_input_not_supported")
            .description(
                "Squeeze with runtime axes (2 inputs but axes is NOT an initializer) should be rejected"
            )
            .override_inputs("A", "axes")  # axes provided as graph input (unsupported)
            .override_input_shapes(A=[1, 3, 1, 5], axes=[2])
            .override_output_shapes(squeeze_output=[3, 5])
            .expects_error(
                InvalidParamError, match="Dynamic axes input is not supported"
            )
            .tags("error", "squeeze", "axes_input")
            .build(),
            # --- E2E TESTS ---
            e2e_test("e2e_axes_omitted")
            .description("End-to-end Squeeze test (axes omitted)")
            .override_inputs("A")
            .override_input_shapes(A=[1, 3, 1, 5])
            .override_output_shapes(squeeze_output=[3, 5])
            .tags("e2e", "squeeze")
            .build(),
            e2e_test("e2e_axes_init")
            .description("End-to-end Squeeze test (axes initializer)")
            .override_inputs("A", "axes")
            .override_initializer("axes", np.array([0, 2], dtype=np.int64))
            .override_input_shapes(A=[1, 3, 1, 5])
            .override_output_shapes(squeeze_output=[3, 5])
            .tags("e2e", "squeeze", "axes_initializer")
            .build(),
        ]
