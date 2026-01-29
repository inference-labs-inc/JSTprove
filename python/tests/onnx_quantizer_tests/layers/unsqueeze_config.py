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


class UnsqueezeConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for Unsqueeze"""

    @property
    def layer_name(self) -> str:
        return "Unsqueeze"

    def get_config(self) -> LayerTestConfig:
        # Test opset-newer form: Unsqueeze(data, axes)
        # where axes is an int64 initializer.
        return LayerTestConfig(
            op_type="Unsqueeze",
            valid_inputs=["A", "axes"],
            valid_attributes={},  # no attribute-based axes
            required_initializers={},
            input_shapes={
                "A": [3, 5],
                # "axes" will be removed from graph inputs automatically
                # when it is an initializer.
                "axes": [2],
            },
            output_shapes={
                "unsqueeze_output": [1, 3, 1, 5],
            },
        )

    def get_test_specs(self) -> list:

        return [
            # --- VALID TESTS ---
            valid_test("axes_init_basic")
            .description("Unsqueeze with axes initializer [0,2] on [3,5] -> [1,3,1,5]")
            .override_inputs("A", "axes")
            .override_initializer("axes", np.array([0, 2], dtype=np.int64))
            .override_input_shapes(A=[3, 5])
            .override_output_shapes(unsqueeze_output=[1, 3, 1, 5])
            .tags("basic", "unsqueeze", "axes_initializer")
            .build(),
            valid_test("axes_init_single_axis")
            .description("Unsqueeze with axes initializer [1] on [3,5] -> [3,1,5]")
            .override_inputs("A", "axes")
            .override_initializer("axes", np.array([1], dtype=np.int64))
            .override_input_shapes(A=[3, 5])
            .override_output_shapes(unsqueeze_output=[3, 1, 5])
            .tags("unsqueeze", "axes_initializer")
            .build(),
            valid_test("axes_init_negative")
            .description("Unsqueeze with negative axis [-1] on [3,5] -> [3,5,1]")
            .override_inputs("A", "axes")
            .override_initializer("axes", np.array([-1], dtype=np.int64))
            .override_input_shapes(A=[3, 5])
            .override_output_shapes(unsqueeze_output=[3, 5, 1])
            .tags("unsqueeze", "axes_initializer", "negative_axis")
            .build(),
            valid_test("axes_init_two_axes_append")
            .description("Unsqueeze with axes [2,3] on [3,5] -> [3,5,1,1]")
            .override_inputs("A", "axes")
            .override_initializer("axes", np.array([2, 3], dtype=np.int64))
            .override_input_shapes(A=[3, 5])
            .override_output_shapes(unsqueeze_output=[3, 5, 1, 1])
            .tags("unsqueeze", "axes_initializer")
            .build(),
            # --- ERROR TESTS ---
            error_test("duplicate_axes_init")
            .description("Duplicate axes in initializer should be rejected")
            .override_inputs("A", "axes")
            .override_initializer("axes", np.array([1, 1], dtype=np.int64))
            .override_input_shapes(A=[3, 5])
            .override_output_shapes(
                unsqueeze_output=[3, 1, 5],
            )  # not used; kept consistent
            .expects_error(InvalidParamError, match="axes must not contain duplicates")
            .tags("error", "unsqueeze", "axes_initializer")
            .build(),
            error_test("dynamic_axes_input_not_supported")
            .description(
                "Unsqueeze with runtime axes (2 inputs but axes is NOT an initializer) "
                "should be rejected",
            )
            .override_inputs("A", "axes")  # axes provided as graph input (unsupported)
            .override_input_shapes(A=[3, 5], axes=[2])
            .override_output_shapes(unsqueeze_output=[1, 3, 1, 5])
            .expects_error(
                InvalidParamError,
                match="Dynamic axes input is not supported",
            )
            .tags("error", "unsqueeze", "axes_input")
            .build(),
            # --- E2E TESTS ---
            e2e_test("e2e_axes_init")
            .description("End-to-end Unsqueeze test (axes initializer)")
            .override_inputs("A", "axes")
            .override_initializer("axes", np.array([0, 2], dtype=np.int64))
            .override_input_shapes(A=[3, 5])
            .override_output_shapes(unsqueeze_output=[1, 3, 1, 5])
            .tags("e2e", "unsqueeze", "axes_initializer")
            .build(),
        ]
