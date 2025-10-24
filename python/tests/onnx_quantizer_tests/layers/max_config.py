# python/tests/onnx_quantizer_tests/layers/max_config.py
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


class MaxConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for elementwise Max"""

    @property
    def layer_name(self) -> str:
        return "Max"

    def get_config(self) -> LayerTestConfig:
        # IMPORTANT: factory expects numpy arrays for required_initializers
        init_b = np.ones((1, 3, 4, 4), dtype=np.int64)
        return LayerTestConfig(
            op_type="Max",
            # Two inputs: x is dynamic, b is an initializer
            valid_inputs=["x", "b"],
            valid_attributes={},  # Max has no attributes
            required_initializers={"b": init_b},
        )

    def get_test_specs(self) -> list:
        return [
            # --- VALID ---
            valid_test("basic")
            .description(
                "Elementwise Max(x, b) with identical shapes (no broadcasting)."
            )
            .override_input_shapes(x=[1, 3, 4, 4])
            .override_input_dtypes(x=np.int64)  # make x INT64 to match b
            .tags("basic", "max", "elementwise")
            .build(),
            e2e_test("e2e_basic")
            .description("End-to-end test for Max with equal-shaped inputs.")
            .override_input_shapes(x=[1, 3, 4, 4])
            .override_input_dtypes(x=np.int64)
            .override_output_shapes(max_output=[1, 3, 4, 4])
            .tags("e2e", "max", "elementwise")
            .build(),
            # --- ERRORS ---
            error_test("mismatched_initializer_shape")
            .description(
                "Initializer 'b' has a different shape: no broadcasting allowed."
            )
            .override_input_shapes(x=[1, 3, 4, 4])
            .override_input_dtypes(x=np.int64)
            .override_initializer("b", np.ones((1, 3, 5, 4), dtype=np.int64))
            .expects_error(InvalidParamError, "Broadcasting is not supported for Max")
            .tags("error", "shape", "no-broadcast")
            .build(),
            error_test("initializer_wrong_dtype")
            .description("Initializer 'b' is float32; checker should insist on INT64.")
            .override_input_shapes(x=[1, 3, 4, 4])
            .override_input_dtypes(x=np.int64)
            .override_initializer("b", np.ones((1, 3, 4, 4), dtype=np.float32))
            .expects_error(InvalidParamError, "expects INT64 initializers")
            .tags("error", "dtype")
            .build(),
        ]
