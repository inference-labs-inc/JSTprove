# python/tests/onnx_quantizer_tests/layers/max_config.py
from __future__ import annotations

import numpy as np
from onnx import numpy_helper

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
        # Base config: two inputs x, b; we’ll turn x into an initializer in the tests
        # so both inputs are known int64 tensors (no dtype mismatch, shapes checkable).
        init_b = numpy_helper.from_array(
            np.ones((1, 3, 4, 4), dtype=np.int64), name="b"
        )
        return LayerTestConfig(
            op_type="Max",
            valid_inputs=["x", "b"],
            valid_attributes={},  # Max has no attributes
            required_initializers={"b": init_b},
        )

    def get_test_specs(self) -> list:
        # Common int64 initializer for x with the "correct" shape.
        x_ok = numpy_helper.from_array(
            np.arange(1 * 3 * 4 * 4, dtype=np.int64).reshape(1, 3, 4, 4),
            name="x",
        )

        return [
            # --- VALID ---
            valid_test("basic").description(
                "Elementwise Max(x, b) with identical shapes (no broadcasting)."
            )
            # Make *both* inputs initializers so shapes are known and dtypes match.
            .override_initializer("x", x_ok)
            .tags("basic", "max", "elementwise")
            .build(),
            e2e_test("e2e_basic")
            .description("End-to-end test for Max with equal-shaped int64 inputs.")
            .override_initializer("x", x_ok)
            .override_output_shapes(max_output=[1, 3, 4, 4])
            .tags("e2e", "max", "elementwise")
            .build(),
            # --- ERROR: mismatched shapes (no broadcasting allowed) ---
            error_test("mismatched_initializer_shape")
            .description(
                "Initializer 'b' has a different shape; checker should reject (no broadcasting)."
            )
            .override_initializer("x", x_ok)
            .override_initializer(
                "b",
                numpy_helper.from_array(
                    np.ones((1, 3, 5, 4), dtype=np.int64), name="b"
                ),
            )
            .expects_error(InvalidParamError, "Broadcasting is not supported for Max")
            .tags("error", "shape", "no-broadcast")
            .build(),
        ]
