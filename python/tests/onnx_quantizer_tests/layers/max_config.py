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
        # One dynamic tensor input "x" and one initializer "b" so the checker can read shapes.
        init = numpy_helper.from_array(np.ones((1, 3, 4, 4), dtype=np.int64), name="b")
        return LayerTestConfig(
            op_type="Max",
            valid_inputs=["x", "b"],
            valid_attributes={},  # Max has no attrs
            required_initializers={"b": init},
        )

    def get_test_specs(self) -> list:
        return [
            # --- VALID ---
            valid_test("basic")
            .description(
                "Elementwise Max(x, b) with identical shapes (no broadcasting)."
            )
            .override_input_shapes(x=[1, 3, 4, 4])
            .tags("basic", "max", "elementwise")
            .build(),
            e2e_test("e2e_basic")
            .description("End-to-end test for Max with equal-shaped inputs.")
            .override_input_shapes(x=[1, 3, 4, 4])
            .override_output_shapes(max_output=[1, 3, 4, 4])
            .tags("e2e", "max", "elementwise")
            .build(),
            # --- ERROR: no broadcasting allowed ---
            error_test("mismatched_initializer_shape")
            .description(
                "Initializer 'b' has a different shape; checker should reject."
            )
            .override_input_shapes(x=[1, 3, 4, 4])
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
