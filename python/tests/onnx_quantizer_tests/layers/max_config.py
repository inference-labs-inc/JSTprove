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
        # Provide a NumPy array (NOT a TensorProto). The test harness will cast/build tensors.
        init = np.ones((1, 3, 4, 4), dtype=np.float32)
        return LayerTestConfig(
            op_type="Max",
            valid_inputs=["x", "b"],
            valid_attributes={},  # Max has no attrs
            required_initializers={"b": init},  # NumPy array here
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
            # --- ERROR: no broadcasting allowed ---
            error_test("mismatched_initializer_shape").description(
                "Both inputs are initializers with different shapes; checker should reject."
            )
            # Make x an initializer too (shape matches the VALID case)
            .override_initializer("x", np.ones((1, 3, 4, 4), dtype=np.float32))
            # And make b a different shape so the checker sees a mismatch
            .override_initializer("b", np.ones((1, 3, 5, 4), dtype=np.float32))
            .expects_error(InvalidParamError, "Broadcasting is not supported for Max")
            .tags("error", "shape", "no-broadcast")
            .build(),
        ]
