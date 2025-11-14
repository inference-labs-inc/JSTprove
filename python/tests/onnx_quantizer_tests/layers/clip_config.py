# python/tests/onnx_quantizer_tests/layers/clip_config.py
from __future__ import annotations

import numpy as np

from python.tests.onnx_quantizer_tests import TEST_RNG_SEED
from python.tests.onnx_quantizer_tests.layers.base import (
    e2e_test,
    edge_case_test,
    valid_test,
)
from python.tests.onnx_quantizer_tests.layers.factory import (
    BaseLayerConfigProvider,
    LayerTestConfig,
)


class ClipConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for ONNX Clip (elementwise)."""

    @property
    def layer_name(self) -> str:
        return "Clip"

    def get_config(self) -> LayerTestConfig:
        # We exercise the 3-input form: X, min, max.
        # onnxruntime's CPU Clip kernel requires min/max to be scalar tensors.
        return LayerTestConfig(
            op_type="Clip",
            valid_inputs=["A", "min", "max"],
            valid_attributes={},  # no Clip attrs used here
            required_initializers={},  # by default everything is a dynamic input
            input_shapes={
                "A": [1, 3, 4, 4],
                # true scalars (shape = []) so that min/max pass IsScalar()
                "min": [],
                "max": [],
            },
            output_shapes={
                "clip_output": [1, 3, 4, 4],
            },
        )

    def get_test_specs(self) -> list:
        rng = np.random.default_rng(TEST_RNG_SEED)

        return [
            # --- VALID TESTS ---
            # Scalar bounds supplied as *inputs* (min/max are separate scalar inputs).
            valid_test("broadcast_bounds")
            .description(
                "Clip with scalar bounds supplied as inputs; scalars broadcast over A."
            )
            .override_input_shapes(A=[1, 3, 2, 4], min=[], max=[])
            .override_output_shapes(clip_output=[1, 3, 2, 4])
            .tags("broadcast", "elementwise", "clip", "onnxruntime")
            .build(),
            # Scalar bounds supplied as *initializers* (min/max are 0-D tensors).
            valid_test("initializer_bounds")
            .description(
                "Clip where min/max are scalar initializers instead of dynamic inputs."
            )
            .override_input_shapes(A=[1, 3, 4, 4])  # only A is a real input
            # Scalar numpy values → ONNX initializers with shape ()
            .override_initializer(
                "min",
                np.array(rng.uniform(-1.0, 0.0), dtype=np.float64),
            )
            .override_initializer(
                "max",
                np.array(rng.uniform(0.0, 2.0), dtype=np.float64),
            )
            .override_output_shapes(clip_output=[1, 3, 4, 4])
            .tags("initializer", "elementwise", "clip", "onnxruntime")
            .build(),
            # --- E2E TESTS ---
            e2e_test("e2e_small")
            .description("End-to-end Clip with small random tensor and scalar bounds.")
            .override_input_shapes(A=[1, 3, 4, 4], min=[], max=[])
            .override_output_shapes(clip_output=[1, 3, 4, 4])
            .tags("e2e", "clip")
            .build(),
            # --- EDGE / STRESS ---
            edge_case_test("empty_tensor")
            .description("Clip with empty tensor input and scalar bounds.")
            .override_input_shapes(A=[0], min=[], max=[])
            .tags("edge", "empty", "clip")
            .build(),
        ]
