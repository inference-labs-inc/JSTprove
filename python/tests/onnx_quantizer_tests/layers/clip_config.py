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
    """Test configuration provider for elementwise Clip."""

    @property
    def layer_name(self) -> str:
        return "Clip"

    def get_config(self) -> LayerTestConfig:
        # Treat min / max as optional extra inputs, scalar-shaped by default.
        # Scalars are encoded as shape [1] for the test harness; ONNX/ORT
        # will still broadcast them over A.
        return LayerTestConfig(
            op_type="Clip",
            valid_inputs=["A", "min", "max"],
            valid_attributes={},  # no Clip-specific attrs
            required_initializers={},  # by default, all three can be dynamic inputs
            input_shapes={
                "A": [1, 3, 4, 4],
                "min": [1],  # scalar-ish bound
                "max": [1],  # scalar-ish bound
            },
            output_shapes={
                "clip_output": [1, 3, 4, 4],
            },
        )

    def get_test_specs(self) -> list:
        rng = np.random.default_rng(TEST_RNG_SEED)

        return [
            # --- VALID TESTS ---
            # Basic Clip with scalar min/max as dynamic inputs.
            valid_test("basic_scalar_bounds")
            .description("Clip with A, min, max all as inputs; min/max are scalars.")
            .override_input_shapes(A=[1, 3, 4, 4], min=[1], max=[1])
            .override_output_shapes(clip_output=[1, 3, 4, 4])
            .tags("basic", "elementwise", "clip")
            .build(),
            # This keeps the name used by the integration tests:
            # Clip_broadcast_bounds
            # Broadcasting here is just scalar → full tensor broadcast.
            valid_test("broadcast_bounds")
            .description(
                "Clip with scalar bounds broadcast over all elements of A "
                "(mirrors Max/Min broadcast tests but respects ORT's scalar bound "
                "rules).",
            )
            .override_input_shapes(A=[1, 3, 2, 4], min=[1], max=[1])
            .override_output_shapes(clip_output=[1, 3, 2, 4])
            .tags("broadcast", "elementwise", "clip", "onnxruntime")
            .build(),
            # This keeps the name used by the integration tests:
            # Clip_initializer_bounds
            valid_test("initializer_bounds")
            .description(
                "Clip where min/max are scalar initializers instead of inputs.",
            )
            .override_input_shapes(A=[1, 3, 4, 4])  # only A is a true input
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
            e2e_test("e2e_small").description(
                "End-to-end Clip with small random tensor and scalar bounds.",
            )
            # All three are treated as runtime inputs here;
            # min/max are scalar-shaped [1].
            .override_input_shapes(
                A=[1, 3, 4, 4],
                min=[1],
                max=[1],
            )
            .override_output_shapes(clip_output=[1, 3, 4, 4])
            .tags("e2e", "clip")
            .build(),
            e2e_test("e2e_initializer_bounds").description(
                "End-to-end Clip where min/max are scalar initializers "
                "instead of inputs.",
            )
            # Only A is a true runtime input; min/max are scalar initializers.
            .override_input_shapes(
                A=[1, 3, 4, 4],
            )
            .override_initializer(
                "min",
                np.array(rng.uniform(-1.0, 0.0), dtype=np.float64),
            )
            .override_initializer(
                "max",
                np.array(rng.uniform(0.0, 2.0), dtype=np.float64),
            )
            .override_output_shapes(clip_output=[1, 3, 4, 4])
            .tags("e2e", "initializer", "clip")
            .build(),
            # --- EDGE / STRESS ---
            edge_case_test("empty_tensor")
            .description("Clip with empty tensor input and scalar bounds.")
            .override_input_shapes(A=[0], min=[1], max=[1])
            .override_output_shapes(clip_output=[0])
            .tags("edge", "empty", "clip")
            .build(),
        ]
