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
    """Test configuration provider for elementwise Clip"""

    @property
    def layer_name(self) -> str:
        return "Clip"

    def get_config(self) -> LayerTestConfig:
        """
        Treat Clip as a 3-input elementwise op:
            A  : data tensor
            min: lower bound
            max: upper bound

        ONNX allows min/max to be optional; here we focus on the common
        3-input form so we exercise the quantizer on all inputs.
        """
        return LayerTestConfig(
            op_type="Clip",
            valid_inputs=["A", "min", "max"],
            valid_attributes={},  # Clip has no layer-specific attributes
            required_initializers={},  # defaults: all three can be dynamic inputs
            input_shapes={
                "A": [1, 3, 4, 4],
                "min": [1],  # scalar bounds (broadcasted)
                "max": [1],
            },
            output_shapes={
                "clip_output": [1, 3, 4, 4],
            },
        )

    def get_test_specs(self) -> list:
        rng = np.random.default_rng(TEST_RNG_SEED)

        # Helper to build a pair of compatible (min, max) tensors where max > min pointwise.
        def make_min_max(shape):
            base = rng.normal(loc=0.0, scale=1.0, size=shape)
            # Ensure max is strictly greater than min everywhere
            offset = np.abs(rng.normal(loc=0.5, scale=0.5, size=shape)) + 0.1
            min_val = base.astype(np.float64)
            max_val = (base + offset).astype(np.float64)
            return min_val, max_val

        # Scalar bounds (1-element) for “basic” and “e2e_scalar_bounds”
        min_scalar, max_scalar = make_min_max((1,))

        # Broadcasted per-channel bounds: shape [1, 3, 1, 1] to match A=[1,3,4,4]
        min_broadcast, max_broadcast = make_min_max((1, 3, 1, 1))

        # Initializer bounds with full spatial shape, just to exercise initializer path
        min_full, max_full = make_min_max((1, 3, 4, 4))

        return [
            # --- VALID TESTS ---
            valid_test("basic_scalar_bounds")
            .description(
                "Basic Clip with scalar bounds (three dynamic inputs: A, min, max)"
            )
            .override_input_shapes(A=[1, 3, 4, 4], min=[1], max=[1])
            .override_initializer("A", rng.normal(0.0, 1.0, (1, 3, 4, 4)))
            .override_initializer("min", min_scalar)
            .override_initializer("max", max_scalar)
            .tags("basic", "elementwise", "clip")
            .build(),
            valid_test("broadcast_bounds")
            .description(
                "Clip with broadcasted per-channel bounds; "
                "min/max have shape [1, 3, 1, 1] broadcasting over H,W"
            )
            .override_input_shapes(A=[1, 3, 4, 4], min=[1, 3, 1, 1], max=[1, 3, 1, 1])
            .override_initializer("A", rng.normal(0.0, 1.0, (1, 3, 4, 4)))
            .override_initializer("min", min_broadcast)
            .override_initializer("max", max_broadcast)
            .tags("broadcast", "elementwise", "clip", "onnx14")
            .build(),
            valid_test("initializer_bounds")
            .description(
                "Clip where min and max are full-tensor initializers (no dynamic bounds)"
            )
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("A", rng.normal(0.0, 1.0, (1, 3, 4, 4)))
            .override_initializer("min", min_full)
            .override_initializer("max", max_full)
            .tags("initializer", "elementwise", "clip", "onnxruntime")
            .build(),
            # --- E2E TESTS ---
            e2e_test("e2e_scalar_bounds")
            .description(
                "End-to-end Clip test with scalar bounds; "
                "checks ORT(float) vs quantized int64 pipeline"
            )
            .override_input_shapes(A=[1, 3, 4, 4], min=[1], max=[1])
            .override_output_shapes(clip_output=[1, 3, 4, 4])
            .override_initializer("A", rng.normal(0.0, 1.0, (1, 3, 4, 4)))
            .override_initializer("min", min_scalar)
            .override_initializer("max", max_scalar)
            .tags("e2e", "clip", "scalar_bounds", "2d")
            .build(),
            e2e_test("e2e_broadcast_bounds")
            .description("End-to-end Clip test with broadcasted per-channel bounds")
            .override_input_shapes(A=[1, 3, 4, 4], min=[1, 3, 1, 1], max=[1, 3, 1, 1])
            .override_output_shapes(clip_output=[1, 3, 4, 4])
            .override_initializer("A", rng.normal(0.0, 1.0, (1, 3, 4, 4)))
            .override_initializer("min", min_broadcast)
            .override_initializer("max", max_broadcast)
            .tags("e2e", "clip", "broadcast", "2d")
            .build(),
            # --- EDGE / STRESS ---
            edge_case_test("empty_tensor")
            .description("Clip with empty tensor input (zero elements)")
            .override_input_shapes(A=[0], min=[1], max=[1])
            .tags("edge", "empty", "clip")
            .build(),
            valid_test("large_tensor")
            .description("Large tensor Clip performance/stress test")
            .override_input_shapes(A=[1, 64, 256, 256], min=[1], max=[1])
            .override_initializer("A", rng.normal(0.0, 1.0, (1, 64, 256, 256)))
            .override_initializer("min", np.array([0.0], dtype=np.float64))
            .override_initializer("max", np.array([6.0], dtype=np.float64))
            .tags("large", "performance", "clip")
            .skip("Performance test, skipped by default")
            .build(),
        ]
