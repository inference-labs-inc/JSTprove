# python/tests/onnx_quantizer_tests/layers/clip_config.py
from __future__ import annotations

import numpy as np

from .base import (
    LayerTestConfig,
    Spec,
    expects_error,
)  # same helpers your other configs use


def make_configs() -> list[LayerTestConfig]:
    """
    Config & specs for ONNX Clip (elementwise, broadcasting OK).
    Mirrors the style of max_config.py/min_config.py for reviewer parity.
    """
    cfg = LayerTestConfig(
        op_type="Clip",
        valid_inputs=["A"],  # ONNX Clip can have 1 to 3 inputs: X[, min][, max]
        output_name="clip_output",
        default_shapes={
            "A": (1, 2, 3),
        },
        # No required attributes. min/max may be inputs or attributes; we test input form.
    )

    specs: list[Spec] = []

    # VALID: basic clip with scalar bounds
    specs.append(
        cfg.make_valid_spec(
            name="basic_scalar_bounds",
            inputs={
                "A": np.array([[-2, -1, 0], [1, 2, 3]])
                .reshape(1, 2, 3)
                .astype(np.float64),
                "min": np.array([0.0], dtype=np.float64),
                "max": np.array([2.0], dtype=np.float64),
            },
            # Expected behavior is checked by the integration harness comparing
            # ORT(float) → quantized → int64 pipeline vs. reference.
            extra_inputs_order=["min", "max"],
        )
    )

    # VALID: broadcasted per-axis min/max
    specs.append(
        cfg.make_valid_spec(
            name="broadcast_bounds",
            inputs={
                "A": np.array(
                    [[-5, -2, 1], [4, 9, 10]],
                    dtype=np.float64,
                ).reshape(1, 2, 3),
                "min": np.array(
                    [[0.0, -1.0, 0.0]], dtype=np.float64
                ),  # shape (1, 3) → broadcasts
                "max": np.array([[3.0, 5.0, 8.0]], dtype=np.float64),
            },
            extra_inputs_order=["min", "max"],
        )
    )

    # E2E: small randomized tensor with scalar bounds (kept tiny for speed)
    rng = np.random.default_rng(123)
    A = (rng.standard_normal((1, 2, 3)) * 2.0).astype(np.float64)
    specs.append(
        cfg.make_e2e_spec(
            name="e2e_small",
            inputs={
                "A": A,
                "min": np.array([0.0], dtype=np.float64),
                "max": np.array([1.5], dtype=np.float64),
            },
            extra_inputs_order=["min", "max"],
        )
    )

    # ERROR: (example) swap bound order – still valid per ONNX (min > max is undefined for some runtimes),
    # but our pipeline defers to runtime semantics; keep no “expects_error” here to avoid false negatives.

    return [cfg.with_specs(specs)]
