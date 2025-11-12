# python/tests/onnx_quantizer_tests/layers/min_config.py
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


class MinConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for elementwise Min"""

    @property
    def layer_name(self) -> str:
        return "Min"

    def get_config(self) -> LayerTestConfig:
        # Elementwise, 2 inputs → 1 output, no attributes
        return LayerTestConfig(
            op_type="Min",
            valid_inputs=["X", "Y"],
            valid_attributes={},
            required_initializers={},
            input_shapes={
                "X": [1, 3, 4, 4],
                "Y": [1, 3, 4, 4],
            },
            output_shapes={
                "min_output": [1, 3, 4, 4],
            },
        )

    def get_test_specs(self) -> list:
        rng = np.random.default_rng(TEST_RNG_SEED)
        return [
            # Basic: same-shaped tensors
            valid_test("basic")
            .description("Elementwise Min(X, Y) with identical shapes")
            .override_input_shapes(X=[1, 3, 4, 4], Y=[1, 3, 4, 4])
            .tags("basic", "elementwise", "min")
            .build(),
            # Broadcasting allowed (mirror Add/Max)
            valid_test("broadcast_min")
            .description("Min with NumPy-style broadcasting along spatial dims")
            .override_input_shapes(X=[1, 3, 4, 4], Y=[1, 3, 1, 1])
            .tags("broadcast", "elementwise", "min", "onnx14")
            .build(),
            # One input can be initializer
            valid_test("initializer_min")
            .description("Min where Y is an initializer")
            .override_input_shapes(X=[1, 3, 4, 4])
            .override_initializer("Y", rng.normal(0, 1, (1, 3, 4, 4)))
            .tags("initializer", "elementwise", "min", "onnxruntime")
            .build(),
            # End-to-end run
            e2e_test("e2e_min")
            .description("End-to-end Min test with random inputs")
            .override_input_shapes(X=[1, 3, 4, 4], Y=[1, 3, 4, 4])
            .override_output_shapes(min_output=[1, 3, 4, 4])
            .tags("e2e", "min", "2d")
            .build(),
            # Edge example (optional)
            edge_case_test("empty_tensor")
            .description("Min with empty tensor inputs")
            .override_input_shapes(X=[0], Y=[0])
            .tags("edge", "empty", "min")
            .build(),
        ]
