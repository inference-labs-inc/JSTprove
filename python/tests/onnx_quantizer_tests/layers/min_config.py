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
        return LayerTestConfig(
            op_type="Min",
            valid_inputs=["A", "B"],
            valid_attributes={},  # Min has no layer-specific attributes
            required_initializers={},  # default: both A and B are dynamic inputs
            input_shapes={
                "A": [1, 3, 4, 4],
                "B": [1, 3, 4, 4],
            },
            output_shapes={
                "min_output": [1, 3, 4, 4],
            },
        )

    def get_test_specs(self) -> list:
        rng = np.random.default_rng(TEST_RNG_SEED)
        return [
            # --- VALID TESTS ---
            valid_test("basic")
            .description("Basic elementwise Min of two same-shaped tensors")
            .override_input_shapes(A=[1, 3, 4, 4], B=[1, 3, 4, 4])
            .tags("basic", "elementwise", "min")
            .build(),
            valid_test("broadcast_min")
            .description("Min with Numpy-style broadcasting along spatial dimensions")
            .override_input_shapes(A=[1, 3, 4, 4], B=[1, 3, 1, 1])
            .tags("broadcast", "elementwise", "min", "onnx14")
            .build(),
            valid_test("initializer_min")
            .description("Min where B is an initializer instead of an input")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", rng.normal(0, 1, (1, 3, 4, 4)))
            .tags("initializer", "elementwise", "min", "onnxruntime")
            .build(),
            # --- E2E TESTS ---
            e2e_test("e2e_min")
            .description("End-to-end Min test with random inputs")
            .override_input_shapes(A=[1, 3, 4, 4], B=[1, 3, 4, 4])
            .override_output_shapes(min_output=[1, 3, 4, 4])
            .tags("e2e", "min", "2d")
            .build(),
            e2e_test("e2e_broadcast_min")
            .description(
                "End-to-end Min with Numpy-style broadcasting along spatial dimensions",
            )
            .override_input_shapes(A=[1, 3, 4, 4], B=[1, 3, 1, 1])
            .override_output_shapes(min_output=[1, 3, 4, 4])
            .tags("e2e", "broadcast", "elementwise", "min", "onnx14")
            .build(),
            e2e_test("e2e_initializer_min")
            .description("End-to-end Min where B is an initializer")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer("B", rng.normal(0, 1, (1, 3, 4, 4)))
            .override_output_shapes(min_output=[1, 3, 4, 4])
            .tags("e2e", "initializer", "elementwise", "min", "onnxruntime")
            .build(),
            # --- EDGE / STRESS ---
            edge_case_test("empty_tensor")
            .description("Min with empty tensor input (zero elements)")
            .override_input_shapes(A=[0], B=[0])
            .override_output_shapes(min_output=[0])
            .tags("edge", "empty", "min")
            .build(),
            valid_test("large_tensor")
            .description("Large tensor min performance/stress test")
            .override_input_shapes(A=[1, 64, 256, 256], B=[1, 64, 256, 256])
            .tags("large", "performance", "min")
            .skip("Performance test, skipped by default")
            .build(),
        ]
