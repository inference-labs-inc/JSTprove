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


class MaxConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for elementwise Max"""

    @property
    def layer_name(self) -> str:
        return "Max"

    def get_config(self) -> LayerTestConfig:
        return LayerTestConfig(
            op_type="Max",
            valid_inputs=["A", "B"],
            valid_attributes={},  # Max has no layer-specific attributes
            required_initializers={},  # default: both A and B are dynamic inputs
            input_shapes={
                "A": [1, 3, 4, 4],
                "B": [1, 3, 4, 4],
            },
            output_shapes={
                "max_output": [1, 3, 4, 4],
            },
        )

    def get_test_specs(self) -> list:
        rng = np.random.default_rng(TEST_RNG_SEED)
        return [
            # --- VALID TESTS ---
            valid_test("basic")
            .description("Basic elementwise Max of two same-shaped tensors")
            .override_input_shapes(A=[1, 3, 4, 4], B=[1, 3, 4, 4])
            .tags("basic", "elementwise", "max")
            .build(),
            valid_test("broadcast_max")
            .description("Max with Numpy-style broadcasting along spatial dimensions")
            .override_input_shapes(A=[1, 3, 4, 4], B=[1, 3, 1, 1])
            .tags("broadcast", "elementwise", "max", "onnx14")
            .build(),
            valid_test("initializer_max")
            .description("Max where B is an initializer instead of an input")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer(
                "B",
                rng.normal(0, 1, (1, 3, 4, 4)).astype(np.float32),
            )
            .tags("initializer", "elementwise", "max", "onnxruntime")
            .build(),
            # --- E2E TESTS ---
            e2e_test("e2e_max")
            .description("End-to-end Max test with random inputs")
            .override_input_shapes(A=[1, 3, 4, 4], B=[1, 3, 4, 4])
            .override_output_shapes(max_output=[1, 3, 4, 4])
            .tags("e2e", "max", "2d")
            .build(),
            e2e_test("e2e_broadcast_max")
            .description(
                "End-to-end Max with Numpy-style broadcasting along spatial dimensions",
            )
            .override_input_shapes(A=[1, 3, 4, 4], B=[1, 3, 1, 1])
            .override_output_shapes(max_output=[1, 3, 4, 4])
            .tags("e2e", "broadcast", "elementwise", "max", "onnx14")
            .build(),
            e2e_test("e2e_initializer_max")
            .description("End-to-end Max where B is an initializer")
            .override_input_shapes(A=[1, 3, 4, 4])
            .override_initializer(
                "B",
                rng.normal(0, 1, (1, 3, 4, 4)).astype(np.float32),
            )
            .override_output_shapes(max_output=[1, 3, 4, 4])
            .tags("e2e", "initializer", "elementwise", "max", "onnxruntime")
            .build(),
            # --- EDGE / STRESS ---
            edge_case_test("empty_tensor")
            .description("Max with empty tensor input (zero elements)")
            .override_input_shapes(A=[0], B=[0])
            .override_output_shapes(max_output=[0])
            .tags("edge", "empty", "max")
            .build(),
            valid_test("large_tensor")
            .description("Large tensor max performance/stress test")
            .override_input_shapes(A=[1, 64, 256, 256], B=[1, 64, 256, 256])
            .tags("large", "performance", "max")
            .skip("Performance test, skipped by default")
            .build(),
        ]
