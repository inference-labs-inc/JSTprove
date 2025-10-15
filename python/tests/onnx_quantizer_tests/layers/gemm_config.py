import numpy as np

from python.tests.onnx_quantizer_tests import TEST_RNG_SEED
from python.tests.onnx_quantizer_tests.layers.base import (
    LayerTestSpec,
    e2e_test,
    valid_test,
)
from python.tests.onnx_quantizer_tests.layers.factory import (
    BaseLayerConfigProvider,
    LayerTestConfig,
)


class GemmConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for Gemm layers"""

    @property
    def layer_name(self) -> str:
        return "Gemm"

    def get_config(self) -> LayerTestConfig:
        rng = np.random.default_rng(TEST_RNG_SEED)
        return LayerTestConfig(
            op_type="Gemm",
            valid_inputs=["input", "gemm_weight", "gemm_bias"],
            valid_attributes={"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0},
            required_initializers={
                "gemm_weight": rng.normal(0, 1, (128, 256)),
                "gemm_bias": rng.normal(0, 1, (1, 256)),
            },
            input_shapes={"input": [1, 128]},  # Match weight input dimension K=128
            output_shapes={
                "gemm_output": [1, 256],
            },  # Match weight output dimension N=256
        )

    def get_test_specs(self) -> list[LayerTestSpec]:
        """Return test specifications for Gemm layers"""
        rng = np.random.default_rng(TEST_RNG_SEED)
        return [
            valid_test("basic")
            .description("Basic 2D convolution")
            .tags("basic", "2d")
            .build(),
            e2e_test("e2e_basic")
            .description("End-to-end test for basic Gemm layer")
            .override_attrs(alpha=1.0, beta=1.0, transA=0, transB=0)
            .override_input_shapes(input=[1, 4])
            .override_output_shapes(gemm_output=[1, 8])
            .override_initializer("gemm_weight", rng.normal(0, 1, (4, 8)))
            .override_initializer("gemm_bias", rng.normal(0, 1, (1, 8)))
            .tags("e2e", "basic")
            .build(),
        ]
