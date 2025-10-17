import numpy as np

from python.core.model_processing.onnx_quantizer.exceptions import InvalidParamError
from python.tests.onnx_quantizer_tests import TEST_RNG_SEED
from python.tests.onnx_quantizer_tests.layers.base import (
    LayerTestSpec,
    e2e_test,
    error_test,
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
            # --- VALID TESTS ---
            valid_test("basic")
            .description("Basic Gemm operation (no transposes, alpha=1, beta=1)")
            .tags("basic")
            .build(),
            valid_test("transposed_weights")
            .description("Gemm with transposed weight matrix (transB=1)")
            .override_attrs(transB=1)
            .override_initializer(
                "gemm_weight",
                rng.normal(0, 1, (256, 128)),
            )  # Transposed shape
            .tags("transpose", "transB")
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
            # --- ERROR TESTS ---
            # Add check on weights matrix in check_supported
            error_test("invalid_alpha_type")
            .description("Invalid alpha type (should be numeric)")
            .override_attrs(alpha=-1.0)
            .expects_error(
                InvalidParamError,
                "alpha value of -1.0 not supported [Attribute: alpha] [Expected: 1.0]",
            )
            .tags("invalid_param", "alpha")
            .build(),
            error_test("no_bias")
            .description("Gemm without bias term (beta=0 should ignore bias)")
            .override_inputs("input", "gemm_weight")
            .override_attrs(beta=0.0)
            .expects_error(InvalidParamError, match="3 inputs")
            .tags("no_bias")
            .build(),
            error_test("different_alpha_beta")
            .description("Gemm with different alpha and beta scaling factors")
            .override_attrs(alpha=0.5, beta=2.0)
            .expects_error(
                InvalidParamError,
                "alpha value of 0.5 not supported [Attribute: alpha] [Expected: 1.0]",
            )
            .tags("scaling", "alpha_beta")
            .build(),
            # --- EDGE CASE / SKIPPED TESTS ---
            valid_test("large_matrix")
            .description("Large matrix multiplication performance test")
            .override_initializer("gemm_weight", rng.normal(0, 1, (1024, 2048)))
            .override_initializer("gemm_bias", rng.normal(0, 1, (1, 2048)))
            .override_input_shapes(input=[1, 1024])
            .override_output_shapes(gemm_output=[1, 2048])
            .tags("large", "performance")
            .skip("Performance test, not run by default")
            .build(),
        ]
