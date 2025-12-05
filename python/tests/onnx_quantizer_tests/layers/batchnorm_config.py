import numpy as np

from python.tests.onnx_quantizer_tests import TEST_RNG_SEED
from python.tests.onnx_quantizer_tests.layers.base import (
    BaseLayerConfigProvider,
    LayerTestConfig,
    LayerTestSpec,
    e2e_test,
    valid_test,
)


class BatchNormConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for BatchNorm (ONNX BatchNormalization op)"""

    @property
    def layer_name(self) -> str:
        return "BatchNormalization"

    def get_config(self) -> LayerTestConfig:
        rng = np.random.default_rng(TEST_RNG_SEED)

        # default shapes: N x C x H x W
        default_input_shape = [1, 3, 4, 4]
        c = default_input_shape[1]

        # typical required initializers (scale, bias, mean, var) are length C
        return LayerTestConfig(
            op_type="BatchNormalization",
            valid_inputs=["X", "scale", "B", "input_mean", "input_var"],
            valid_attributes={
                "epsilon": 1e-5,
                "momentum": 0.9,
                "training_mode": 0,
            },
            required_initializers={
                # Defaults are stored as numpy arrays with shape (C,)
                "scale": rng.normal(1.0, 0.5, c).astype(np.float32),
                "B": rng.normal(0.0, 0.5, c).astype(np.float32),
                "input_mean": rng.normal(0.0, 1.0, c).astype(np.float32),
                "input_var": np.abs(rng.normal(1.0, 0.5, c)).astype(np.float32),
            },
            input_shapes={"X": default_input_shape},
            output_shapes={"batchnormalization_output": default_input_shape},
        )

    def get_test_specs(self) -> list[LayerTestSpec]:
        rng = np.random.default_rng(TEST_RNG_SEED)
        c = 3

        return [
            # Basic valid tests
            valid_test("basic_inference")
            .description("Basic BatchNormalization inference: standard shapes")
            .tags("basic", "inference", "batchnorm")
            .build(),
            valid_test("different_input_shape")
            .description("Inference with different spatial dims")
            .override_input_shapes(X=[2, c, 8, 8])
            .override_output_shapes(batchnormalization_output=[2, c, 8, 8])
            .tags("inference", "spatial")
            .build(),
            valid_test("epsilon_variation")
            .description("Inference with larger epsilon for numerical stability")
            .override_attrs(epsilon=1e-3)
            .tags("epsilon")
            .build(),
            valid_test("momentum_variation")
            .description(
                "Inference with non-default momentum (has no effect in inference mode)",
            )
            .override_attrs(momentum=0.5)
            .tags("momentum")
            .build(),
            valid_test("zero_mean_input")
            .description("Input with zero mean")
            .override_initializer("input_mean", np.zeros((c,), dtype=np.float32))
            .tags("edge", "zero_mean")
            .build(),
            # Scalar / broadcast style tests
            valid_test("per_channel_zero_variance")
            .description(
                "Edge case: very small variance values (clamped by epsilon), inference",
            )
            .override_initializer("input_var", np.full((c,), 1e-8, dtype=np.float32))
            .override_attrs(epsilon=1e-5)
            .tags("edge", "small_variance")
            .build(),
            # E2E tests that set explicit initializer values
            e2e_test("e2e_inference")
            .description("E2E inference test with explicit initializers")
            .override_input_shapes(X=[1, c, 2, 2])
            .override_output_shapes(batchnormalization_output=[1, c, 2, 2])
            .override_initializer("scale", rng.normal(1.0, 0.1, c).astype(np.float32))
            .override_initializer("B", rng.normal(0.0, 0.1, c).astype(np.float32))
            .override_initializer(
                "input_mean",
                rng.normal(0.0, 0.1, c).astype(np.float32),
            )
            .override_initializer(
                "input_var",
                np.abs(rng.normal(0.5, 0.2, c)).astype(np.float32),
            )
            .tags("e2e", "inference")
            .build(),
            e2e_test("e2e_inference_small_2x2")
            .description("E2E inference with small 2x2 spatial input")
            .override_input_shapes(X=[1, 3, 2, 2])
            .override_output_shapes(batchnormalization_output=[1, 3, 2, 2])
            .override_initializer("scale", np.array([1.0, 0.9, 1.1], dtype=np.float32))
            .override_initializer("B", np.array([0.0, 0.1, -0.1], dtype=np.float32))
            .override_initializer(
                "input_mean",
                np.array([0.5, -0.5, 0.0], dtype=np.float32),
            )
            .override_initializer(
                "input_var",
                np.array([0.25, 0.5, 0.1], dtype=np.float32),
            )
            .tags("e2e", "small", "2x2")
            .build(),
            e2e_test("e2e_inference_wide_input")
            .description("E2E inference with wider input shape (C=4, H=2, W=8)")
            .override_input_shapes(X=[2, 4, 2, 8])
            .override_output_shapes(batchnormalization_output=[2, 4, 2, 8])
            .override_initializer(
                "scale",
                np.array([1.0, 0.8, 1.2, 0.9], dtype=np.float32),
            )
            .override_initializer(
                "B",
                np.array([0.0, 0.1, -0.1, 0.05], dtype=np.float32),
            )
            .override_initializer(
                "input_mean",
                np.array([0.0, 0.5, -0.5, 0.2], dtype=np.float32),
            )
            .override_initializer(
                "input_var",
                np.array([1.0, 0.5, 0.25, 0.1], dtype=np.float32),
            )
            .tags("e2e", "wide", "C4")
            .build(),
            e2e_test("e2e_inference_batch2_channels3")
            .description("E2E inference with batch size 2 and 3 channels")
            .override_input_shapes(X=[2, 3, 4, 4])
            .override_output_shapes(batchnormalization_output=[2, 3, 4, 4])
            .override_initializer("scale", np.array([0.5, 1.0, 1.5], dtype=np.float32))
            .override_initializer("B", np.array([0.0, 0.0, 0.0], dtype=np.float32))
            .override_initializer(
                "input_mean",
                np.array([-0.5, 0.0, 0.5], dtype=np.float32),
            )
            .override_initializer(
                "input_var",
                np.array([0.2, 0.5, 0.8], dtype=np.float32),
            )
            .tags("e2e", "batch2", "C3")
            .build(),
            e2e_test("e2e_inference_high_epsilon")
            .description("E2E inference with high epsilon for numerical stability")
            .override_input_shapes(X=[1, 2, 4, 4])
            .override_output_shapes(batchnormalization_output=[1, 2, 4, 4])
            .override_initializer("scale", np.array([1.0, 1.0], dtype=np.float32))
            .override_initializer("B", np.array([0.1, -0.1], dtype=np.float32))
            .override_initializer("input_mean", np.array([0.0, 0.5], dtype=np.float32))
            .override_initializer(
                "input_var",
                np.array([0.0, 0.0], dtype=np.float32),
            )  # tiny variance
            .override_attrs(epsilon=1e-2)
            .tags("e2e", "high_epsilon", "numerical_stability")
            .build(),
            e2e_test("e2e_inference_non_square")
            .description("E2E inference with non-square spatial dimensions")
            .override_input_shapes(X=[1, 3, 2, 5])
            .override_output_shapes(batchnormalization_output=[1, 3, 2, 5])
            .override_initializer("scale", np.array([1.0, 0.9, 1.1], dtype=np.float32))
            .override_initializer("B", np.array([0.0, 0.1, -0.1], dtype=np.float32))
            .override_initializer(
                "input_mean",
                np.array([0.1, -0.1, 0.0], dtype=np.float32),
            )
            .override_initializer(
                "input_var",
                np.array([0.5, 0.25, 0.75], dtype=np.float32),
            )
            .tags("e2e", "non_square", "C3")
            .build(),
        ]
