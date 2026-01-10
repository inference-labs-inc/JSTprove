import numpy as np

from python.tests.onnx_quantizer_tests import TEST_RNG_SEED
from python.tests.onnx_quantizer_tests.layers.base import (
    BaseLayerConfigProvider,
    LayerTestConfig,
    LayerTestSpec,
    e2e_test,
    edge_case_test,
    valid_test,
)


class TransposeConfigProvider(BaseLayerConfigProvider):
    """
    Test configuration provider for ONNX Transpose

    Note, this implementation assumes that the perm parameter
    fits with the input shape
    """

    @property
    def layer_name(self) -> str:
        return "Transpose"

    def get_config(self) -> LayerTestConfig:
        _rng = np.random.default_rng(TEST_RNG_SEED)

        # Default: simple 3D tensor
        default_input_shape = [2, 3, 4]

        return LayerTestConfig(
            op_type="Transpose",
            valid_inputs=["data"],
            valid_attributes={
                # default perm omitted → reverse dimensions
            },
            required_initializers={},
            input_shapes={"data": default_input_shape},
            output_shapes={"transpose_output": list(reversed(default_input_shape))},
        )

    def get_test_specs(self) -> list[LayerTestSpec]:
        _rng = np.random.default_rng(TEST_RNG_SEED)

        return [
            # ---------------------------------------------------------
            # Basic valid tests
            # ---------------------------------------------------------
            valid_test("default_reverse_3d")
            .description("Default transpose reverses dimensions for 3D input")
            .tags("basic", "default_perm", "3d")
            .build(),
            valid_test("explicit_perm_identity")
            .description("Explicit identity permutation leaves tensor unchanged")
            .override_attrs(perm=[0, 1, 2])
            .override_output_shapes(transpose_output=[2, 3, 4])
            .tags("basic", "identity")
            .build(),
            valid_test("explicit_perm_swap_01")
            .description("Swap first two dimensions of 3D tensor")
            .override_attrs(perm=[1, 0, 2])
            .override_output_shapes(transpose_output=[3, 2, 4])
            .tags("basic", "perm", "swap")
            .build(),
            valid_test("explicit_perm_rotate")
            .description("Rotate dimensions: (1, 2, 0)")
            .override_attrs(perm=[1, 2, 0])
            .override_output_shapes(transpose_output=[3, 4, 2])
            .tags("perm", "rotate")
            .build(),
            # ---------------------------------------------------------
            # Rank variations
            # ---------------------------------------------------------
            valid_test("transpose_2d")
            .description("2D transpose swaps rows and columns")
            .override_input_shapes(data=[5, 7])
            .override_attrs(perm=[1, 0])
            .override_output_shapes(transpose_output=[7, 5])
            .tags("2d", "basic")
            .build(),
            valid_test("transpose_4d_nchw_to_nhwc")
            .description("Transpose 4D tensor from NCHW to NHWC")
            .override_input_shapes(data=[1, 3, 8, 8])
            .override_attrs(perm=[0, 2, 3, 1])
            .override_output_shapes(transpose_output=[1, 8, 8, 3])
            .tags("4d", "layout", "nchw_to_nhwc")
            .build(),
            valid_test("transpose_5d")
            .description("5D tensor transpose with arbitrary permutation")
            .override_input_shapes(data=[2, 3, 4, 5, 6])
            .override_attrs(perm=[4, 3, 2, 1, 0])
            .override_output_shapes(transpose_output=[6, 5, 4, 3, 2])
            .tags("5d", "high_rank")
            .build(),
            # ---------------------------------------------------------
            # Edge cases
            # ---------------------------------------------------------
            edge_case_test("scalar_input")
            .description("Scalar input (rank-0) transpose is identity")
            .override_input_shapes(data=[])
            .override_output_shapes(transpose_output=[])
            .tags("edge", "scalar")
            .build(),
            edge_case_test("single_dim_tensor")
            .description("Rank-1 tensor transpose is identity")
            .override_input_shapes(data=[10])
            .override_output_shapes(transpose_output=[10])
            .tags("edge", "rank1")
            .build(),
            edge_case_test("zero_dim")
            .description("Tensor with zero-sized dimension")
            .override_input_shapes(data=[2, 0, 4])
            .override_attrs(perm=[2, 1, 0])
            .override_output_shapes(transpose_output=[4, 0, 2])
            .tags("edge", "zero_dim")
            .build(),
            # ---------------------------------------------------------
            # E2E correctness tests
            # ---------------------------------------------------------
            e2e_test("e2e_2d_numeric")
            .description("E2E transpose correctness for 2D tensor")
            .override_input_shapes(data=[2, 3])
            .override_output_shapes(transpose_output=[3, 2])
            .override_attrs(perm=[1, 0])
            .tags("e2e", "2d")
            .build(),
            e2e_test("e2e_3d_default_perm")
            .description("E2E transpose correctness with default reverse perm")
            .override_input_shapes(data=[2, 3, 4])
            .tags("e2e", "default_perm", "3d")
            .build(),
            e2e_test("e2e_4d_layout_change")
            .description("E2E transpose for layout change (NCHW → NHWC)")
            .override_input_shapes(data=[1, 2, 3, 4])
            .override_attrs(perm=[0, 2, 3, 1])
            .tags("e2e", "layout", "4d")
            .build(),
        ]
