from python.tests.onnx_quantizer_tests.layers.base import (
    LayerTestSpec,
    valid_test,
)
from python.tests.onnx_quantizer_tests.layers.factory import (
    BaseLayerConfigProvider,
    LayerTestConfig,
)


class ConcatConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for ONNX Concat"""

    @property
    def layer_name(self) -> str:
        return "Concat"

    def get_config(self) -> LayerTestConfig:
        # Default: concatenate 2 tensors along channel axis
        # Shape: N x C x H x W
        default_shape = [1, 2, 4, 4]

        return LayerTestConfig(
            op_type="Concat",
            valid_inputs=["X1", "X2"],
            valid_attributes={
                "axis": 1,
            },
            required_initializers={},
            input_shapes={
                "X1": default_shape,
                "X2": default_shape,
            },
            output_shapes={
                # C doubles since axis=1
                "concat_output": [1, 4, 4, 4],
            },
        )

    def get_test_specs(self) -> list[LayerTestSpec]:
        return [
            # --------------------
            # Basic valid tests
            # --------------------
            valid_test("basic_concat_axis1")
            .description("Basic concat of two 4D tensors along channel axis")
            .override_attrs(axis=0)
            .tags("basic", "axis1", "4d")
            .build(),
            # valid_test("concat_axis0_batch")
            # .description("Concatenate along batch dimension")
            # .override_attrs(axis=0)
            # .override_input_shapes(
            #     X1=[1, 2, 4, 4],
            #     X2=[2, 2, 4, 4],
            # )
            # .override_output_shapes(
            #     concat_result=[3, 2, 4, 4],
            # )
            # .tags("axis0", "batch")
            # .build(),
            # valid_test("concat_negative_axis")
            # .description("Concat using negative axis (-1 => last dimension)")
            # .override_attrs(axis=-1)
            # .override_input_shapes(
            #     X1=[1, 2, 4, 3],
            #     X2=[1, 2, 4, 5],
            # )
            # .override_output_shapes(
            #     concat_result=[1, 2, 4, 8],
            # )
            # .tags("negative_axis", "last_dim")
            # .build(),
            # # --------------------
            # # Variadic inputs
            # # --------------------
            # valid_test("concat_three_inputs")
            # .description("Concatenate three tensors along channel axis")
            # .override_inputs("X1", "X2", "X3")
            # .override_input_shapes(
            #     X1=[1, 1, 4, 4],
            #     X2=[1, 2, 4, 4],
            #     X3=[1, 3, 4, 4],
            # )
            # .override_output_shapes(
            #     concat_result=[1, 6, 4, 4],
            # )
            # .tags("variadic", "3_inputs")
            # .build(),
            # # --------------------
            # # Different ranks
            # # --------------------
            # valid_test("concat_1d")
            # .description("1D concat along axis 0")
            # .override_attrs(axis=0)
            # .override_input_shapes(
            #     X1=[3],
            #     X2=[5],
            # )
            # .override_output_shapes(
            #     concat_result=[8],
            # )
            # .tags("1d")
            # .build(),
            # valid_test("concat_2d_axis1")
            # .description("2D concat along second dimension")
            # .override_attrs(axis=1)
            # .override_input_shapes(
            #     X1=[2, 3],
            #     X2=[2, 5],
            # )
            # .override_output_shapes(
            #     concat_result=[2, 8],
            # )
            # .tags("2d", "axis1")
            # .build(),
            # # --------------------
            # # Error tests
            # # --------------------
            # error_test("mismatched_non_axis_dim")
            # .description("Inputs differ in non-concat dimension")
            # .override_attrs(axis=1)
            # .override_input_shapes(
            #     X1=[1, 2, 4, 4],
            #     X2=[1, 3, 5, 4],
            # )
            # .expects_error(ValueError, "non-concat dimension")
            # .tags("error", "shape_mismatch")
            # .build(),
            # error_test("axis_out_of_range")
            # .description("Axis is outside valid range")
            # .override_attrs(axis=5)
            # .expects_error(ValueError, "axis")
            # .tags("error", "axis")
            # .build(),
            # error_test("rank_mismatch")
            # .description("Inputs have different ranks")
            # .override_input_shapes(
            #     X1=[1, 3, 4, 4],
            #     X2=[3, 4],
            # )
            # .expects_error(ValueError, "rank")
            # .tags("error", "rank")
            # .build(),
            # # --------------------
            # # E2E tests
            # # --------------------
            # e2e_test("e2e_concat_axis1")
            # .description("E2E concat along channel axis with known values")
            # .override_input_shapes(
            #     X1=[1, 2, 1, 1],
            #     X2=[1, 3, 1, 1],
            # )
            # .override_output_shapes(
            #     concat_result=[1, 5, 1, 1],
            # )
            # .tags("e2e", "axis1")
            # .build(),
            # e2e_test("e2e_concat_negative_axis")
            # .description("E2E concat using negative axis")
            # .override_attrs(axis=-1)
            # .override_input_shapes(
            #     X1=[1, 1, 2],
            #     X2=[1, 1, 3],
            # )
            # .override_output_shapes(
            #     concat_result=[1, 1, 5],
            # )
            # .tags("e2e", "negative_axis")
            # .build(),
        ]
