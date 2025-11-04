from python.tests.onnx_quantizer_tests.layers.base import (
    e2e_test,
    valid_test,
)
from python.tests.onnx_quantizer_tests.layers.factory import (
    BaseLayerConfigProvider,
    LayerTestConfig,
)


class AddConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for Add layer"""

    @property
    def layer_name(self) -> str:
        return "Add"

    def get_config(self) -> LayerTestConfig:
        return LayerTestConfig(
            op_type="Add",
            valid_inputs=["A", "B"],
            valid_attributes={},  # Add has no layer-specific attributes
            required_initializers={},
            input_shapes={
                "A": [1, 3, 4, 4],
                "B": [1, 3, 4, 4],
            },  # Match weight input dimension K=128
            output_shapes={
                "add_output": [1, 3, 4, 4],
            },
        )

    def get_test_specs(self) -> list:
        return [
            # --- VALID TESTS ---
            valid_test("basic")
            .description("Basic elementwise Add of two same-shaped tensors")
            .override_input_shapes(A=[1, 3, 4, 4], B=[1, 3, 4, 4])
            .tags("basic", "elementwise", "add")
            .build(),
            e2e_test("e2e_add")
            .description("End-to-end Add test with random inputs")
            .override_input_shapes(A=[1, 3, 4, 4], B=[1, 3, 4, 4])
            .override_output_shapes(add_output=[1, 3, 4, 4])
            .tags("e2e", "add", "2d")
            .build(),
        ]
