from python.tests.onnx_quantizer_tests.layers.base import (
    e2e_test,
    valid_test,
)
from python.tests.onnx_quantizer_tests.layers.factory import (
    BaseLayerConfigProvider,
    LayerTestConfig,
)


class FlattenConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for Flatten layers"""

    @property
    def layer_name(self) -> str:
        return "Flatten"

    def get_config(self) -> LayerTestConfig:
        return LayerTestConfig(
            op_type="Flatten",
            valid_inputs=["input"],
            valid_attributes={"axis": 1},
            required_initializers={},
        )

    def get_test_specs(self) -> list:
        return [
            # --- VALID TESTS ---
            valid_test("basic")
            .description("Basic Flatten from (1,3,4,4) to (1,48)")
            .tags("basic", "flatten")
            .build(),
            valid_test("flatten_axis0")
            .description("Flatten with axis=0 (entire tensor flattened)")
            .override_attrs(axis=0)
            .tags("flatten", "axis0")
            .build(),
            valid_test("flatten_axis2")
            .description("Flatten starting at axis=2")
            .override_attrs(axis=2)
            .tags("flatten", "axis2")
            .build(),
            valid_test("flatten_axis3")
            .description("Flatten starting at axis=3 (minimal flatten)")
            .override_attrs(axis=3)
            .tags("flatten", "axis3")
            .build(),
            e2e_test("e2e_basic")
            .description("End-to-end test for Flatten layer")
            .override_input_shapes(input=[1, 3, 4, 4])
            .override_output_shapes(flatten_output=[1, 48])
            .tags("e2e", "flatten")
            .build(),
            # --- EDGE CASE / SKIPPED TEST ---
            valid_test("large_input")
            .description("Large input flatten (performance test)")
            .override_input_shapes(input=[1, 3, 256, 256])
            .tags("flatten", "large", "performance")
            .skip("Performance test, skipped by default")
            .build(),
        ]
