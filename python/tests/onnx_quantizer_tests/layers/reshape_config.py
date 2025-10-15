import numpy as np

from python.tests.onnx_quantizer_tests.layers.base import e2e_test, valid_test
from python.tests.onnx_quantizer_tests.layers.factory import (
    BaseLayerConfigProvider,
    LayerTestConfig,
)


class ReshapeConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for Reshape layers"""

    @property
    def layer_name(self) -> str:
        return "Reshape"

    def get_config(self) -> LayerTestConfig:
        return LayerTestConfig(
            op_type="Reshape",
            valid_inputs=["input", "shape"],
            valid_attributes={},
            required_initializers={"shape": np.array([1, -1])},
        )

    def get_test_specs(self) -> list:
        return [
            valid_test("basic")
            .description("Basic Reshape from (1,2,3,4) to (1,24)")
            .tags("basic", "reshape")
            .build(),
            e2e_test("e2e_basic")
            .description("End-to-end test for Reshape layer")
            .override_input_shapes(input=[1, 2, 3, 4])
            .override_output_shapes(reshape_output=[1, 24])
            .override_initializer("shape", np.array([1, -1]))
            .tags("e2e", "reshape")
            .build(),
        ]
