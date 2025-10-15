import numpy as np
from onnx import numpy_helper

from python.tests.onnx_quantizer_tests.layers.base import e2e_test, valid_test
from python.tests.onnx_quantizer_tests.layers.factory import (
    BaseLayerConfigProvider,
    LayerTestConfig,
)


class ConstantConfigProvider(BaseLayerConfigProvider):
    """Test configuration provider for Constant layers"""

    @property
    def layer_name(self) -> str:
        return "Constant"

    def get_config(self) -> LayerTestConfig:
        return LayerTestConfig(
            op_type="Constant",
            valid_inputs=[],
            valid_attributes={
                "value": numpy_helper.from_array(np.array([1.0]), name="const_value"),
            },
            required_initializers={},
        )

    def get_test_specs(self) -> list:
        return [
            valid_test("basic")
            .description("Basic Constant node returning scalar 1.0")
            .tags("basic", "constant")
            .build(),
            e2e_test("e2e_basic")
            .description("End-to-end test for Constant node")
            .override_output_shapes(constant_output=[1])
            .tags("e2e", "constant")
            .build(),
        ]
