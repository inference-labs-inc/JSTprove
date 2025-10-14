from __future__ import annotations

import numpy as np
import pytest

from python.core.model_processing.onnx_quantizer.onnx_op_quantizer import (
    ONNXOpQuantizer,
)
from python.tests.onnx_quantizer_tests.layers.base import LayerTestConfig
from python.tests.onnx_quantizer_tests.layers.factory import TestLayerFactory


class TestScalability:
    """Tests (meta) to verify the framework scales with new layers"""

    @pytest.mark.unit
    def test_adding_new_layer_config(self: TestScalability) -> None:
        """Test that adding new layer configs is straightforward"""
        two = 2
        # Simulate adding a new layer type
        new_layer_config = LayerTestConfig(
            op_type="NewCustomOp",
            valid_inputs=["input", "custom_param"],
            valid_attributes={"custom_attr": 42},
            required_initializers={"custom_param": np.array([1, 2, 3])},
        )

        # Verify config can create nodes and initializers
        node = new_layer_config.create_node()
        assert node.op_type == "NewCustomOp"
        assert len(node.input) == two

        initializers = new_layer_config.create_initializers()
        assert "custom_param" in initializers

    @pytest.mark.unit
    def test_layer_config_extensibility(self: TestScalability) -> None:
        """Test that layer configs consists of all registered handlers"""
        configs = TestLayerFactory.get_layer_configs()

        # Verify all expected layers are present
        unsupported = ONNXOpQuantizer().handlers.keys() - set(configs.keys())
        print(unsupported)
        assert (
            unsupported == set()
        ), f"The following layers are not being configured for testing: {unsupported}."
        " Please add configuration in tests/onnx_quantizer_tests/layers/"

        # Verify each config has required components
        for layer_type, config in configs.items():
            assert (
                config.op_type == layer_type
            ), "Quantization test config is not supported yet for {}"
            " and must be implemented"
            assert isinstance(
                config.valid_inputs,
                list,
            ), "Quantization test config is not supported yet for {}"
            " and must be implemented"
            assert isinstance(
                config.valid_attributes,
                dict,
            ), "Quantization test config is not supported yet for {}"
            " and must be implemented"
            assert isinstance(
                config.required_initializers,
                dict,
            ), "Quantization test config is not supported yet for {}"
            " and must be implemented"
