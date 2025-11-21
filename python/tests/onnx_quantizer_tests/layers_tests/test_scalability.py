from __future__ import annotations

import numpy as np
import pytest

from python.core.model_processing.onnx_quantizer.onnx_op_quantizer import (
    ONNXOpQuantizer,
)
from python.tests.onnx_quantizer_tests.layers.base import LayerTestConfig, SpecType
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
        assert unsupported == set(), (
            f"The following layers are not being configured for testing: {unsupported}."
            " Please add configuration in tests/onnx_quantizer_tests/layers/"
        )

        # Verify each config has required components
        for layer_type, config in configs.items():
            err_msg = (
                f"Quantization test config is not supported yet for {layer_type}"
                " and must be implemented"
            )
            assert config.op_type == layer_type, err_msg
            assert isinstance(
                config.valid_inputs,
                list,
            ), err_msg
            assert isinstance(
                config.valid_attributes,
                dict,
            ), err_msg
            assert isinstance(
                config.required_initializers,
                dict,
            ), err_msg

    @pytest.mark.unit
    def test_every_layer_has_basic_and_e2e(self: TestScalability) -> None:
        """Each registered layer must have at least one basic/valid test
        and one e2e test."""
        missing_basic = []
        missing_e2e = []

        # iterate over registered layers
        for layer_name in TestLayerFactory.get_available_layers():
            cases = TestLayerFactory.get_test_cases_by_layer(layer_name)
            specs = [spec for _, _config, spec in cases]

            # Consider a test "basic" if:
            #  - it has tag 'basic' or 'valid', OR
            #  - its spec_type is SpecType.VALID (if you use SpecType)
            has_basic = any(
                (
                    "basic" in getattr(s, "tags", set())
                    or "valid" in getattr(s, "tags", set())
                    or getattr(s, "spec_type", None) == SpecType.VALID
                )
                for s in specs
            )

            # Consider a test "e2e" if:
            #  - it has tag 'e2e', OR
            #  - its spec_type is SpecType.E2E (if you use that enum)
            has_e2e = any(
                (
                    "e2e" in getattr(s, "tags", set())
                    or getattr(s, "spec_type", None) == SpecType.E2E
                )
                for s in specs
            )

            if not has_basic:
                missing_basic.append(layer_name)
            if not has_e2e:
                missing_e2e.append(layer_name)

        assert not missing_basic, f"Layers missing a basic/valid test: {missing_basic}"
        assert not missing_e2e, f"Layers missing an e2e test: {missing_e2e}"
