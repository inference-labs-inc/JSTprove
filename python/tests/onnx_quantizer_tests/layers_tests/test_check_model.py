from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from onnx import TensorProto, helper

from python.core.model_processing.onnx_quantizer.exceptions import (
    InvalidParamError,
    UnsupportedOpError,
)

if TYPE_CHECKING:
    from python.core.model_processing.onnx_quantizer.onnx_op_quantizer import (
        ONNXOpQuantizer,
    )
from python.tests.onnx_quantizer_tests.layers.base import (
    LayerTestConfig,
    LayerTestSpec,
    SpecType,
)
from python.tests.onnx_quantizer_tests.layers.factory import TestLayerFactory
from python.tests.onnx_quantizer_tests.layers_tests.base_test import (
    BaseQuantizerTest,
)


class TestCheckModel(BaseQuantizerTest):
    """Tests for ONNX model checking."""

    __test__ = True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "test_case_data",
        TestLayerFactory.get_test_cases_by_type(SpecType.VALID),  # type: ignore[arg-type]
        ids=BaseQuantizerTest._generate_test_id,
    )
    def test_check_model_individual_valid_cases(
        self: TestCheckModel,
        quantizer: ONNXOpQuantizer,
        test_case_data: tuple[str, LayerTestConfig, LayerTestSpec],
    ) -> None:
        """Test each individual valid test case"""
        layer_name, config, test_spec = test_case_data

        # Skips if layer is not a valid onnx layer
        self._check_validation_dependency(test_case_data)

        if test_spec.skip_reason:
            pytest.skip(f"{layer_name}_{test_spec.name}: {test_spec.skip_reason}")

        # Create model from layer specs
        model = config.create_test_model(test_spec)

        try:
            quantizer.check_model(model)
        except (InvalidParamError, UnsupportedOpError) as e:
            pytest.fail(f"Model check failed for {layer_name}.{test_spec.name}: {e}")
        except Exception as e:
            pytest.fail(f"Model check failed for {layer_name}.{test_spec.name}: {e}")

    @pytest.mark.unit
    def test_check_model_unsupported_layer_fails(
        self: TestCheckModel,
        quantizer: ONNXOpQuantizer,
    ) -> None:
        """Test that models with unsupported layers fail validation"""
        # Create model with unsupported operation
        unsupported_node = helper.make_node(
            "UnsupportedOp",
            inputs=["input"],
            outputs=["output"],
            name="unsupported",
        )

        graph = helper.make_graph(
            [unsupported_node],
            "test_graph",
            [
                helper.make_tensor_value_info(
                    "input",
                    TensorProto.FLOAT,
                    [1, 16, 224, 224],
                ),
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])],
        )

        model = helper.make_model(graph)

        with pytest.raises(UnsupportedOpError):
            quantizer.check_model(model)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "layer_combination",
        [
            ["Conv", "Relu"],
            ["Conv", "Relu", "MaxPool"],
            ["Gemm", "Relu"],
            ["Conv", "Reshape", "Gemm"],
            ["Conv", "Flatten", "Gemm"],
        ],
    )
    def test_check_model_multi_layer_passes(
        self: TestCheckModel,
        quantizer: ONNXOpQuantizer,
        layer_configs: dict[str, LayerTestConfig],
        layer_combination: list[str],
    ) -> None:
        """Test that models with multiple supported layers pass validation"""
        model = self.create_model_with_layers(layer_combination, layer_configs)
        # Should not raise any exception
        quantizer.check_model(model)
