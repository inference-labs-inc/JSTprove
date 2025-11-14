from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

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


class TestErrorCases(BaseQuantizerTest):
    """Tests for ONNX model checking."""

    __test__ = True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "test_case_data",
        TestLayerFactory.get_test_cases_by_type(SpecType.ERROR),  # type: ignore[arg-type]
        ids=BaseQuantizerTest._generate_test_id,
    )
    def test_check_model_individual_error_cases(
        self: TestErrorCases,
        quantizer: ONNXOpQuantizer,
        test_case_data: tuple[str, LayerTestConfig, LayerTestSpec],
    ) -> None:
        """Test each individual error test case"""
        layer_name, config, test_spec = test_case_data

        # Skips if layer is not a valid onnx layer
        self._check_validation_dependency(test_case_data)

        if test_spec.skip_reason:
            pytest.skip(f"{layer_name}_{test_spec.name}: {test_spec.skip_reason}")

        # Create model from layer specs
        model = config.create_test_model(test_spec)

        # Ensures that expected test is in fact raised
        with pytest.raises(test_spec.expected_error) as exc:
            quantizer.check_model(model)

        # Ensures the error message is as expected
        if isinstance(test_spec.error_match, list):
            for e in test_spec.error_match:
                assert e in str(exc.value)
        else:
            assert test_spec.error_match in str(exc.value)
