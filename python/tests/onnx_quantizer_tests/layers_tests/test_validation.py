from __future__ import annotations

from typing import TYPE_CHECKING

import onnx
import pytest

if TYPE_CHECKING:
    from python.tests.onnx_quantizer_tests.layers.base import (
        LayerTestConfig,
        LayerTestSpec,
    )
from python.tests.onnx_quantizer_tests.layers.factory import TestLayerFactory
from python.tests.onnx_quantizer_tests.layers_tests.base_test import (
    BaseQuantizerTest,
)


class TestValidation(BaseQuantizerTest):
    """Ensure that layer factory models produce valid ONNX graphs."""

    __test__ = True

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "test_case_data",
        TestLayerFactory.get_all_test_cases(),
        ids=BaseQuantizerTest._generate_test_id,
    )
    def test_factory_models_pass_onnx_validation(
        self: TestValidation,
        test_case_data: tuple[str, LayerTestConfig, LayerTestSpec],
    ) -> None:
        layer_name, config, test_spec = test_case_data
        test_case_id = f"{layer_name}_{test_spec.name}"

        if test_spec.skip_reason:
            pytest.skip(f"{test_case_id}: {test_spec.skip_reason}")

        model = config.create_test_model(test_spec)
        try:
            onnx.checker.check_model(model)
        except onnx.checker.ValidationError as e:
            self._validation_failed_cases.add(test_case_id)
            pytest.fail(f"Invalid ONNX model: {e}")
