from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import pytest
from onnx import TensorProto, helper

if TYPE_CHECKING:
    from onnx import ModelProto

    from python.tests.onnx_quantizer_tests.layers.base import (
        LayerTestConfig,
        LayerTestSpec,
    )
from python.core.model_processing.onnx_quantizer.onnx_op_quantizer import (
    ONNXOpQuantizer,
)
from python.tests.onnx_quantizer_tests.layers.factory import TestLayerFactory


class BaseQuantizerTest:
    """Base test utilities for ONNX quantizer tests."""

    __test__ = False  # Prevent pytest from collecting this class directly

    _validation_failed_cases: ClassVar[set[str]] = set()

    @pytest.fixture
    def quantizer(self) -> ONNXOpQuantizer:
        return ONNXOpQuantizer()

    @pytest.fixture
    def layer_configs(self) -> dict[str, LayerTestConfig]:
        return TestLayerFactory.get_layer_configs()

    @staticmethod
    def _generate_test_id(
        test_case_tuple: tuple[str, LayerTestConfig, LayerTestSpec],
    ) -> str:
        try:
            layer_name, _, test_spec = test_case_tuple
        except Exception:
            return str(test_case_tuple)
        else:
            return f"{layer_name}_{test_spec.name}"

    @classmethod
    def _check_validation_dependency(
        cls: BaseQuantizerTest,
        test_case_data: tuple[str, LayerTestConfig, LayerTestSpec],
    ) -> None:
        layer_name, _, test_spec = test_case_data
        test_case_id = f"{layer_name}_{test_spec.name}"
        if test_case_id in cls._validation_failed_cases:
            pytest.skip(f"Skipping because ONNX validation failed for {test_case_id}")

    @staticmethod
    def create_model_with_layers(
        layer_types: list[str],
        layer_configs: dict[str, LayerTestConfig],
    ) -> ModelProto:
        """Create a model composed of several layers."""
        nodes, all_initializers = [], {}

        for i, layer_type in enumerate(layer_types):
            config = layer_configs[layer_type]
            node = config.create_node(name_suffix=f"_{i}")
            if i > 0:
                prev_output = f"{layer_types[i-1].lower()}_output_{i-1}"
                if node.input:
                    node.input[0] = prev_output
            nodes.append(node)
            all_initializers.update(config.create_initializers())

        graph = helper.make_graph(
            nodes,
            "test_graph",
            [
                helper.make_tensor_value_info(
                    "input",
                    TensorProto.FLOAT,
                    [1, 16, 224, 224],
                ),
            ],
            [
                helper.make_tensor_value_info(
                    f"{layer_types[-1].lower()}_output_{len(layer_types)-1}",
                    TensorProto.FLOAT,
                    [1, 10],
                ),
            ],
            initializer=list(all_initializers.values()),
        )
        return helper.make_model(graph)
