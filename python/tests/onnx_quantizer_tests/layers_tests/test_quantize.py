from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import onnx
import pytest
from onnx import NodeProto

from python.core.model_processing.onnx_quantizer.exceptions import UnsupportedOpError

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


class TestQuantize(BaseQuantizerTest):
    """Tests for quantization functionality"""

    __test__ = True

    def setup_quantize_test(
        self,
        test_case_data: tuple[str, LayerTestConfig, LayerTestSpec],
        quantizer: ONNXOpQuantizer,
        scale_exponent: int = 2,
        scale_base: int = 10,
        *,
        rescale: bool = True,
    ) -> tuple[
        list[onnx.NodeProto],
        tuple[str, LayerTestConfig, LayerTestSpec, NodeProto],
    ]:
        """Common setup for quantization tests"""
        layer_name, config, test_spec = test_case_data

        self._check_validation_dependency(test_case_data)

        if test_spec.skip_reason:
            pytest.skip(f"{layer_name}_{test_spec.name}: {test_spec.skip_reason}")

        model = config.create_test_model(test_spec)
        node = model.graph.node[0]
        initializer_map = {init.name: init for init in model.graph.initializer}

        mock_graph = Mock()
        if node.op_type == "Constant":
            mock_data_node = Mock()
            mock_data_node.input = [node.output[0]]
            mock_graph.node = [mock_data_node]

        result = quantizer.quantize(
            node=node,
            rescale=rescale,
            graph=mock_graph,
            scale_exponent=scale_exponent,
            scale_base=scale_base,
            initializer_map=initializer_map,
        )

        if not isinstance(result, list):
            result = [result]
        return result, (layer_name, config, test_spec, node)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "test_case_data",
        TestLayerFactory.get_test_cases_by_type(SpecType.VALID),  # type: ignore[arg-type]
        ids=BaseQuantizerTest._generate_test_id,
    )
    def test_quantize_individual_valid_cases(
        self: TestQuantize,
        quantizer: ONNXOpQuantizer,
        test_case_data: tuple[str, LayerTestConfig, LayerTestSpec],
    ) -> None:
        """Test quantization for each individual valid test case"""

        scale_exponent, scale_base = 2, 10
        rescale = True

        result, (layer_name, _config, test_spec, _node) = self.setup_quantize_test(
            test_case_data,
            quantizer,
            scale_exponent,
            scale_base,
            rescale=rescale,
        )

        # Test that the output of the quantizer quantize is in fact a node
        if isinstance(result, list):
            assert (
                len(result) > 0
            ), f"Quantize returned empty list for {layer_name}.{test_spec.name}"
            for node_result in result:
                assert isinstance(
                    node_result,
                    onnx.NodeProto,
                ), f"Invalid node type returned for {layer_name}.{test_spec.name}"
        else:
            assert isinstance(
                result,
                onnx.NodeProto,
            ), f"Quantize returned none node for {layer_name}.{test_spec.name}"

            assert (
                result is not None
            ), f"Quantize returned None for {layer_name}.{test_spec.name}"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "test_case_data",
        TestLayerFactory.get_test_cases_by_type(SpecType.VALID),  # type: ignore[arg-type]
        ids=BaseQuantizerTest._generate_test_id,
    )
    def test_quantize_preserves_node_names(
        self: TestQuantize,
        quantizer: ONNXOpQuantizer,
        test_case_data: tuple[str, LayerTestConfig, LayerTestSpec],
    ) -> None:
        """Test quantization for each individual valid test case"""

        scale_exponent, scale_base = 2, 10
        rescale = True
        result, (_layer_name, config, _test_spec, node) = self.setup_quantize_test(
            test_case_data,
            quantizer,
            scale_exponent,
            scale_base,
            rescale=rescale,
        )
        is_node_present = False

        def check_node_and_analyze_parameters(
            node: NodeProto,
            result_node: NodeProto,
        ) -> bool:
            if node.op_type == "BatchNormalization":
                pytest.skip(f"{node.op_type} alters the node structure by design")
            if node.op_type in result_node.op_type:
                # Assert there are no less attributes in the new node
                assert len(node.attribute) <= len(result_node.attribute)
                # Ensure that each original node's attributes
                # are contained in the new nodes
                for att in node.attribute:
                    assert att.name in [a.name for a in result_node.attribute]
                return True
            return False

        # Check that result nodes have meaningful names and the relevant node is present
        # And ensure that the new node has the same parameters as the old node
        if isinstance(result, list):
            for result_node in result:
                assert (
                    result_node.name
                ), f"Quantized node missing name for {config.op_type}"
                assert (
                    result_node.op_type
                ), f"Quantized node missing op_type for {config.op_type}"

                is_node_present = is_node_present or check_node_and_analyze_parameters(
                    node,
                    result_node,
                )
        else:
            assert result.name, f"Quantized node missing name for {config.op_type}"
            is_node_present = is_node_present or check_node_and_analyze_parameters(
                node,
                result,
            )

        # Assert that the node is in fact present
        assert (
            is_node_present
        ), "Cannot find quantized node relating to prequantized node"

    @pytest.mark.unit
    @pytest.mark.parametrize("scale_params", [(2, 10), (0, 5)])
    @pytest.mark.parametrize(
        "test_case_data",
        TestLayerFactory.get_test_cases_by_type(SpecType.VALID),  # type: ignore[arg-type]
        ids=BaseQuantizerTest._generate_test_id,
    )
    def test_quantize_with_different_scales(
        self: TestQuantize,
        quantizer: ONNXOpQuantizer,
        test_case_data: tuple[str, LayerTestConfig, LayerTestSpec],
        scale_params: tuple[int, int],
    ) -> None:
        """Test quantization for each individual valid test case"""

        # Test for both scale parameters
        scale_exponent, scale_base = scale_params
        rescale = True
        result, (_layer_name, _config, _test_spec, _node) = self.setup_quantize_test(
            test_case_data,
            quantizer,
            scale_exponent,
            scale_base,
            rescale=rescale,
        )

        # Should return valid result regardless of scale values
        assert (
            result is not None
        ), f"Quantize returned None for scale={scale_exponent}, scale_base={scale_base}"

    @pytest.mark.unit
    @pytest.mark.parametrize("rescale", [True, False])
    @pytest.mark.parametrize(
        "test_case_data",
        TestLayerFactory.get_test_cases_by_type(SpecType.VALID),  # type: ignore[arg-type]
        ids=BaseQuantizerTest._generate_test_id,
    )
    def test_quantize_with_different_rescales(
        self: TestQuantize,
        quantizer: ONNXOpQuantizer,
        test_case_data: tuple[str, LayerTestConfig, LayerTestSpec],
        *,
        rescale: bool,
    ) -> None:
        """Test quantization for each individual valid test case"""

        scale_exponent, scale_base = 2, 10

        # Test that quantizing works with both rescaling values
        result, (_layer_name, _config, _test_spec, _node) = self.setup_quantize_test(
            test_case_data,
            quantizer,
            scale_exponent,
            scale_base,
            rescale=rescale,
        )
        assert result is not None, f"Quantize failed with rescale={rescale}"

    @pytest.mark.unit
    def test_quantize_unsupported_layer_returns_original(
        self: TestQuantize,
        quantizer: ONNXOpQuantizer,
    ) -> None:
        """Test that unsupported layers return Error in quantization process"""
        from onnx import helper  # noqa: PLC0415

        mock_graph = Mock()
        scale_exponent, scale_base = 2, 10
        rescale = True

        unsupported_node = helper.make_node(
            "UnsupportedOp",
            inputs=["input"],
            outputs=["output"],
            name="unsupported",
        )
        with pytest.raises(UnsupportedOpError) as excinfo:
            _ = quantizer.quantize(
                node=unsupported_node,
                rescale=rescale,
                graph=mock_graph,
                scale_exponent=scale_exponent,
                scale_base=scale_base,
                initializer_map={},
            )
        assert "Unsupported op type: 'UnsupportedOp'" in str(excinfo.value)
