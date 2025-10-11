from __future__ import annotations

import typing
from typing import Any, Literal
from unittest.mock import Mock

import numpy as np
import onnx
import pytest
from onnx import NodeProto, TensorProto, helper, numpy_helper
from onnxruntime import InferenceSession, SessionOptions
from onnxruntime_extensions import get_library_path

from python.core.model_processing.converters.onnx_converter import ONNXConverter
from python.core.model_processing.onnx_quantizer.exceptions import (
    InvalidParamError,
    UnsupportedOpError,
)
from python.core.model_processing.onnx_quantizer.onnx_op_quantizer import (
    ONNXOpQuantizer,
)

# Import your classes (adjust imports as needed)
from python.tests.onnx_quantizer_tests import TEST_RNG_SEED
from python.tests.onnx_quantizer_tests.layers.base import SpecType, TestSpec
from python.tests.onnx_quantizer_tests.layers.factory import (
    TestLayerFactory,
)
from python.tests.onnx_quantizer_tests.testing_helper_functions import get_input_shapes


class LayerTestConfig:
    """Configuration class for layer-specific test data"""

    def __init__(
        self: LayerTestConfig,
        op_type: str,
        valid_inputs: list[str],
        valid_attributes: dict[str, Any],
        required_initializers: dict[str, np.ndarray],
    ) -> None:
        self.op_type = op_type
        self.valid_inputs = valid_inputs
        self.valid_attributes = valid_attributes
        self.required_initializers = required_initializers

    def create_node(
        self,
        name_suffix: str = "",
        **attr_overrides: dict[str, Any],
    ) -> onnx.NodeProto:
        """Create a valid node for this layer type"""
        attrs = {**self.valid_attributes, **attr_overrides}
        return helper.make_node(
            self.op_type,
            inputs=self.valid_inputs,
            outputs=[f"{self.op_type.lower()}_output{name_suffix}"],
            name=f"test_{self.op_type.lower()}{name_suffix}",
            **attrs,
        )

    def create_initializers(self) -> dict[str, onnx.TensorProto]:
        """Create initializer tensors for this layer"""
        initializers = {}
        for name, data in self.required_initializers.items():
            tensor = numpy_helper.from_array(data.astype(np.float32), name=name)
            initializers[name] = tensor
        return initializers


class TestONNXOpQuantizer:
    """Generic unit tests for ONNX Op Quantizer"""

    _validation_failed_cases: typing.ClassVar = set()

    @pytest.fixture
    def quantizer(self: TestONNXOpQuantizer) -> ONNXOpQuantizer:
        """Create quantizer instance"""
        return ONNXOpQuantizer()

    @pytest.fixture
    def layer_configs(self: TestONNXOpQuantizer) -> dict[str, LayerTestConfig]:
        """Get all layer configurations"""
        return TestLayerFactory.get_layer_configs()

    def _validate_onnx_model(self, model: onnx.ModelProto, test_case_name: str) -> None:
        """
        Validate ONNX model using onnx.checker.check_model().
        Skip test if model is invalid.

        Args:
            model: ONNX model to validate
            test_case_name: Name of the test case for error reporting

        Raises:
            pytest.skip: If model validation fails
        """
        try:
            onnx.checker.check_model(model)
        except onnx.checker.ValidationError as e:
            error_msg = (
                f"ONNX model validation failed for {test_case_name}. "
                f"Model structure is invalid: {e!s}"
            )
            print(error_msg)
            pytest.skip(error_msg)
        except Exception as e:
            error_msg = (
                f"Unexpected error during ONNX model validation for {test_case_name}: "
                f"{type(e).__name__}: {e!s}"
            )
            pytest.skip(error_msg)

    @classmethod
    def _check_validation_dependency(
        cls: TestONNXOpQuantizer,
        test_case_data: tuple[str, LayerTestConfig, TestSpec],
    ) -> None:
        """Check if validation failed for this test case and skip if so"""
        layer_name, _, test_spec = test_case_data
        test_case_id = f"{layer_name}_{test_spec.name}"

        if test_case_id in cls._validation_failed_cases:
            pytest.skip(
                "Skipping because ONNX validation failed for "
                f"{layer_name}.{test_spec.name}",
            )

    @staticmethod
    def _generate_test_id(
        test_case_tuple: tuple[str, LayerTestConfig, TestSpec],
    ) -> str:
        """Generate test ID from test case tuple"""
        try:
            layer_name, _, test_spec = test_case_tuple
        except (IndexError, AttributeError):
            return str(test_case_tuple)
        except Exception:
            return "unknown"
        else:
            return f"{layer_name}_{test_spec.name}"

    def create_model_with_layers(
        self: TestONNXOpQuantizer,
        layer_types: list[str],
        layer_configs: dict[str, LayerTestConfig],
    ) -> onnx.ModelProto:
        """Create a model with specified layer types"""
        nodes = []
        all_initializers = {}

        for i, layer_type in enumerate(layer_types):
            config = layer_configs[layer_type]
            node = config.create_node(name_suffix=f"_{i}")

            # Update inputs to chain layers together
            if i > 0:
                prev_output = f"{layer_types[i-1].lower()}_output_{i-1}"
                if node.input:
                    node.input[0] = prev_output

            nodes.append(node)
            all_initializers.update(config.create_initializers())

        # Create graph
        input_info = helper.make_tensor_value_info(
            "input",
            TensorProto.FLOAT,
            [1, 16, 224, 224],
        )
        output_name = f"{layer_types[-1].lower()}_output_{len(layer_types)-1}"
        output_info = helper.make_tensor_value_info(
            output_name,
            TensorProto.FLOAT,
            [1, 10],
        )

        graph = helper.make_graph(
            nodes,
            "test_graph",
            [input_info],
            [output_info],
            initializer=list(all_initializers.values()),
        )

        return helper.make_model(graph)

    # ===== VAlIDATE MODEL =======

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "test_case_data",
        TestLayerFactory.get_all_test_cases(),
        ids=_generate_test_id.__func__,
    )
    def test_factory_models_pass_onnx_validation(
        self: TestONNXOpQuantizer,
        test_case_data: tuple[str, LayerTestConfig, TestSpec],
    ) -> None:
        """Test that all factory-created models pass ONNX validation"""
        layer_name, config, test_spec = test_case_data
        test_case_id = f"{layer_name}_{test_spec.name}"

        if test_spec.skip_reason:
            pytest.skip(f"{layer_name}_{test_spec.name}: {test_spec.skip_reason}")

        # Create model
        model = config.create_test_model(test_spec)
        test_case_name = f"{layer_name}.{test_spec.name}"

        try:
            # Onnx check of model
            onnx.checker.check_model(model)
        except onnx.checker.ValidationError as e:
            # Mark this test case as failed for other tests to check
            self._validation_failed_cases.add(test_case_id)
            pytest.fail(
                f"ONNX model validation failed for {test_case_name}. "
                f"Model structure is invalid: {e!s}",
            )
        except Exception as e:
            # Mark this test case as failed for other tests to check
            self._validation_failed_cases.add(test_case_id)
            pytest.fail(
                f"Unexpected error during ONNX model validation for {test_case_name}: "
                f"{type(e).__name__}: {e!s}",
            )

    # ===== CHECK_MODEL TESTS =====
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "test_case_data",
        TestLayerFactory.get_test_cases_by_type(SpecType.VALID),
        ids=_generate_test_id.__func__,
    )
    def test_check_model_individual_valid_cases(
        self: TestONNXOpQuantizer,
        quantizer: ONNXOpQuantizer,
        test_case_data: tuple[str, LayerTestConfig, TestSpec],
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
        self: TestONNXOpQuantizer,
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
        self: TestONNXOpQuantizer,
        quantizer: ONNXOpQuantizer,
        layer_configs: dict[str, LayerTestConfig],
        layer_combination: list[str],
    ) -> None:
        """Test that models with multiple supported layers pass validation"""
        model = self.create_model_with_layers(layer_combination, layer_configs)
        # Should not raise any exception
        quantizer.check_model(model)

    # ===== QUANTIZE TESTS =====
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "test_case_data",
        TestLayerFactory.get_test_cases_by_type(SpecType.VALID),
        ids=_generate_test_id.__func__,
    )
    def test_quantize_individual_valid_cases(
        self,
        quantizer: ONNXOpQuantizer,
        test_case_data: tuple[str, LayerTestConfig, TestSpec],
    ) -> None:
        """Test quantization for each individual valid test case"""
        layer_name, config, test_spec = test_case_data

        # Skips if layer is not a valid onnx layer
        self._check_validation_dependency(test_case_data)

        if test_spec.skip_reason:
            pytest.skip(f"{layer_name}_{test_spec.name}: {test_spec.skip_reason}")

        # Create model from layer specs
        model = config.create_test_model(test_spec)
        node = model.graph.node[0]
        initializer_map = {init.name: init for init in model.graph.initializer}

        mock_graph = Mock()
        if node.op_type == "Constant":
            mock_data_node = Mock()
            mock_data_node.input = [node.output[0]]
            mock_graph.node = [mock_data_node]

        scale_exponent, scale_base = 2, 10
        rescale = True

        result = quantizer.quantize(
            node=node,
            rescale=rescale,
            graph=mock_graph,
            scale_exponent=scale_exponent,
            scale_base=scale_base,
            initializer_map=initializer_map,
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
        TestLayerFactory.get_test_cases_by_type(SpecType.VALID),
        ids=_generate_test_id.__func__,
    )
    def test_quantize_preserves_node_names(
        self: TestONNXOpQuantizer,
        quantizer: ONNXOpQuantizer,
        test_case_data: tuple[str, LayerTestConfig, TestSpec],
    ) -> None:
        """Test quantization for each individual valid test case"""
        layer_name, config, test_spec = test_case_data

        # Skips if layer is not a valid onnx layer
        self._check_validation_dependency(test_case_data)

        if test_spec.skip_reason:
            pytest.skip(f"{layer_name}_{test_spec.name}: {test_spec.skip_reason}")

        # Create model from layer specs
        model = config.create_test_model(test_spec)
        node = model.graph.node[0]
        initializer_map = {init.name: init for init in model.graph.initializer}

        mock_graph = Mock()
        if node.op_type == "Constant":
            mock_data_node = Mock()
            mock_data_node.input = [node.output[0]]
            mock_graph.node = [mock_data_node]

        scale_exponent, scale_base = 2, 10
        rescale = True
        result = quantizer.quantize(
            node=node,
            rescale=rescale,
            graph=mock_graph,
            scale_exponent=scale_exponent,
            scale_base=scale_base,
            initializer_map=initializer_map,
        )
        is_node_present = False

        def check_node_and_analyze_parameters(
            node: NodeProto,
            result_node: NodeProto,
        ) -> bool:
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
        TestLayerFactory.get_test_cases_by_type(SpecType.VALID),
        ids=_generate_test_id.__func__,
    )
    def test_quantize_with_different_scales(
        self: TestONNXOpQuantizer,
        quantizer: ONNXOpQuantizer,
        test_case_data: tuple[str, LayerTestConfig, TestSpec],
        scale_params: Literal[2, 0],
    ) -> None:
        """Test quantization for each individual valid test case"""
        layer_name, config, test_spec = test_case_data

        # Skips if layer is not a valid onnx layer
        self._check_validation_dependency(test_case_data)

        if test_spec.skip_reason:
            pytest.skip(f"{layer_name}_{test_spec.name}: {test_spec.skip_reason}")

        # Create model from layer specs
        model = config.create_test_model(test_spec)
        node = model.graph.node[0]
        initializer_map = {init.name: init for init in model.graph.initializer}

        mock_graph = Mock()
        if node.op_type == "Constant":
            mock_data_node = Mock()
            mock_data_node.input = [node.output[0]]
            mock_graph.node = [mock_data_node]

        # Test for both scale parameters
        scale_exponent, scale_base = scale_params
        rescale = True
        result = quantizer.quantize(
            node=node,
            rescale=rescale,
            graph=mock_graph,
            scale_exponent=scale_exponent,
            scale_base=scale_base,
            initializer_map=initializer_map,
        )

        # Should return valid result regardless of scale values
        assert (
            result is not None
        ), f"Quantize returned None for scale={scale_exponent}, scale_base={scale_base}"

    @pytest.mark.unit
    @pytest.mark.parametrize("rescale", [True, False])
    @pytest.mark.parametrize(
        "test_case_data",
        TestLayerFactory.get_test_cases_by_type(SpecType.VALID),
        ids=_generate_test_id.__func__,
    )
    def test_quantize_with_different_rescales(
        self: TestONNXOpQuantizer,
        quantizer: ONNXOpQuantizer,
        test_case_data: tuple[str, LayerTestConfig, TestSpec],
        *,
        rescale: bool,
    ) -> None:
        """Test quantization for each individual valid test case"""
        layer_name, config, test_spec = test_case_data

        # Skips if layer is not a valid onnx layer
        self._check_validation_dependency(test_case_data)

        if test_spec.skip_reason:
            pytest.skip(f"{layer_name}_{test_spec.name}: {test_spec.skip_reason}")

        # Create model from layer specs
        model = config.create_test_model(test_spec)
        node = model.graph.node[0]
        initializer_map = {init.name: init for init in model.graph.initializer}

        mock_graph = Mock()
        if node.op_type == "Constant":
            mock_data_node = Mock()
            mock_data_node.input = [node.output[0]]
            mock_graph.node = [mock_data_node]

        scale_exponent, scale_base = 2, 10

        # Test that quantizing works with both rescaling values
        result = quantizer.quantize(
            node=node,
            rescale=rescale,
            graph=mock_graph,
            scale_exponent=scale_exponent,
            scale_base=scale_base,
            initializer_map=initializer_map,
        )
        assert result is not None, f"Quantize failed with rescale={rescale}"

    @pytest.mark.unit
    def test_quantize_unsupported_layer_returns_original(
        self: TestONNXOpQuantizer,
        quantizer: ONNXOpQuantizer,
    ) -> None:
        """Test that unsupported layers return Error in quantization process"""
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

    # ===== INTEGRATION TESTS =====

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "layer_combination",
        [["Conv", "Relu"], ["Gemm", "Relu"], ["Conv", "MaxPool", "Flatten", "Gemm"]],
    )
    def test_check_then_quantize_workflow(
        self: TestONNXOpQuantizer,
        quantizer: ONNXOpQuantizer,
        layer_configs: dict[str, LayerTestConfig],
        layer_combination: list[str],
    ) -> None:
        """Test the typical workflow: check model then quantize layers"""
        mock_graph = Mock()
        scale_exponent, scale_base = 2, 10
        rescale = True

        # Step 1: Create and validate model
        model = self.create_model_with_layers(layer_combination, layer_configs)
        quantizer.check_model(model)  # Should not raise

        # Step 2: Quantize each layer
        initializer_map = quantizer.get_initializer_map(model)

        for node in model.graph.node:
            result = quantizer.quantize(
                node=node,
                rescale=rescale,
                graph=mock_graph,
                scale_exponent=scale_exponent,
                scale_base=scale_base,
                initializer_map=initializer_map,
            )
            assert result is not None, f"Quantization failed for {node.op_type}"
            f" in combination {layer_combination}"

    # ===== Error TESTS =====

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "test_case_data",
        TestLayerFactory.get_test_cases_by_type(SpecType.ERROR),
        ids=_generate_test_id.__func__,
    )
    def test_check_model_individual_error_cases(
        self,
        quantizer: ONNXOpQuantizer,
        test_case_data: tuple[str, LayerTestConfig, TestSpec],
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

    # ===== END-TO-END ACCURACY TESTS =====

    def skip_by_layer_name(
        self,
        layer_name: str,
        test_spec: TestSpec,
        skip_layer: str,
    ) -> None:
        # Skip Constant nodes as they don't depend on scaled inputs
        if layer_name == skip_layer:
            pytest.skip(
                f"Skipping accuracy test for {layer_name}."
                f"{test_spec.name} as constants are scaled differently",
            )

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "test_case_data",
        TestLayerFactory.get_test_cases_by_type(SpecType.VALID),
        ids=_generate_test_id.__func__,
    )
    def test_end_to_end_quantization_accuracy(
        self: TestONNXOpQuantizer,
        test_case_data: tuple[str, LayerTestConfig, TestSpec],
    ) -> None:
        """Test end-to-end quantization accuracy for each valid test case.

        Builds a model from the layer config, runs inference on the original model,
        quantizes the model, runs inference on the quantized model, and ensures
        the outputs are close.
        """
        cosine_similarity = 0.995
        rng = np.random.default_rng(TEST_RNG_SEED)

        layer_name, config, test_spec = test_case_data
        self.skip_by_layer_name(layer_name, test_spec, skip_layer="Constant")

        # Skip if validation failed or test is skipped
        self._check_validation_dependency(test_case_data)
        if test_spec.skip_reason:
            pytest.skip(f"{layer_name}_{test_spec.name}: {test_spec.skip_reason}")

        # Create original model
        original_model = config.create_test_model(test_spec)

        input_shapes = get_input_shapes(original_model)

        # Skip if no inputs (e.g., Constant nodes)
        if not input_shapes:
            pytest.skip(
                f"No inputs for {layer_name}.{test_spec.name}, skipping accuracy test",
            )

        # Create dummy inputs for all graph inputs

        dummy_inputs = {}
        for name, shape in input_shapes.items():
            dummy_inputs[name] = rng.normal(0, 1, shape).astype(np.float32)

        # Run inference on original model
        opts = SessionOptions()
        opts.register_custom_ops_library(get_library_path())
        original_session = InferenceSession(
            original_model.SerializeToString(),
            opts,
            providers=["CPUExecutionProvider"],
        )
        output_name = original_session.get_outputs()[0].name
        original_output = original_session.run([output_name], dummy_inputs)[0]

        # Quantize the model

        converter = ONNXConverter()
        scale_base, scale_exponent = (
            2,
            10,
        )  # Smaller scale to reduce quantization errors
        quantized_model = converter.quantize_model(
            original_model,
            scale_base=scale_base,
            scale_exponent=scale_exponent,
            rescale_config=None,  # Use default rescale
        )

        # Run inference on quantized model
        quantized_session = InferenceSession(
            quantized_model.SerializeToString(),
            opts,
            providers=["CPUExecutionProvider"],
        )
        quantized_input_names = [inp.name for inp in quantized_session.get_inputs()]
        quantized_output_name = quantized_session.get_outputs()[0].name

        # For quantized model, scale the inputs
        scaled_inputs = {}
        for name in quantized_input_names:
            if name in dummy_inputs:
                scaled_inputs[name] = (
                    dummy_inputs[name] * (scale_base**scale_exponent)
                ).astype(np.float64)
            else:
                # If quantized model has different inputs, skip or handle
                pytest.skip(
                    f"Quantized model input mismatch for {layer_name}.{test_spec.name}",
                )

        quantized_output = quantized_session.run(
            [quantized_output_name],
            scaled_inputs,
        )[0]
        quantized_output = quantized_output / (scale_base ** (scale_exponent * 2))

        ratio = np.mean(quantized_output / (original_output + 1e-12))
        print(f"Mean output ratio (quantized/original): {ratio:.4f}")

        tol = max(1e-3, 0.1 * np.abs(original_output).max())

        # Compare outputs (quantized output should be close to original if rescale=True)
        # Allow some tolerance due to quantization
        np.testing.assert_allclose(
            original_output,
            quantized_output,
            rtol=1,  # Relative tolerance
            atol=tol,  # Absolute tolerance
            err_msg=f"Quantization accuracy failed for {layer_name}.{test_spec.name}",
        )

        cos_sim = np.dot(original_output.flatten(), quantized_output.flatten()) / (
            np.linalg.norm(original_output.flatten())
            * np.linalg.norm(quantized_output.flatten())
            + 1e-12
        )
        print(f"Cosine similarity: {cos_sim:.6f}")
        assert cos_sim > cosine_similarity, f"Low cosine similarity ({cos_sim:.6f})"


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
