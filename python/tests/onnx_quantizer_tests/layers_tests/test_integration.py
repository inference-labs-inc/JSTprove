from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import numpy as np
import pytest
from onnxruntime import InferenceSession, SessionOptions
from onnxruntime_extensions import get_library_path

from python.core.model_processing.converters.onnx_converter import ONNXConverter

if TYPE_CHECKING:
    from python.core.model_processing.onnx_quantizer.onnx_op_quantizer import (
        ONNXOpQuantizer,
    )
from python.tests.onnx_quantizer_tests import TEST_RNG_SEED
from python.tests.onnx_quantizer_tests.layers.base import (
    LayerTestConfig,
    LayerTestSpec,
    SpecType,
)
from python.tests.onnx_quantizer_tests.layers.factory import TestLayerFactory
from python.tests.onnx_quantizer_tests.layers_tests.base_test import (
    BaseQuantizerTest,
)


class TestIntegration(BaseQuantizerTest):
    """Integration tests for ONNX quantizer"""

    __test__ = True

    @pytest.mark.integration
    @pytest.mark.parametrize(
        "layer_combination",
        [["Conv", "Relu"], ["Gemm", "Relu"], ["Conv", "MaxPool", "Flatten", "Gemm"]],
    )
    def test_check_then_quantize_workflow(
        self: TestIntegration,
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
            assert result is not None, (
                f"Quantization failed for {node.op_type}"
                f" in combination {layer_combination}"
            )

    def skip_by_layer_name(
        self,
        layer_name: str,
        test_spec: LayerTestSpec,
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
        TestLayerFactory.get_test_cases_by_type(SpecType.VALID),  # type: ignore[arg-type]
        ids=BaseQuantizerTest._generate_test_id,
    )
    def test_end_to_end_quantization_accuracy(
        self: TestIntegration,
        test_case_data: tuple[str, LayerTestConfig, LayerTestSpec],
    ) -> None:
        """Test end-to-end quantization accuracy for each valid test case.

        Builds a model from the layer config, runs inference on the original model,
        quantizes the model, runs inference on the quantized model, and ensures
        the outputs are close.
        """
        cosine_similarity = 0.995
        rng = np.random.default_rng(TEST_RNG_SEED + 1)

        layer_name, config, test_spec = test_case_data
        self.skip_by_layer_name(layer_name, test_spec, skip_layer="Constant")

        # Skip if validation failed or test is skipped
        self._check_validation_dependency(test_case_data)
        if test_spec.skip_reason:
            pytest.skip(f"{layer_name}_{test_spec.name}: {test_spec.skip_reason}")

        # Create original model
        original_model = config.create_test_model(test_spec)
        opts = SessionOptions()
        opts.register_custom_ops_library(get_library_path())
        original_session = InferenceSession(
            original_model.SerializeToString(),
            opts,
            providers=["CPUExecutionProvider"],
        )

        input_shapes = {i.name: tuple(i.shape) for i in original_session.get_inputs()}

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

        # For the quantized model, cast inputs to float64 for ORT
        quantized_inputs = {}
        for name in quantized_input_names:
            if name in dummy_inputs:
                quantized_inputs[name] = dummy_inputs[name].astype(np.float64)
            else:
                # If quantized model has different inputs, skip this case
                pytest.skip(
                    f"Quantized model input mismatch for {layer_name}.{test_spec.name}",
                )

        quantized_output = quantized_session.run(
            [quantized_output_name],
            quantized_inputs,
        )[0]

        quantized_output = quantized_output / (scale_base ** (scale_exponent))

        ratio = np.mean(quantized_output / (original_output + 1e-12))
        print(f"Mean output ratio (quantized/original): {ratio:.4f}")

        # Compare outputs (quantized output should be close to original if rescale=True)
        # Allow some tolerance due to quantization
        np.testing.assert_allclose(
            original_output,
            quantized_output,
            rtol=0.05,  # Relative tolerance
            atol=0.05,  # Absolute tolerance
            err_msg=f"Quantization accuracy failed for {layer_name}.{test_spec.name}",
        )

        cos_sim = np.dot(original_output.flatten(), quantized_output.flatten()) / (
            np.linalg.norm(original_output.flatten())
            * np.linalg.norm(quantized_output.flatten())
            + 1e-12
        )
        print(f"Cosine similarity: {cos_sim:.6f}")
        assert cos_sim > cosine_similarity, f"Low cosine similarity ({cos_sim:.6f})"
