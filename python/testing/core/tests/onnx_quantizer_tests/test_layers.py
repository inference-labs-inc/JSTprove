import os
import pytest
import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, MagicMock

# Import your classes (adjust imports as needed)
from python.testing.core.tests.onnx_quantizer_tests.layers.base import SpecType
from python.testing.core.utils.onnx_quantizer.onnx_op_quantizer import ONNXOpQuantizer
from python.testing.core.utils.onnx_quantizer.exceptions import InvalidParamError, UnsupportedOpError

from python.testing.core.tests.onnx_quantizer_tests.layers.factory import TestLayerFactory

from onnx.numpy_helper import from_array




class LayerTestConfig:
    """Configuration class for layer-specific test data"""
    
    def __init__(self, op_type: str, valid_inputs: List[str], 
                 valid_attributes: Dict[str, Any], 
                 required_initializers: Dict[str, np.ndarray]):
        self.op_type = op_type
        self.valid_inputs = valid_inputs
        self.valid_attributes = valid_attributes
        self.required_initializers = required_initializers
    
    def create_node(self, name_suffix: str = "", **attr_overrides) -> onnx.NodeProto:
        """Create a valid node for this layer type"""
        attrs = {**self.valid_attributes, **attr_overrides}
        return helper.make_node(
            self.op_type,
            inputs=self.valid_inputs,
            outputs=[f"{self.op_type.lower()}_output{name_suffix}"],
            name=f"test_{self.op_type.lower()}{name_suffix}",
            **attrs
        )
    
    def create_initializers(self) -> Dict[str, onnx.TensorProto]:
        """Create initializer tensors for this layer"""
        initializers = {}
        for name, data in self.required_initializers.items():
            tensor = numpy_helper.from_array(data.astype(np.float32), name=name)
            initializers[name] = tensor
        return initializers

class TestONNXOpQuantizer:
    """Generic unit tests for ONNX Op Quantizer"""
    _validation_failed_cases = set()
    
    @pytest.fixture
    def quantizer(self):
        """Create quantizer instance"""
        return ONNXOpQuantizer()
    
    @pytest.fixture
    def layer_configs(self):
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
                f"Model structure is invalid: {str(e)}"
            )
            print(error_msg)
            pytest.skip(error_msg)
            # assert False, f"ERROR with model: {error_msg}"
        except Exception as e:
            error_msg = (
                f"Unexpected error during ONNX model validation for {test_case_name}: "
                f"{type(e).__name__}: {str(e)}"
            )
            pytest.skip(error_msg)

    @classmethod
    def _check_validation_dependency(cls, test_case_data):
        """Check if validation failed for this test case and skip if so"""
        layer_name, config, test_spec = test_case_data
        test_case_id = f"{layer_name}_{test_spec.name}"
        
        if test_case_id in cls._validation_failed_cases:
            pytest.skip(f"Skipping because ONNX validation failed for {layer_name}.{test_spec.name}")
    
    


    @staticmethod
    def _generate_test_id(test_case_tuple):
        """Generate test ID from test case tuple"""
        try:
            layer_name, config, test_spec = test_case_tuple
            return f"{layer_name}_{test_spec.name}"
        except (IndexError, AttributeError):
            return str(test_case_tuple)
        except Exception:
            return "unknown"
    
    def create_model_with_layers(self, layer_types: List[str], 
                               layer_configs: Dict[str, LayerTestConfig]) -> onnx.ModelProto:
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
        input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16, 224, 224])
        output_name = f"{layer_types[-1].lower()}_output_{len(layer_types)-1}"
        output_info = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, [1, 10])
        
        graph = helper.make_graph(
            nodes,
            "test_graph",
            [input_info],
            [output_info],
            initializer=list(all_initializers.values())
        )
        
        return helper.make_model(graph)
    
    # ===== VAlIDATE MODEL =======

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "test_case_data", 
        TestLayerFactory.get_all_test_cases(),
        ids=_generate_test_id.__func__
    )
    def test_factory_models_pass_onnx_validation(self, test_case_data):
        """Test that all factory-created models pass ONNX validation"""
        layer_name, config, test_spec = test_case_data
        test_case_id = f"{layer_name}_{test_spec.name}"
        
        if test_spec.skip_reason:
            pytest.skip(test_spec.skip_reason)
            
        model = config.create_test_model(test_spec)
        test_case_name = f"{layer_name}.{test_spec.name}"
        
        try:
            onnx.checker.check_model(model)
        except onnx.checker.ValidationError as e:
            # Mark this test case as failed for other tests to check
            self._validation_failed_cases.add(test_case_id)
            pytest.fail(
                f"ONNX model validation failed for {test_case_name}. "
                f"Model structure is invalid: {str(e)}"
            )
        except Exception as e:
            # Mark this test case as failed for other tests to check
            self._validation_failed_cases.add(test_case_id)
            pytest.fail(
                f"Unexpected error during ONNX model validation for {test_case_name}: "
                f"{type(e).__name__}: {str(e)}"
            )

    # ===== CHECK_MODEL TESTS =====
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "test_case_data", 
        TestLayerFactory.get_test_cases_by_type(SpecType.VALID),
        ids=_generate_test_id.__func__
    )
    def test_check_model_individual_valid_cases(self, quantizer, test_case_data):
        """Test each individual valid test case"""
        layer_name, config, test_spec = test_case_data

        self._check_validation_dependency(test_case_data)
        
        if test_spec.skip_reason:
            pytest.skip(test_spec.skip_reason)
            
        model = config.create_test_model(test_spec)
        
        try:
            quantizer.check_model(model)
        except (InvalidParamError, UnsupportedOpError) as e:
            pytest.fail(f"Model check failed for {layer_name}.{test_spec.name}: {e}")

    @pytest.mark.unit
    def test_check_model_unsupported_layer_fails(self, quantizer):
        """Test that models with unsupported layers fail validation"""
        # Create model with unsupported operation
        unsupported_node = helper.make_node(
            "UnsupportedOp",
            inputs=["input"],
            outputs=["output"],
            name="unsupported"
        )
        
        graph = helper.make_graph(
            [unsupported_node],
            "test_graph",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16, 224, 224])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10])]
        )
        
        model = helper.make_model(graph)
        
        with pytest.raises(UnsupportedOpError):
            quantizer.check_model(model)
    
    @pytest.mark.unit
    @pytest.mark.parametrize("layer_combination", [
        ["Conv", "Relu"],
        ["Conv", "Relu", "MaxPool"],
        ["Gemm", "Relu"],
        ["Conv", "Reshape", "Gemm"],
        ["Conv", "Flatten", "Gemm"]
    ])
    def test_check_model_multi_layer_passes(self, quantizer, layer_configs, layer_combination):
        """Test that models with multiple supported layers pass validation"""
        model = self.create_model_with_layers(layer_combination, layer_configs)
        # Should not raise any exception
        quantizer.check_model(model)

    # ===== QUANTIZE TESTS =====
    @pytest.mark.unit  
    @pytest.mark.parametrize(
        "test_case_data", 
        TestLayerFactory.get_test_cases_by_type(SpecType.VALID),
        ids=_generate_test_id.__func__
    )
    def test_quantize_individual_valid_cases(self, quantizer, test_case_data):
        """Test quantization for each individual valid test case"""
        layer_name, config, test_spec = test_case_data

        self._check_validation_dependency(test_case_data)
        
        # if test_spec.skip_reason:
        #     pytest.skip(test_spec.skip_reason)
            
        model = config.create_test_model(test_spec)
        node = model.graph.node[0]
        initializer_map = {init.name: init for init in model.graph.initializer}
        
        mock_graph = Mock()
        if node.op_type == "Constant":
            mock_data_node = Mock()
            mock_data_node.input = [node.output[0]]
            mock_graph.node = [mock_data_node]

        scale, scale_base = 2, 10
        rescale = True

        
        result = quantizer.quantize(node, rescale, mock_graph, scale, scale_base, initializer_map)

        if isinstance(result, list):
            assert len(result) > 0, f"Quantize returned empty list for {layer_name}.{test_spec.name}"
            for node_result in result:
                assert isinstance(node_result, onnx.NodeProto), f"Invalid node type returned for {layer_name}.{test_spec.name}"
        else:
            assert isinstance(result, onnx.NodeProto), f"Quantize returned none node for {layer_name}.{test_spec.name}"
        
            assert result is not None, f"Quantize returned None for {layer_name}.{test_spec.name}"

    @pytest.mark.unit  
    @pytest.mark.parametrize(
        "test_case_data", 
        TestLayerFactory.get_test_cases_by_type(SpecType.VALID),
        ids=_generate_test_id.__func__
    )
    def test_quantize_preserves_node_names(self, quantizer, test_case_data):
        """Test quantization for each individual valid test case"""
        layer_name, config, test_spec = test_case_data

        self._check_validation_dependency(test_case_data)
        
        # if test_spec.skip_reason:
        #     pytest.skip(test_spec.skip_reason)
            
        model = config.create_test_model(test_spec)
        node = model.graph.node[0]
        initializer_map = {init.name: init for init in model.graph.initializer}
        
        mock_graph = Mock()
        if node.op_type == "Constant":
            mock_data_node = Mock()
            mock_data_node.input = [node.output[0]]
            mock_graph.node = [mock_data_node]

        scale, scale_base = 2, 10
        rescale = True
        result = quantizer.quantize(node, rescale, mock_graph, scale, scale_base, initializer_map)
        is_node_present = False
        
        def check_node_and_analyze_parameters(node, result_node):
            if node.op_type in result_node.op_type:
                # Assert there are no less attributes in the new node
                assert len(node.attribute) <= len(result_node.attribute)
                # Ensure that each original node's attributes are contained in the new nodes
                for att in node.attribute:
                    assert att.name in [a.name for a in result_node.attribute]

                return True
        
        # Check that result nodes have meaningful names
        if isinstance(result, list):
            for result_node in result:
                assert result_node.name, f"Quantized node missing name for {config.op_type}"
                assert result_node.op_type, f"Quantized node missing op_type for {config.op_type}"

                is_node_present = is_node_present or check_node_and_analyze_parameters(node, result_node)
        else:
            assert result.name, f"Quantized node missing name for {config.op_type}"
            is_node_present = is_node_present or check_node_and_analyze_parameters(node, result)
        assert is_node_present, "Cannot find quantized node relating to prequantized node"

    @pytest.mark.unit  
    @pytest.mark.parametrize("scale_params", [(2, 10), (0, 5)])
    @pytest.mark.parametrize(
        "test_case_data", 
        TestLayerFactory.get_test_cases_by_type(SpecType.VALID),
        ids=_generate_test_id.__func__
    )
    def test_quantize_with_different_scales(self, quantizer, test_case_data, scale_params):
        """Test quantization for each individual valid test case"""
        layer_name, config, test_spec = test_case_data

        self._check_validation_dependency(test_case_data)
        
        # if test_spec.skip_reason:
        #     pytest.skip(test_spec.skip_reason)
            
        model = config.create_test_model(test_spec)
        node = model.graph.node[0]
        initializer_map = {init.name: init for init in model.graph.initializer}
        
        mock_graph = Mock()
        if node.op_type == "Constant":
            mock_data_node = Mock()
            mock_data_node.input = [node.output[0]]
            mock_graph.node = [mock_data_node]

        scale, scale_base = scale_params
        rescale = True
        result = quantizer.quantize(node, rescale, mock_graph, scale, scale_base, initializer_map)
        
        # Should return valid result regardless of scale values
        assert result is not None, f"Quantize returned None for scale={scale}, scale_base={scale_base}"
    
    @pytest.mark.unit  
    @pytest.mark.parametrize("rescale", [True, False])
    @pytest.mark.parametrize(
        "test_case_data", 
        TestLayerFactory.get_test_cases_by_type(SpecType.VALID),
        ids=_generate_test_id.__func__
    )
    def test_quantize_with_different_scales(self, quantizer, test_case_data, rescale):
        """Test quantization for each individual valid test case"""
        layer_name, config, test_spec = test_case_data

        self._check_validation_dependency(test_case_data)
        
        # if test_spec.skip_reason:
        #     pytest.skip(test_spec.skip_reason)
            
        model = config.create_test_model(test_spec)
        node = model.graph.node[0]
        initializer_map = {init.name: init for init in model.graph.initializer}
        
        mock_graph = Mock()
        if node.op_type == "Constant":
            mock_data_node = Mock()
            mock_data_node.input = [node.output[0]]
            mock_graph.node = [mock_data_node]

        scale, scale_base = 2, 10
        
        result = quantizer.quantize(node, rescale, mock_graph, scale, scale_base, initializer_map)
        assert result is not None, f"Quantize failed with rescale={rescale}"
    
    @pytest.mark.unit
    def test_quantize_unsupported_layer_returns_original(self, quantizer):
        """Test that unsupported layers return the original node"""
        mock_graph = Mock()
        scale, scale_base = 2, 10
        rescale = True
        
        unsupported_node = helper.make_node(
            "UnsupportedOp",
            inputs=["input"],
            outputs=["output"],
            name="unsupported"
        )
        
        result = quantizer.quantize(unsupported_node, rescale, mock_graph, scale, scale_base, {})
        
        # Should return original node unchanged
        assert result == unsupported_node

    # ===== INTEGRATION TESTS =====
    
    @pytest.mark.integration
    @pytest.mark.parametrize("layer_combination", [
        ["Conv", "Relu"],
        ["Gemm", "Relu"],
        ["Conv", "MaxPool", "Flatten", "Gemm"]
    ])
    def test_check_then_quantize_workflow(self, quantizer, layer_configs, layer_combination):
        """Test the typical workflow: check model then quantize layers"""
        mock_graph = Mock()
        scale, scale_base = 2, 10
        rescale = True
        
        # Step 1: Create and validate model
        model = self.create_model_with_layers(layer_combination, layer_configs)
        quantizer.check_model(model)  # Should not raise
        
        # Step 2: Quantize each layer
        initializer_map = quantizer.get_initializer_map(model)
        
        for node in model.graph.node:
            result = quantizer.quantize(node, rescale, mock_graph, scale, scale_base, initializer_map)
            assert result is not None, f"Quantization failed for {node.op_type} in combination {layer_combination}"
        
    # TEST ERRORS
    
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "test_case_data", 
        TestLayerFactory.get_test_cases_by_type(SpecType.ERROR),
        ids=_generate_test_id.__func__
    )
    def test_check_model_individual_error_cases(self, quantizer, test_case_data):
        """Test each individual error test case"""
        layer_name, config, test_spec = test_case_data

        self._check_validation_dependency(test_case_data)
        
        if test_spec.skip_reason:
            pytest.skip(test_spec.skip_reason)
            
        model = config.create_test_model(test_spec)
        
        with pytest.raises(test_spec.expected_error) as exc:
            quantizer.check_model(model)

        if isinstance(test_spec.error_match, list):
            for e in test_spec.error_match:
                assert e in str(exc.value)
        else:
            assert test_spec.error_match in str(exc.value)
    
    
    # # ===== TAGGED TESTS =====
    
    # @pytest.mark.unit
    # @pytest.mark.parametrize(
    #     "test_case_data",
    #     TestLayerFactory.get_test_cases_by_tag("transpose"),
    #     ids=_generate_test_id.__func__
    # )
    # def test_transpose_operations(self, quantizer, test_case_data):
    #     """Test operations specifically tagged with 'transpose'"""
    #     layer_name, config, test_spec = test_case_data
    #     model = config.create_test_model(test_spec)
    #     # Implementation specific to transpose tests
    #     quantizer.check_model(model)
    
    # @pytest.mark.unit
    # @pytest.mark.parametrize(
    #     "test_case_data",
    #     TestLayerFactory.get_test_cases_by_tag("invalid_params"),
    #     ids=_generate_test_id.__func__
    # )
    # def test_invalid_parameter_handling(self, quantizer, test_case_data):
    #     """Test handling of invalid parameters across all layers"""
    #     layer_name, config, test_spec = test_case_data
    #     model = config.create_test_model(test_spec)
    #     with pytest.raises(test_spec.expected_error, match=test_spec.error_match):
    #         quantizer.check_model(model)



class TestScalability:
    """Tests (meta) to verify the framework scales with new layers"""
    
    @pytest.mark.unit
    def test_adding_new_layer_config(self):
        """Test that adding new layer configs is straightforward"""
        # Simulate adding a new layer type
        new_layer_config = LayerTestConfig(
            op_type="NewCustomOp",
            valid_inputs=["input", "custom_param"],
            valid_attributes={"custom_attr": 42},
            required_initializers={"custom_param": np.array([1, 2, 3])}
        )
        
        # Verify config can create nodes and initializers
        node = new_layer_config.create_node()
        assert node.op_type == "NewCustomOp"
        assert len(node.input) == 2
        
        initializers = new_layer_config.create_initializers()
        assert "custom_param" in initializers
    
    @pytest.mark.unit
    def test_layer_config_extensibility(self):
        """Test that layer configs consists of all registered handlers"""
        configs = TestLayerFactory.get_layer_configs()
        
        # Verify all expected layers are present
        unsupported = ONNXOpQuantizer().handlers.keys() - set(configs.keys())
        print(unsupported)
        assert unsupported == set(), f"The following layers are not being configured for testing: {unsupported}. Please add configuration in tests/onnx_quantizer_tests/layers/"
        
        # Verify each config has required components
        for layer_type, config in configs.items():
            assert config.op_type == layer_type, "Quantization test config is not supported yet for {} and must be implemented"
            assert isinstance(config.valid_inputs, list), "Quantization test config is not supported yet for {} and must be implemented"
            assert isinstance(config.valid_attributes, dict), "Quantization test config is not supported yet for {} and must be implemented"
            assert isinstance(config.required_initializers, dict), "Quantization test config is not supported yet for {} and must be implemented"


# WORK IN PROGRESS BELOW
@pytest.mark.unit
@pytest.mark.parametrize("attr_name, attr_value, match", [
    ("alpha", 2.0, "alpha value of 2.0 not supported"),
    ("beta", 0.0, "beta value of 0.0 not supported"),
    ("transA", 3, "transA value of 3 not supported"),
    ("transB", -1, "transB value of -1 not supported"),
])
def test_gemm_invalid_params(attr_name, attr_value, match):
    attrs = {
        "alpha": 1.0,
        "beta": 1.0,
        "transA": 0,
        "transB": 0,
    }
    attrs[attr_name] = attr_value

    gemm_node = helper.make_node(
        op_type="Gemm",
        inputs=["input", "gemm_weight", "gemm_bias"],
        outputs=["output"],
        name="gemm_invalid",
        **attrs
    )

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 256])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 128])
    weight = from_array(np.random.randn(256, 128).astype(np.float32), name="gemm_weight")
    bias = from_array(np.random.randn(128).astype(np.float32), name="gemm_bias")

    graph = helper.make_graph(
        nodes=[gemm_node],
        name="gemm_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight, bias]
    )

    model = helper.make_model(graph)
    quantizer = ONNXOpQuantizer()
    initializer_map = quantizer.get_initializer_map(model)

    with pytest.raises(InvalidParamError, match=match):
        quantizer.check_layer(gemm_node, initializer_map)

@pytest.mark.unit
@pytest.mark.parametrize("transA, transB", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_gemm_transpose_combinations_supported(transA, transB):
    gemm_node = helper.make_node(
        op_type="Gemm",
        inputs=["input", "gemm_weight", "gemm_bias"],
        outputs=["output"],
        name=f"gemm_transA_{transA}_transB_{transB}",
        alpha=1.0,
        beta=1.0,
        transA=transA,
        transB=transB
    )

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 256])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 128])

    weight = from_array(np.random.randn(256, 128).astype(np.float32), name="gemm_weight")
    bias = from_array(np.random.randn(128).astype(np.float32), name="gemm_bias")

    graph = helper.make_graph(
        nodes=[gemm_node],
        name="gemm_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight, bias]
    )

    model = helper.make_model(graph)
    quantizer = ONNXOpQuantizer()
    initializer_map = quantizer.get_initializer_map(model)

    # Should not raise
    quantizer.check_layer(gemm_node, initializer_map)