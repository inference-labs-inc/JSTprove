import os
import pytest
import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, MagicMock

# Import your classes (adjust imports as needed)
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


# class TestLayerFactory:
#     """Factory for creating test configurations for different layer types"""
    
#     @staticmethod
#     def get_layer_configs() -> Dict[str, LayerTestConfig]:
#         """Get test configurations for all supported layers"""
#         return {
#             "Conv": LayerTestConfig(
#                 op_type="Conv",
#                 valid_inputs=["input", "conv_weight", "conv_bias"],
#                 valid_attributes={
#                     "strides": [1, 1],
#                     "kernel_shape": [3, 3],
#                     "dilations": [1, 1],
#                     "pads": [1, 1, 1, 1]
#                 },
#                 required_initializers={
#                     "conv_weight": np.random.randn(32, 16, 3, 3),
#                     "conv_bias": np.random.randn(32)
#                 }
#             ),
#             "Gemm": LayerTestConfig(
#                 op_type="Gemm",
#                 valid_inputs=["input", "gemm_weight", "gemm_bias"],
#                 valid_attributes={
#                     "alpha": 1.0,
#                     "beta": 1.0,
#                     "transA": 0,
#                     "transB": 0
#                 },
#                 required_initializers={
#                     "gemm_weight": np.random.randn(128, 256),
#                     "gemm_bias": np.random.randn(128)
#                 }
#             ),
#             "Relu": LayerTestConfig(
#                 op_type="Relu",
#                 valid_inputs=["input"],
#                 valid_attributes={},
#                 required_initializers={}
#             ),
#             "Reshape": LayerTestConfig(
#                 op_type="Reshape",
#                 valid_inputs=["input", "shape"],
#                 valid_attributes={},
#                 required_initializers={
#                     "shape": np.array([1, -1])
#                 }
#             ),
#             "MaxPool": LayerTestConfig(
#                 op_type="MaxPool",
#                 valid_inputs=["input"],
#                 valid_attributes={
#                     "kernel_shape": [2, 2],
#                     "strides": [2, 2],
#                     "dilations": 1,
#                     "pads": 1
#                 },
#                 required_initializers={}
#             ),
#             "Flatten": LayerTestConfig(
#                 op_type="Flatten",
#                 valid_inputs=["input"],
#                 valid_attributes={"axis": 1},
#                 required_initializers={}
#             ),
#             "Constant": LayerTestConfig(
#                 op_type="Constant",
#                 valid_inputs=[],
#                 valid_attributes={
#                     "value": numpy_helper.from_array(np.array([1.0]), name="const_value")
#                 },
#                 required_initializers={}
#             )
#         }


class TestONNXOpQuantizer:
    """Generic unit tests for ONNX Op Quantizer"""
    
    @pytest.fixture
    def quantizer(self):
        """Create quantizer instance"""
        return ONNXOpQuantizer()
    
    @pytest.fixture
    def layer_configs(self):
        """Get all layer configurations"""
        return TestLayerFactory.get_layer_configs()
    
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

    # ===== CHECK_MODEL TESTS =====
    
    @pytest.mark.unit
    @pytest.mark.parametrize("config", TestLayerFactory.get_layer_configs().values(), 
                           ids=TestLayerFactory.get_layer_configs().keys())
    def test_check_model_single_layer_passes(self, quantizer, config):
        """Test that models with single supported layers pass validation"""
        model = self.create_model_with_layers([config.op_type], {config.op_type: config})
        try:
            quantizer.check_model(model)
        except InvalidParamError as e:
            pytest.fail(f"Model check failed for op_type={config.op_type}, {e.message}")
        except UnsupportedOpError as e:
            pytest.fail(f"Model check failed as op_type={config.op_type} is unsupported")
    
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
    @pytest.mark.parametrize("config", TestLayerFactory.get_layer_configs().values(),
                           ids=TestLayerFactory.get_layer_configs().keys())
    def test_quantize_single_layer_returns_nodes(self, quantizer, config):
        """Test that quantizing layers returns appropriate node structures"""
        node = config.create_node()
        initializer_map = config.create_initializers()

        mock_graph = Mock()
        if node.op_type == "Constant":
            mock_data_node = Mock()
            mock_data_node.input = [f"{config.op_type.lower()}_output"]
            mock_graph.node = [mock_data_node]
        scale, scale_base = 2, 10
        rescale = True
        
        
        result = quantizer.quantize(node, rescale, mock_graph, scale, scale_base, initializer_map)
        
        # Result should be either a single node or list of nodes
        if isinstance(result, list):
            assert len(result) > 0, f"Quantize returned empty list for {config.op_type}"
            for node_result in result:
                assert isinstance(node_result, onnx.NodeProto), f"Invalid node type returned for {config.op_type}"
        else:
            assert isinstance(result, onnx.NodeProto), f"Invalid result type for {config.op_type}"
    
    @pytest.mark.unit
    @pytest.mark.parametrize("config", TestLayerFactory.get_layer_configs().values(),
                           ids=TestLayerFactory.get_layer_configs().keys())
    def test_quantize_preserves_node_names(self, quantizer, config):
        """Test that quantization preserves or properly transforms node names"""

        original_name = f"test_{config.op_type.lower()}_node"
        node = config.create_node()
        node.name = original_name
        initializer_map = config.create_initializers()

        mock_graph = Mock()
        if node.op_type == "Constant":
            mock_data_node = Mock()
            mock_data_node.input = [f"{config.op_type.lower()}_output"]
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
    @pytest.mark.parametrize("config", TestLayerFactory.get_layer_configs().values(),
                           ids=TestLayerFactory.get_layer_configs().keys())
    def test_quantize_with_different_scales(self, quantizer, config, scale_params):
        """Test quantization with different scale parameters"""
        node = config.create_node()
        initializer_map = config.create_initializers()

        mock_graph = Mock()
        if node.op_type == "Constant":
            mock_data_node = Mock()
            mock_data_node.input = [f"{config.op_type.lower()}_output"]
            mock_graph.node = [mock_data_node]
        rescale = True
        scale, scale_base = scale_params
        
        result = quantizer.quantize(node, rescale, mock_graph, scale, scale_base, initializer_map)
        
        # Should return valid result regardless of scale values
        assert result is not None, f"Quantize returned None for scale={scale}, scale_base={scale_base}"
    
    @pytest.mark.unit
    @pytest.mark.parametrize("rescale", [True, False])
    @pytest.mark.parametrize("config", TestLayerFactory.get_layer_configs().values(),
                           ids=TestLayerFactory.get_layer_configs().keys())
    def test_quantize_with_rescale_variations(self, quantizer, config, rescale):
        """Test quantization with different rescale settings"""
        node = config.create_node()
        initializer_map = config.create_initializers()

        mock_graph = Mock()
        if node.op_type == "Constant":
            mock_data_node = Mock()
            mock_data_node.input = [f"{config.op_type.lower()}_output"]
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

@pytest.mark.unit
def test_conv3d_should_raise_invalid_param_error():
    """Conv3D is unsupported and should raise InvalidParamError due to 5D weights."""
    conv_node = helper.make_node(
        op_type="Conv",
        inputs=["input", "conv_weight", "conv_bias"],
        outputs=["output"],
        name="conv3d",
        kernel_shape=[3, 3, 3],
        strides=[1, 1, 1],
        dilations=[1, 1, 1],
        pads=[1, 1, 1, 1, 1, 1]
    )

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 16, 10, 224, 224])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32, 10, 224, 224])

    weight_array = np.random.randn(32, 16, 3, 3, 3).astype(np.float32)
    bias_array = np.random.randn(32).astype(np.float32)

    weight_initializer = from_array(weight_array, name="conv_weight")
    bias_initializer = from_array(bias_array, name="conv_bias")

    graph = helper.make_graph(
        nodes=[conv_node],
        name="conv3d_graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_initializer, bias_initializer],
    )

    model = helper.make_model(graph)
    quantizer = ONNXOpQuantizer()

    initializer_map = quantizer.get_initializer_map(model)

    with pytest.raises(InvalidParamError, match="3D|unsupported|rank"):
        quantizer.check_layer(conv_node, initializer_map)