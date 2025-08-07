import onnx
import numpy as np
from onnx import helper, numpy_helper, TensorProto
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum


class SpecType(Enum):
    """Types of test specifications that can be run"""
    VALID = "valid"
    ERROR = "error"
    EDGE_CASE = "edge_case"


@dataclass
class TestSpec:
    """Individual test specification that can be applied to a LayerTestConfig"""
    name: str
    spec_type: SpecType
    description: str = ""
    
    # Overrides for the base config
    attr_overrides: Dict[str, Any] = field(default_factory=dict)
    initializer_overrides: Dict[str, np.ndarray] = field(default_factory=dict)
    input_overrides: List[str] = field(default_factory=list)
    
    # Error test specific
    expected_error: Optional[type] = None
    error_match: Optional[str] = None
    
    # Custom validation
    custom_validator: Optional[Callable] = None
    
    # Test metadata
    tags: List[str] = field(default_factory=list)
    skip_reason: Optional[str] = None

    # Omit attributes
    omit_attrs: List[str] = field(default_factory=list)

    
    # Remove __post_init__ validation - we'll validate in the builder instead


class LayerTestConfig:
    """Enhanced configuration class for layer-specific test data"""
    
    def __init__(self, op_type: str, valid_inputs: List[str], 
                 valid_attributes: Dict[str, Any], 
                 required_initializers: Dict[str, np.ndarray],
                 input_shapes: Optional[Dict[str, List[int]]] = None,
                 output_shapes: Optional[Dict[str, List[int]]] = None):
        self.op_type = op_type
        self.valid_inputs = valid_inputs
        self.valid_attributes = valid_attributes
        self.required_initializers = required_initializers
        self.input_shapes = input_shapes or {"input": [1, 16, 224, 224]}
        self.output_shapes = output_shapes or {f"{op_type.lower()}_output": [1, 10]}
    
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
    
    def create_initializers(self, **initializer_overrides) -> Dict[str, onnx.TensorProto]:
        """Create initializer tensors for this layer"""
        initializers = {}
        combined_inits = {**self.required_initializers, **initializer_overrides}
        for name, data in combined_inits.items():
            tensor = numpy_helper.from_array(data.astype(np.float32), name=name)
            initializers[name] = tensor
        return initializers
    
    def create_test_model(self, test_spec: TestSpec) -> onnx.ModelProto:
        """Create a complete model for a specific test case"""
        # Apply overrides from test spec
        inputs = test_spec.input_overrides or self.valid_inputs
        attrs = {**self.valid_attributes, **test_spec.attr_overrides}
        
        node = helper.make_node(
            self.op_type,
            inputs=inputs,
            outputs=[f"{self.op_type.lower()}_output"],
            name=f"test_{self.op_type.lower()}_{test_spec.name}",
            **attrs
        )
        
        initializers = self.create_initializers(**test_spec.initializer_overrides)
        
        # Create input/output tensor info
        graph_inputs = [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            for name, shape in self.input_shapes.items()
        ]
        
        graph_outputs = [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)
            for name, shape in self.output_shapes.items()
        ]
        
        graph = helper.make_graph(
            nodes=[node],
            name=f"{self.op_type.lower()}_test_graph_{test_spec.name}",
            inputs=graph_inputs,
            outputs=graph_outputs,
            initializer=list(initializers.values())
        )

        attrs = {**self.valid_attributes, **test_spec.attr_overrides}
        
        # Remove omitted attributes
        for key in getattr(test_spec, "omit_attrs", []):
            attrs.pop(key, None)

        return helper.make_model(graph)


class TestSpecBuilder:
    """Builder for creating test specifications"""
    
    def __init__(self, name: str, spec_type: SpecType):
        self._spec = TestSpec(name=name, spec_type=spec_type)
    
    def description(self, desc: str) -> 'TestSpecBuilder':
        self._spec.description = desc
        return self
    
    def override_attrs(self, **attrs) -> 'TestSpecBuilder':
        self._spec.attr_overrides.update(attrs)
        return self
    
    def omit_attrs(self, *attrs: str) -> 'TestSpecBuilder':
        self._spec.omit_attrs.extend(attrs)
        return self

    
    def override_initializer(self, name: str, data: np.ndarray) -> 'TestSpecBuilder':
        self._spec.initializer_overrides[name] = data
        return self
    
    def override_inputs(self, *inputs: str) -> 'TestSpecBuilder':
        self._spec.input_overrides = list(inputs)
        return self
    
    def expects_error(self, error_type: type, match: str = None) -> 'TestSpecBuilder':
        if self._spec.spec_type != SpecType.ERROR:
            raise ValueError("expects_error can only be used with ERROR spec type")
        self._spec.expected_error = error_type
        self._spec.error_match = match
        return self
    
    def tags(self, *tags: str) -> 'TestSpecBuilder':
        self._spec.tags.extend(tags)
        return self
    
    def skip(self, reason: str) -> 'TestSpecBuilder':
        self._spec.skip_reason = reason
        return self
    
    def build(self) -> TestSpec:
        # Validate before building
        if self._spec.spec_type == SpecType.ERROR and not self._spec.expected_error:
            raise ValueError(f"Error test {self._spec.name} must specify expected_error using .expects_error()")
        return self._spec

# Convenience functions
def valid_test(name: str) -> TestSpecBuilder:
    return TestSpecBuilder(name, SpecType.VALID)

def error_test(name: str) -> TestSpecBuilder:
    return TestSpecBuilder(name, SpecType.ERROR)

def edge_case_test(name: str) -> TestSpecBuilder:
    return TestSpecBuilder(name, SpecType.EDGE_CASE)

class BaseLayerConfigProvider(ABC):
    """Abstract base class for layer config providers"""
    
    @abstractmethod
    def get_config(self) -> LayerTestConfig:
        """Return the base configuration for this layer"""
        pass
    
    @property
    @abstractmethod
    def layer_name(self) -> str:
        """Return the layer name/op_type"""
        pass
    
    def get_test_specs(self) -> List[TestSpec]:
        """Return test specifications for this layer (override for custom tests)"""
        return []
    
    def get_valid_test_specs(self) -> List[TestSpec]:
        """Get only valid test specifications"""
        return [spec for spec in self.get_test_specs() if spec.test_type == SpecType.VALID]
    
    def get_error_test_specs(self) -> List[TestSpec]:
        """Get only error test specifications"""
        return [spec for spec in self.get_test_specs() if spec.test_type == SpecType.ERROR]
