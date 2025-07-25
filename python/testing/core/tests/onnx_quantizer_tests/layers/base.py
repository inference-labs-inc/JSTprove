import onnx
import numpy as np
from onnx import helper, numpy_helper
from typing import Dict, List, Any
from abc import ABC, abstractmethod


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


class BaseLayerConfigProvider(ABC):
    """Abstract base class for layer config providers"""
    
    @abstractmethod
    def get_config(self) -> LayerTestConfig:
        """Return the test configuration for this layer"""
        pass
    
    @property
    @abstractmethod
    def layer_name(self) -> str:
        """Return the layer name/op_type"""
        pass