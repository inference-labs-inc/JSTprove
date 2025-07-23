import numpy as np
import onnx
from onnx import helper, numpy_helper
from typing import Callable, Dict, List, Union

from python.testing.core.utils.onnx_helpers import create_quantized_initializer, extract_attributes, replace_input_references
from onnx.numpy_helper import to_array, from_array

from python.testing.core.utils.onnx_quantizer.layers.base import PassthroughQuantizer
from python.testing.core.utils.onnx_quantizer.layers.constant import ConstantQuantizer
from python.testing.core.utils.onnx_quantizer.layers.conv import ConvQuantizer
from python.testing.core.utils.onnx_quantizer.layers.gemm import GemmQuantizer
from python.testing.core.utils.onnx_quantizer.layers.maxpool import MaxpoolQuantizer
from python.testing.core.utils.onnx_quantizer.layers.relu import ReluQuantizer

from python.testing.core.utils.onnx_quantizer.exceptions import UnsupportedOpError


class ONNXOpQuantizer:
    def __init__(self):
        self.handlers: Dict[str, Callable[[onnx.NodeProto, bool], Union[onnx.NodeProto, List[onnx.NodeProto]]]] = {}
        self.new_initializers = [] 

        # Register handlers
        self.register("Conv", ConvQuantizer(self.new_initializers))
        self.register("Relu", ReluQuantizer()) # might work with quantize_passthrough instead for some models
        self.register("Reshape", PassthroughQuantizer())
        self.register("Gemm", GemmQuantizer(self.new_initializers))
        self.register("Constant", ConstantQuantizer())
        self.register("MaxPool", MaxpoolQuantizer())
        self.register("Flatten", PassthroughQuantizer())

    def register(self, op_type: str, handler: Callable[[onnx.NodeProto, bool], Union[onnx.NodeProto, List[onnx.NodeProto]]]):
        self.handlers[op_type] = handler

    def quantize(self, node: onnx.NodeProto, rescale: bool, graph: onnx.GraphProto, scale: int, scale_base: int, initializer_map: dict[str, onnx.TensorProto]) -> Union[onnx.NodeProto, List[onnx.NodeProto]]:
        handler = self.handlers.get(node.op_type)
        if handler:
            return handler.quantize(node, rescale, graph, scale, scale_base, initializer_map)
        else:
            print(f"⚠️ No quantizer implemented for op_type: {node.op_type}")
            return node
    
    # TODO extend this beyond just supported layers, but also supported parameters - gemm alpha = 1, beta = 1,
    def check_model(self, model: onnx.ModelProto) -> None:
        initializer_map = self.get_initializer_map(model)

        model_ops = {node.op_type for node in model.graph.node}
        unsupported = model_ops - self.handlers.keys()

        if unsupported:
            raise UnsupportedOpError(unsupported)
        
        # Call check_layer on each node (e.g., for param validation)
        for node in model.graph.node:
            self.check_layer(node, initializer_map)
        
    def check_layer(self, node: onnx.NodeProto, initializer_map: dict[str, onnx.TensorProto]) -> None:
        handler = self.handlers.get(node.op_type)
        if not handler:
            raise ValueError(f"No handler registered for op: {node.op_type}")

        if hasattr(handler, "check_supported") and callable(handler.check_supported):
            handler.check_supported(node, initializer_map)

    def get_initializer_map(self, model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
        return {init.name: init for init in model.graph.initializer}