import onnx
from typing import Callable, Dict, List, Union

from python.core.model_processing.onnx_quantizer.layers.base import PassthroughQuantizer
from python.core.model_processing.onnx_quantizer.layers.constant import ConstantQuantizer
from python.core.model_processing.onnx_quantizer.layers.conv import ConvQuantizer
from python.core.model_processing.onnx_quantizer.layers.gemm import GemmQuantizer
from python.core.model_processing.onnx_quantizer.layers.maxpool import MaxpoolQuantizer
from python.core.model_processing.onnx_quantizer.layers.relu import ReluQuantizer

from python.core.model_processing.onnx_quantizer.exceptions import UnsupportedOpError


class ONNXOpQuantizer:
    """
    Registry for ONNX operator quantizers. This should be used to obtain the quantized layer based on any provided operation of that layer type

    Attributes
    ----------
    handlers : Dict[str, Callable]
        Maps ONNX op_type strings to quantizer handler instances.
    new_initializers : List[onnx.TensorProto]
        A list of newly created ONNX initializers (weights or biases typically) during quantization.
        This is shared with handlers that may add new constants.

    Methods
    -------
    register(op_type, handler)
        Registers a handler for an ONNX op_type.
    quantize(node, rescale, graph, scale_exponent, scale_base, initializer_map)
        Apply quantization to a specific ONNX node using its registered handler.
    check_model(model)
        Ensure all operations in the model are supported and validate each layer's parameters are valid and supported.
    check_layer(node, initializer_map)
        Validate a single ONNX node using its handler's check_supported method, to check that the given layers parameters and structure is supported.
    get_initializer_map(model)
        Build a {name: TensorProto} mapping for the model's initializers.
    """
    def __init__(self):
        self.handlers: Dict[str, Callable[[onnx.NodeProto, bool], Union[onnx.NodeProto, List[onnx.NodeProto]]]] = {}
        self.new_initializers = [] 

        # Register handlers
        self.register("Conv", ConvQuantizer(self.new_initializers))
        self.register("Relu", ReluQuantizer()) 
        self.register("Reshape", PassthroughQuantizer())
        self.register("Gemm", GemmQuantizer(self.new_initializers))
        self.register("Constant", ConstantQuantizer())
        self.register("MaxPool", MaxpoolQuantizer())
        self.register("Flatten", PassthroughQuantizer())

    def register(self, op_type: str, handler: Callable[[onnx.NodeProto, bool], Union[onnx.NodeProto, List[onnx.NodeProto]]]):
        """Register a quantizer handler for a given ONNX op_type.

        Args:
            op_type (str): Name of the ONNX operator type (e.g., "Conv", "Relu").
            handler (Callable[[onnx.NodeProto, bool], Union[onnx.NodeProto, List[onnx.NodeProto]]]): 
                - Handler instance implementing `quantize()` (and optionally `check_supported()`).
        """
        self.handlers[op_type] = handler

    def quantize(self, node: onnx.NodeProto, rescale: bool, graph: onnx.GraphProto, scale_exponent: int, scale_base: int, initializer_map: dict[str, onnx.TensorProto]) -> Union[onnx.NodeProto, List[onnx.NodeProto]]:
        """Quantize an ONNX node using its registered handler.

        Args:
            node (onnx.NodeProto): The ONNX node to quantize.
            rescale (bool): Whether to apply rescaling.
            graph (onnx.GraphProto): The ONNX graph containing the node.
            scale_exponent (int): Quantization scale value. The scaling becomes scale_base**scale_exponent.
            scale_base (int): Base for the quantization scale. The scaling becomes scale_base**scale.
            initializer_map (dict[str, onnx.TensorProto]): Mapping of initializer names (typically weights and biases) to tensors.

        Returns:
            Union[onnx.NodeProto, List[onnx.NodeProto]]: The quantized node + any additional nodes created in the process.
        """
        handler = self.handlers.get(node.op_type)
        if handler:
            return handler.quantize(node, rescale, graph, scale_exponent, scale_base, initializer_map)
        
        print(f"⚠️ No quantizer implemented for op_type: {node.op_type}")
        return node
    
    def check_model(self, model: onnx.ModelProto) -> None:
        """Verify that all nodes in the model are supported and valid.

        Args:
            model (onnx.ModelProto): The ONNX model to check.

        Raises:
            UnsupportedOpError: If the model contains unsupported operators.
        """
        initializer_map = self.get_initializer_map(model)

        model_ops = {node.op_type for node in model.graph.node}
        unsupported = model_ops - self.handlers.keys()

        if unsupported:
            raise UnsupportedOpError(unsupported)
        
        # Call check_layer on each node (e.g., for param validation)
        for node in model.graph.node:
            self.check_layer(node, initializer_map)
        
    def check_layer(self, node: onnx.NodeProto, initializer_map: dict[str, onnx.TensorProto]) -> None:
        """
        Check an individual node using its handler. 
        Parameters for the node will be checked that they meet the supported parameter requirements.

        Args:
            node (onnx.NodeProto): The node to check.
            initializer_map (dict[str, onnx.TensorProto]): Mapping of initializer names to tensor typically used in weights and biases.

        Raises:
            ValueError: If no handler is registered for the given node.
        """
        handler = self.handlers.get(node.op_type)
        if not handler:
            raise ValueError(f"No handler registered for op: {node.op_type}")

        if hasattr(handler, "check_supported") and callable(handler.check_supported):
            handler.check_supported(node, initializer_map)

    def get_initializer_map(self, model: onnx.ModelProto) -> dict[str, onnx.TensorProto]:
        """Build a dictionary mapping initializer names to tensors in graph.

        Args:
            model (onnx.ModelProto): The ONNX model.

        Returns:
            dict[str, onnx.TensorProto]: Mapping from initializer name to tensors in graph.
        """
        return {init.name: init for init in model.graph.initializer}
