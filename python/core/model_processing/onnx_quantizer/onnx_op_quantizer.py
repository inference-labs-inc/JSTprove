from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

import onnx

from python.core.model_processing.onnx_quantizer.exceptions import (
    HandlerImplementationError,
    MissingHandlerError,
    UnsupportedOpError,
)
from python.core.model_processing.onnx_quantizer.layers.add import AddQuantizer
from python.core.model_processing.onnx_quantizer.layers.base import (
    PassthroughQuantizer,
    ScaleConfig,
)
from python.core.model_processing.onnx_quantizer.layers.batchnorm import (
    BatchnormQuantizer,
)
from python.core.model_processing.onnx_quantizer.layers.clip import ClipQuantizer
from python.core.model_processing.onnx_quantizer.layers.constant import (
    ConstantQuantizer,
)
from python.core.model_processing.onnx_quantizer.layers.conv import ConvQuantizer
from python.core.model_processing.onnx_quantizer.layers.gemm import GemmQuantizer
from python.core.model_processing.onnx_quantizer.layers.max import MaxQuantizer
from python.core.model_processing.onnx_quantizer.layers.maxpool import MaxpoolQuantizer
from python.core.model_processing.onnx_quantizer.layers.min import MinQuantizer
from python.core.model_processing.onnx_quantizer.layers.mul import MulQuantizer
from python.core.model_processing.onnx_quantizer.layers.relu import ReluQuantizer
from python.core.model_processing.onnx_quantizer.layers.sub import SubQuantizer


class ONNXOpQuantizer:
    """
    Registry for ONNX operator quantizers.
    This should be used to obtain the quantized
    layer based on any provided operation of that layer type

    Attributes
    ----------
    handlers : Dict[str, Callable]
        Maps ONNX op_type strings to quantizer handler instances.
    new_initializers : List[onnx.TensorProto]
        A list of newly created ONNX initializers
        (weights or biases typically) during quantization.
        This is shared with handlers that may add new constants.

    Methods
    -------
    register(op_type, handler)
        Registers a handler for an ONNX op_type.
    quantize(node, rescale, graph, scale_exponent, scale_base, initializer_map)
        Apply quantization to a specific ONNX node using its registered handler.
    check_model(model)
        Ensure all operations in the model are supported and validate
        each layer's parameters are valid and supported.
    check_layer(node, initializer_map)
        Validate a single ONNX node using its handler's check_supported method,
        to check that the given layers parameters and structure is supported.
    get_initializer_map(model)
        Build a {name: TensorProto} mapping for the model's initializers.
    """

    def __init__(self: ONNXOpQuantizer) -> None:
        self.handlers: dict[
            str,
            Callable[
                [onnx.NodeProto, bool],
                onnx.NodeProto | list[onnx.NodeProto],
            ],
        ] = {}
        self.new_initializers = []

        # Register handlers
        self.register("Add", AddQuantizer(self.new_initializers))
        self.register("Clip", ClipQuantizer(self.new_initializers))
        self.register("Sub", SubQuantizer(self.new_initializers))
        self.register("Mul", MulQuantizer(self.new_initializers))
        self.register("Conv", ConvQuantizer(self.new_initializers))
        self.register("Relu", ReluQuantizer())
        self.register("Reshape", PassthroughQuantizer())
        self.register("Gemm", GemmQuantizer(self.new_initializers))
        self.register("Constant", ConstantQuantizer())
        self.register("MaxPool", MaxpoolQuantizer())
        self.register("Flatten", PassthroughQuantizer())
        self.register("Max", MaxQuantizer(self.new_initializers))
        self.register("Min", MinQuantizer(self.new_initializers))
        self.register("BatchNormalization", BatchnormQuantizer(self.new_initializers))

    def register(
        self: ONNXOpQuantizer,
        op_type: str,
        handler: Callable[
            [onnx.NodeProto, bool],
            onnx.NodeProto | list[onnx.NodeProto],
        ],
    ) -> None:
        """Register a quantizer handler for a given ONNX op_type.

        Args:
            op_type (str): Name of the ONNX operator type (e.g., "Conv", "Relu").
            handler (Callable[[onnx.NodeProto, bool],
                    Union[onnx.NodeProto, list[onnx.NodeProto]]]):
                - Handler instance implementing `quantize()`
                    (and optionally `check_supported()`).

        Raises:
            HandlerImplementationError: If handler has not properly implemented
                `quantize` method
        """
        if not hasattr(handler, "quantize") or not callable(handler.quantize):
            raise HandlerImplementationError(op_type, "Missing 'quantize' method.")

        self.handlers[op_type] = handler

    def quantize(  # noqa: PLR0913
        self: ONNXOpQuantizer,
        node: onnx.NodeProto,
        graph: onnx.GraphProto,
        scale_exponent: int,
        scale_base: int,
        initializer_map: dict[str, onnx.TensorProto],
        *,
        rescale: bool = True,
    ) -> onnx.NodeProto | list[onnx.NodeProto]:
        """Quantize an ONNX node using its registered handler.

        Args:
            node (onnx.NodeProto): The ONNX node to quantize.
            rescale (bool): Whether to apply rescaling.
            graph (onnx.GraphProto): The ONNX graph containing the node.
            scale_exponent (int): Quantization scale value.
                The scaling becomes scale_base**scale_exponent.
            scale_base (int): Base for the quantization scale.
                The scaling becomes scale_base**scale.
            initializer_map (dict[str, onnx.TensorProto]):
                Mapping of initializer names (typically weights and biases) to tensors.

        Returns:
            Union[onnx.NodeProto, List[onnx.NodeProto]]: The quantized node +
                any additional nodes created in the process.
        """
        handler = self.handlers.get(node.op_type)
        if handler:
            result = handler.quantize(
                node=node,
                graph=graph,
                scale_config=ScaleConfig(scale_exponent, scale_base, rescale),
                initializer_map=initializer_map,
            )
            if isinstance(result, onnx.NodeProto):
                return [result]
            return result

        raise UnsupportedOpError(node.op_type)

    def check_model(self: ONNXOpQuantizer, model: onnx.ModelProto) -> None:
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

    def check_layer(
        self: ONNXOpQuantizer,
        node: onnx.NodeProto,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> None:
        """
        Check an individual node using its handler.
        Parameters for the node will be checked that they
        meet the supported parameter requirements.

        Args:
            node (onnx.NodeProto): The node to check.
            initializer_map (dict[str, onnx.TensorProto]): Mapping of initializer names
                to tensor typically used in weights and biases.

        Raises:
            MissingHandlerError: If no handler is registered for the given node.
        """
        handler = self.handlers.get(node.op_type)
        if not handler:
            raise MissingHandlerError(node.op_type)

        if hasattr(handler, "check_supported") and callable(handler.check_supported):
            handler.check_supported(node, initializer_map)

    def get_initializer_map(
        self: ONNXOpQuantizer,
        model: onnx.ModelProto,
    ) -> dict[str, onnx.TensorProto]:
        """Build a dictionary mapping initializer names to tensors in graph.

        Args:
            model (onnx.ModelProto): The ONNX model.

        Returns:
            dict[str, onnx.TensorProto]: Map from initializer name to tensors in graph.
        """
        return {init.name: init for init in model.graph.initializer}

    def apply_pre_analysis_transforms(
        self: ONNXOpQuantizer,
        model: onnx.ModelProto,
        scale_exponent: int,
        scale_base: int,
    ) -> onnx.ModelProto:
        """
        Give each registered handler a chance to rewrite the model before analysis.
        """
        graph = model.graph
        initializer_map = self.get_initializer_map(model)

        # We allow handlers to modify graph in-place.
        # (Nodes may be replaced, removed, or new nodes added.)
        for node in list(graph.node):
            handler = self.handlers.get(node.op_type)
            if handler and hasattr(handler, "pre_analysis_transform"):
                handler.pre_analysis_transform(
                    node,
                    graph,
                    initializer_map,
                    scale_exponent=scale_exponent,
                    scale_base=scale_base,
                )
            # Refresh map if transforms may add initializers
            initializer_map = self.get_initializer_map(model)

        return model
