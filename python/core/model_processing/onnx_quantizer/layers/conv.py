import numpy as np
import onnx
from onnx import numpy_helper
from typing import List

from python.core.model_processing.onnx_custom_ops.onnx_helpers import extract_attributes

from python.core.model_processing.onnx_quantizer.exceptions import InvalidParamError
from python.core.model_processing.onnx_quantizer.layers.base import BaseOpQuantizer

class ConvQuantizer(BaseOpQuantizer):
    """
    Quantizer for ONNX Conv layers.

    - Replaces standard Conv with Int64Conv from the `ai.onnx.contrib` domain and makes relevant additional changes to the graph.
    - Validates that all required Conv parameters are present.
    """
    def __init__(self, new_initializers):
        self.new_initializers = new_initializers

    def quantize(
        self,
        node: onnx.NodeProto,
        rescale: bool,
        graph: onnx.GraphProto,
        scale_exponent: int,
        scale_base: int,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> List[onnx.NodeProto]:
        """
        Quantize a Conv node by:
        1. Quantizing its weights and bias.
        2. Adding a scale constant.
        3. Replacing it with an Int64Conv node.

        Args:
            node (onnx.NodeProto): The node to quantize.
            rescale (bool): Whether rescaling is enabled (Doesnt have an affect on this op type)
            graph (onnx.GraphProto): The ONNX graph.
            scale_exponent (int): Scale exponent.
            scale_base (int): The base of scaling.
            initializer_map (dict[str, onnx.TensorProto]): Map of initializer names to tensor data.

        Returns:
            List[onnx.NodeProto]: A list of ONNX nodes (quantized and any auxiliary nodes).
        """
        nodes = []
        output_name = f"{node.name}_int"

        nodes, node.input[:] = self.add_nodes_w_and_b(node = node, scale_exponent = scale_exponent, scale_base = scale_base, initializer_map = initializer_map, graph = graph)
        attrs = extract_attributes(node)
        attrs.setdefault("group", 1)
        attrs.setdefault("auto_pad", "NOTSET")
        # for attr in node.attribute:
        #     print(f"{attr.name}: type={attr.type} ({onnx.AttributeProto.AttributeType.Name(attr.type)})")
            
        attrs["rescale"] = int(rescale)

        scale_value = scale_base ** scale_exponent
        
        # TODO make this constant to all layers
        # Create scale constant 
        scale_const_name = f"{output_name}_scaler"
        scale_tensor = numpy_helper.from_array(
            np.array([scale_value], dtype=np.int64), name=scale_const_name
        )
        self.new_initializers.append(scale_tensor)
        node.input.append(scale_const_name)
        int64_conv_node = onnx.helper.make_node(
                                        "Int64Conv",
                                        inputs=node.input,
                                        outputs=node.output,  # preserve original output name
                                        name=node.name,
                                        domain="ai.onnx.contrib",
                                        **attrs
                                    )

        nodes.append(int64_conv_node)
        return nodes
    
    def check_supported(self, node: onnx.NodeProto, initializer_map: dict[str, onnx.TensorProto]) -> None:
        """
        Perform high-level validation to ensure that this Conv node
        can be quantized safely.

        Args:
            node (onnx.NodeProto): ONNX node to be checked
            initializer_map (dict[str, onnx.TensorProto]): Initializer map (name of weight or bias and tensor)

        Raises:
            InvalidParamError: If any requirement is not met.
        """
        if len(node.input) < 2:
            raise InvalidParamError(node.name, node.op_type, f"Expected at least 2 inputs (input, weights), got {len(node.input)}")
        
        # TODO currently requires bias layer
        if len(node.input) < 3:
            raise InvalidParamError(node.name, node.op_type, f"Expected at least 3 inputs (input, weights, bias), got {len(node.input)}")
        
        self.check_supported_shape(node, initializer_map)
        self.check_all_params_exist(node)

    def check_all_params_exist(self, node: onnx.NodeProto) -> None:
        """Verify that all required Conv attributes are present.

        Args:
            node (onnx.NodeProto): The Conv node being validated.

        Raises:
            InvalidParamError: If any required parameter is missing.
        """        
        strides = next((attr.f for attr in node.attribute if attr.name == "strides"), "N/A")
        kernel_shape = next((attr.f for attr in node.attribute if attr.name == "kernel_shape"), "N/A")
        dilations = next((attr.f for attr in node.attribute if attr.name == "dilations"), "N/A")
        pads = next((attr.f for attr in node.attribute if attr.name == "pads"), "N/A")
        

        if strides == "N/A":
            raise InvalidParamError(node.name, node.op_type, f"Missing strides parameter", "strides")
        if kernel_shape == "N/A":
            raise InvalidParamError(node.name, node.op_type, f"Missing kernel_shape parameter", "kernel_shape")
        if dilations == "N/A":
            raise InvalidParamError(node.name, node.op_type, f"Missing dilations parameter", "dilations")
        if pads == "N/A":
            raise InvalidParamError(node.name, node.op_type, f"Missing pads parameter", "pads")

    def check_supported_shape(self, node: onnx.NodeProto, initializer_map: dict[str, onnx.TensorProto]) -> None:
        """ Ensure that Conv weights are available and have the correct dimensionality.

        Args:
            node (onnx.NodeProto): The node being validated.
            initializer_map (dict[str, onnx.TensorProto]): Mapping of initializer tensor names to TensorProtos.

        Raises:
            InvalidParamError: If weights are missing or have an unsupported shape.
        """        
        weight_name = node.input[1]
        initializer = initializer_map.get(weight_name)

        if initializer is None:
            raise InvalidParamError(node.name, node.op_type, f"Weight tensor '{weight_name}' for conv layer {node.name} not found in initializers")


        weight_dims = [dim for dim in initializer.dims]

        if len(weight_dims) != 4:
            msg = ""
            if len(weight_dims) == 3:
                msg += "1D Conv is not currently supported. "
            if len(weight_dims) == 5:
                msg += "3D Conv is not currently supported. "
            msg += f"Expected 4D weights for Conv2D, got shape {weight_dims}"
            raise InvalidParamError(node.name, node.op_type, msg)
