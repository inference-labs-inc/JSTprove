import numpy as np
import onnx
from onnx import helper, numpy_helper
from typing import List, Optional

from python.core.model_processing.onnx_custom_ops.onnx_helpers import create_quantized_initializer, replace_input_references


class BaseOpQuantizer:
    """
    Abstract base class for ONNX operator quantizers.

    Subclasses must implement:
        - `quantize`: Apply quantization logic to an ONNX node.
        - `check_supported`: Checks if the layer and param specs are supported.

    Attributes:
        new_initializers (list[onnx.TensorProto]):
            A list of initializers created during quantization.
            These should be added to the graph after processing.
    """
    def __init__(self):
        self.new_initializers: list[onnx.TensorProto] = []

    def quantize(self,
        node: onnx.NodeProto,
        rescale: bool,
        graph: onnx.GraphProto,
        scale_exponent: int,
        scale_base: int,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> List[onnx.NodeProto]:      
        """
        Quantize the given node.

        Must be implemented by subclasses.

        Raises:
            NotImplementedError: If called on BaseOpQuantizer directly.
        """
        # TODO should indicate this is a developer error, for not implementing the op
        raise NotImplementedError("Must implement quantize method for used layer")
    
    def check_supported(self, node: onnx.NodeProto, initializer_map: dict[str, onnx.TensorProto] = None) -> Optional[str]:
        """
        Check if the node is supported by the quantizer.

        Must be overridden by subclasses to validate parameters.

        Raises:
            NotImplementedError: If called on BaseOpQuantizer directly.
        """
        # TODO should indicate this is a developer error, for not implementing the op
        raise NotImplementedError("Must implement check_supported method for used layer")
    
    def rescale_layer(self, node: onnx.NodeProto, scale_base: int, scale_exponent: int, graph: onnx.GraphProto) -> List[onnx.NodeProto]:
        """
        Helper function for any quantizer. Used to add a rescaling step after the given node.

        This replaces the node's output with a scaled version using a Div op. This function incorporates the logic to insert and restructure the graph.

        Args:
            node (onnx.NodeProto): Node to rescale.
            scale_base (int): Base for the scaling exponent.
            scale_exponent (int): Scaling exponent.
            graph (onnx.GraphProto): The ONNX graph.

        Returns:
            List[onnx.NodeProto]: Original node and the inserted Div node.
        """
        original_output = node.output[0]
        quantized_output = original_output + "_raw"
        node.output[0] = quantized_output

        # Create scale constant initializer
        scale_const_name = node.name + "_scale"
        scale_value = scale_base ** scale_exponent
        scale_tensor = numpy_helper.from_array(np.array([scale_value], dtype=np.int64), name=scale_const_name)
        self.new_initializers.append(scale_tensor)

        # Create Div node for rescaling output
        div_node = helper.make_node(
            "Div",
            inputs=[quantized_output, scale_const_name],
            outputs=[original_output],  # restore original output name
            name=node.name + "_rescale"
        )

        # Rewire consumers to point to the new output
        replace_input_references(graph = graph, old_output = original_output, new_output = div_node.output[0])

        return [node, div_node]

    def add_nodes_w_and_b(self, node: onnx.NodeProto, scale_exponent: int, scale_base: int, initializer_map: dict[str, onnx.TensorProto], graph: onnx.GraphProto) -> tuple[list[onnx.NodeProto], list[str]]:
        """Insert scaling and casting nodes for weight and bias, to convert from float to scaled int64 values.

        Args:
            node (onnx.NodeProto): Node to find used weights and biases.
            scale_exponent (int): Scaling exponent.
            scale_base (int): Base for the scaling exponent.
            initializer_map (dict[str, onnx.TensorProto]): The initializer map.
            graph (onnx.GraphProto): ONNX Graph

        Returns:
            tuple[list[onnx.NodeProto], list[str]]: List of new nodes added, updated input names for nodes
        """        
        # Quantize weight
        weight_name = node.input[1]
        weight_tensor = initializer_map[weight_name]
        quant_weight_name, mul_node, cast_node = self.insert_scale_node(tensor = weight_tensor, scale_base = scale_base, scale_exponent = scale_exponent, graph = graph)

        # Quantize bias if present
        new_inputs = [node.input[0], quant_weight_name]
        nodes = [mul_node, cast_node]

        if len(node.input) > 2:
            bias_name = node.input[2]
            bias_tensor = initializer_map[bias_name]
            quant_bias_name, mul_node_2, cast_node_2 = self.insert_scale_node(tensor = bias_tensor, scale_base = scale_base, scale_exponent = (scale_exponent*2), graph = graph)
            new_inputs.append(quant_bias_name)
            nodes.append(mul_node_2)
            nodes.append(cast_node_2)


        # === Mutate the original node ===
        return  nodes, new_inputs
    
    def insert_scale_node(self, tensor: onnx.TensorProto, scale_base: int, scale_exponent: int, graph: onnx.GraphProto) -> tuple[str, onnx.NodeProto, onnx.NodeProto]: 
        """Insert Mul and Cast nodes to apply scaling to a tensor.

        Args:
            tensor (onnx.TensorProto): Tensor to scale.
            scale_base (int): Base for scaling exponent.
            scale_exponent (int): Scaling exponent.
            graph (onnx.GraphProto): ONNX graph.

        Returns:
            tuple[str, onnx.NodeProto, onnx.NodeProto]: New tensor name, Mul node, Cast node.
        """        
        scale_value = scale_base ** scale_exponent

        # Create scale constant
        scale_const_name = tensor.name + "_scale"
        scale_tensor = numpy_helper.from_array(
            np.array([scale_value], dtype=np.float64), name=scale_const_name
        )
        self.new_initializers.append(scale_tensor)

        # Add Mul node
        scaled_output_name = f"{tensor.name}_scaled"
        mul_node = helper.make_node(
            "Mul",
            inputs=[tensor.name, scale_const_name],
            outputs=[scaled_output_name],
            name=f"{tensor.name}_mul",
        )

        # Add cast node
        output_name = f"{scaled_output_name}_cast"
        rounded_output_name = scaled_output_name
        cast_to_int64 = helper.make_node(
            "Cast",
            inputs=[scaled_output_name],
            outputs=[output_name],
            to=onnx.TensorProto.INT64,
            name = rounded_output_name
        )
        return output_name, mul_node, cast_to_int64
    

class PassthroughQuantizer(BaseOpQuantizer):
    """
    Quantizer that leaves the node unchanged.
    Useful for operators that do not require quantization, such as shaping operations.
    """
    def __init__(self, new_initializer = None):
        super().__init__()
    def quantize(self, node, rescale, graph, scale_exponent, scale_base, initializer_map):
        return node
    def check_supported(self, node, initializer_map = None):
        return None
