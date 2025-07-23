import numpy as np
import onnx
from onnx import helper, numpy_helper
from typing import Callable, Dict, List, Optional, Union

from python.testing.core.utils.onnx_helpers import create_quantized_initializer, extract_attributes, replace_input_references
from onnx.numpy_helper import to_array, from_array


class BaseOpQuantizer:
    def quantize(self, node, rescale, graph, scale, scale_base, initializer_map):
        # TODO should indicate this is a developer error, for not implementing the op
        raise NotImplementedError("Must implement quantize method for used layer")
    
    def check_supported(self, node: onnx.NodeProto, initializer_map: dict[str, onnx.TensorProto] = None) -> Optional[str]:
        return None
    
    def rescale_layer(self, node: onnx.NodeProto, scale_base: int, scale: int, graph: onnx.GraphProto):
        original_output = node.output[0]
        quantized_output = original_output + "_raw"
        node.output[0] = quantized_output

        # Create scale constant initializer
        scale_const_name = node.name + "_scale"
        scale_value = scale_base ** scale
        scale_tensor = numpy_helper.from_array(np.array([scale_value], dtype=np.int64), name=scale_const_name)
        self.new_initializers.append(scale_tensor)

        # Create Div node for rescaling output
        div_node = helper.make_node(
            "Div",
            inputs=[quantized_output, scale_const_name],
            outputs=[original_output],  # restore output name
            name=node.name + "_rescale"
        )

        # Rewire consumers to point to the new output
        replace_input_references(graph, original_output, div_node.output[0])

        return [node, div_node]
    
    def quantize_w_and_b(self, node: onnx.NodeProto, scale: int, scale_base: int, initializer_map: dict[str, onnx.TensorProto]):
        # === Quantize weight ===
        weight_name = node.input[1]
        weight_tensor = initializer_map[weight_name]
        quant_weight_tensor, quant_weight_name = create_quantized_initializer(weight_tensor, scale_exponent=1, scale = scale, scale_base = scale_base)
        self.new_initializers.append(quant_weight_tensor)

        # === Quantize bias if present ===
        new_inputs = [node.input[0], quant_weight_name]
        if len(node.input) > 2:
            bias_name = node.input[2]
            bias_tensor = initializer_map[bias_name]
            quant_bias_tensor, quant_bias_name = create_quantized_initializer(bias_tensor, scale_exponent=2, scale = scale, scale_base = scale_base)
            self.new_initializers.append(quant_bias_tensor)
            new_inputs.append(quant_bias_name)

        # === Mutate the original node ===
        return new_inputs

    def add_nodes_w_and_b(self, node: onnx.NodeProto, scale: int, scale_base: int, initializer_map: dict[str, onnx.TensorProto], graph: onnx.GraphProto):
        # === Quantize weight ===
        weight_name = node.input[1]
        weight_tensor = initializer_map[weight_name]
        quant_weight_name, mul_node, cast_node = self.insert_scale_node(weight_tensor, scale_base, scale, graph)

        # === Quantize bias if present ===
        new_inputs = [node.input[0], quant_weight_name]
        nodes = [mul_node, cast_node]

        if len(node.input) > 2:
            bias_name = node.input[2]
            bias_tensor = initializer_map[bias_name]
            quant_bias_name, mul_node_2, cast_node_2 = self.insert_scale_node(bias_tensor, scale_base, (scale*2), graph)
            new_inputs.append(quant_bias_name)
            nodes.append(mul_node_2)
            # nodes.append(floor_node_2)
            nodes.append(cast_node_2)


        # === Mutate the original node ===
        return  nodes, new_inputs
    
    def insert_scale_node(self, tensor: onnx.TensorProto, scale_base: int, scale: int, graph: onnx.GraphProto):
        """
        Inserts Mul and Floor node before a node input to simulate quantization.

        Returns:
            New input name to be used by the consuming node.
        """
        scale_value = scale_base ** scale

        # === Create scale constant ===
        scale_const_name = tensor.name + "_scale"
        scale_tensor = numpy_helper.from_array(
            np.array([scale_value], dtype=np.float64), name=scale_const_name
        )
        self.new_initializers.append(scale_tensor)

        # === Add Mul node ===
        scaled_output_name = f"{tensor.name}_scaled"
        mul_node = helper.make_node(
            "Mul",
            inputs=[tensor.name, scale_const_name],
            outputs=[scaled_output_name],
            name=f"{tensor.name}_mul",
        )

        # === Add cast node ===
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
    def __init__(self, new_initializer = None):
        super().__init__()
    def quantize(self, node, rescale, graph, scale, scale_base, initializer_map):
        return node