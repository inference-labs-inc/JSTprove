import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper
from typing import Callable, Dict, List, Tuple, Union

from python_testing.utils.onnx_helpers import create_quantized_initializer, extract_attributes, replace_input_references
from onnx.numpy_helper import to_array, from_array

class ONNXOpQuantizer:
    def __init__(self):
        self.handlers: Dict[str, Callable[[onnx.NodeProto, bool], Union[onnx.NodeProto, List[onnx.NodeProto]]]] = {}
        self.new_initializers = [] 

        # Register handlers
        self.register("Conv", self._quantize_conv)
        self.register("MatMul", self._quantize_matmul)
        self.register("Relu", self._quantize_passthrough)
        self.register("Reshape", self._quantize_passthrough)
        self.register("Gemm", self._quantize_gemm)
        self.register("Constant", self._quantize_constant)

        



    def register(self, op_type: str, handler: Callable[[onnx.NodeProto, bool], Union[onnx.NodeProto, List[onnx.NodeProto]]]):
        self.handlers[op_type] = handler

    def quantize(self, node: onnx.NodeProto, rescale: bool, graph: onnx.GraphProto, scale: int, scale_base: int, initializer_map: dict[str, onnx.TensorProto]) -> Union[onnx.NodeProto, List[onnx.NodeProto]]:
        handler = self.handlers.get(node.op_type)
        if handler:
            return handler(node, rescale, graph, scale, scale_base, initializer_map)
        else:
            print(f"⚠️ No quantizer implemented for op_type: {node.op_type}")
            return node

    def _quantize_conv(
        self,
        node: onnx.NodeProto,
        rescale: bool,
        graph: onnx.GraphProto,
        scale: int,
        scale_base: int,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> List[onnx.NodeProto]:
        nodes = []
        # node.input[:] = self.quantize_w_and_b(node, scale, scale_base, initializer_map)
        nodes, node.input[:] = self.add_nodes_w_and_b(node, scale, scale_base, initializer_map, graph)
        attrs = extract_attributes(node)
        attrs.setdefault("group", 1)
        attrs.setdefault("auto_pad", "NOTSET")
        for attr in node.attribute:
            print(f"{attr.name}: type={attr.type} ({onnx.AttributeProto.AttributeType.Name(attr.type)})")

        
        if rescale:
            (node, div_node) = self.rescale_layer(node, scale_base, scale, graph)
            # attributes = list(node.attribute)
            
            output_name = f"{node.name}_int"
            
            int64_conv_node = onnx.helper.make_node(
                                            "Int64Conv",
                                            inputs=node.input,
                                            outputs=node.output,  # preserve original output name
                                            name=output_name,
                                            domain="ai.onnx.contrib",
                                            **attrs
                                        )
            
            # int64_conv_node.output[0] = "Y"
            nodes.append(int64_conv_node)
            new_div_node_output = f"{div_node.name}_out"
            cast_out_to_int64 = helper.make_node(
                "Cast",
                inputs=[new_div_node_output],
                outputs=div_node.output,
                to=onnx.TensorProto.INT64,
                name = f"{div_node.name}_cast"
            )
            div_node.output[0] = new_div_node_output
            nodes.append(div_node)
            nodes.append(cast_out_to_int64)
            return nodes

        else:
            output_name = f"{node.name}_int"
            int64_conv_node = onnx.helper.make_node(
                                            "Int64Conv",
                                            inputs=node.input,
                                            outputs=node.output,  # preserve original output name
                                            name=output_name,
                                            domain="ai.onnx.contrib",
                                            **attrs
                                        )
            nodes.append(int64_conv_node)
            return nodes

    def _quantize_matmul(self, node: onnx.NodeProto, rescale: bool, graph: onnx.GraphProto, scale: int, scale_base: int, initializer_map: dict[str, onnx.TensorProto]):
        # Stub for now
        raise NotImplementedError(f"Quantizing Matmul is not yet implemented {node.name}")
        print(f"Quantizing MatMul: {node.name}")
        return node
    
    def _quantize_gemm(self, node: onnx.NodeProto, rescale: bool, graph: onnx.GraphProto, scale: int, scale_base: int, initializer_map: dict[str, onnx.TensorProto]):
        nodes = []

        # node.input[:] = self.quantize_w_and_b(node, scale, scale_base, initializer_map)
        nodes, node.input[:] = self.add_nodes_w_and_b(node, scale, scale_base, initializer_map, graph)

        
        if rescale:
            (node, div_node) = self.rescale_layer(node, scale_base, scale, graph)
            
            prev_outputs = div_node.output[0]
            # node.input[0] = output_name
            output_name = f"{node.name}_int"
            attrs = extract_attributes(node)
            attrs.setdefault("transA", 0)
            attrs.setdefault("transB", 0)
            for attr in node.attribute:
                print(f"{attr.name}: type={attr.type} ({onnx.AttributeProto.AttributeType.Name(attr.type)})")

            int64_gemm = onnx.helper.make_node(
                                            "Int64Gemm",
                                            inputs=node.input,
                                            outputs=node.output,  # preserve original output name
                                            name=output_name,
                                            domain="ai.onnx.contrib",
                                            **attrs
                                        )
            nodes.append(int64_gemm)
            nodes.append(div_node)
            return nodes

        else:
            node.name = node.name + "_quant"
            nodes.append(node)
            return nodes

    def _quantize_passthrough(self, node: onnx.NodeProto, *args):
        return node  # e.g. ReLU: just pass it through
    

    def _quantize_constant(self, node: onnx.NodeProto, rescale: bool, graph: onnx.GraphProto, scale: int, scale_base: int, initializer_map: dict[str, onnx.TensorProto]) -> onnx.NodeProto:
        output_name = node.output[0]

        data_ops = {"Add", "Mul", "Conv", "MatMul", "Sub", "Div", "Gemm"}  # ops that consume numeric constants
        is_data_constant = any(
            output_name in n.input and n.op_type in data_ops
            for n in graph.node
        )

        if not is_data_constant:
            return node  # ✅ Used for shape or index — don't quantize

        # Safe to quantize: numeric constant used in computation
        for attr in node.attribute:
            if attr.name == "value" and attr.type == onnx.AttributeProto.TENSOR:
                arr = numpy_helper.to_array(attr.t).astype(np.float64)
                arr *= scale_base ** scale
                attr.t.CopyFrom(numpy_helper.from_array(arr, name=""))

        node.name = node.name + "_quant"
        return node
    
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
        original_output = tensor.name

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
        # graph.node.append(mul_node)
        # replace_input_references(graph, original_output, mul_node.output[0])

        # # === Floor node (simulate rounding) ===
        # rounded_output_name = f"{tensor.name}_scaled_floor"
        # floor_node = helper.make_node(
        #     "Floor",
        #     inputs=[scaled_output_name],
        #     outputs=[rounded_output_name],
        #     name=f"{scaled_output_name}",
        # )
        # output_name = f"{rounded_output_name}_int"
        # cast_to_int64 = helper.make_node(
        #     "Cast",
        #     inputs=[rounded_output_name],
        #     outputs=[output_name],
        #     to=onnx.TensorProto.INT64,
        #     name = rounded_output_name
        # )
        # The following is with removed floor node
        output_name = f"{scaled_output_name}_cast"
        rounded_output_name = scaled_output_name
        cast_to_int64 = helper.make_node(
            "Cast",
            inputs=[scaled_output_name],
            outputs=[output_name],
            to=onnx.TensorProto.INT64,
            name = rounded_output_name
        )
        # graph.node.append(floor_node)
        # replace_input_references(graph, original_output, floor_node.output[0])
        return output_name, mul_node, cast_to_int64
        # replace_input_references(graph, original_output, mul_node.output[0])
        # return scaled_output_name
    
    