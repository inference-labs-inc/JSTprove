import copy
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import onnx
from onnx import NodeProto, TensorProto, shape_inference, helper, numpy_helper

from python.core.model_processing.onnx_custom_ops.onnx_helpers import extract_shape_dict, get_input_shapes, parse_attributes
from python.core.model_processing.onnx_quantizer.onnx_op_quantizer import ONNXOpQuantizer
from python.core.model_processing.converters.base import ModelConverter

"""
Keep the ununused import below as it
must remain due to 'SessionOptions' dependency.
"""
import python.core.model_processing.onnx_custom_ops
from onnxruntime import InferenceSession, SessionOptions
from onnxruntime_extensions import get_library_path

@dataclass
class ONNXLayer:
    """
    A dataclass representing an ONNX layer, in the form to be sent to the circuit building process
    """    
    id: int
    name: str
    op_type: str #This is the operation type. eg. "Conv" for convolution layers
    inputs: List[str] #This will be a list of other input layers. the str inside the list, can either be the name or id. Unsure of the best way to tackle this
    outputs: List[str]
    shape: Dict[str, List[int]] # This will be a hashmap where the key is the output layer and the List[int] is the shape of that output layer
    tensor: Optional[List] # This will be empty for layers, and will contain the weights or biases, for Const nodes
    params: Optional[Dict] # Most layers will have params attached to them. For eg, params for conv would be - dilation, kernel_shape, pad, strides, group,...
    opset_version_number: int # This is the version number of the operation used. So far this is not used in rust, but I think for infrastructure purposes we can include

@dataclass
class ONNXIO:
    """
    A dataclass representing an ONNX input or output, in the form to be sent to the circuit building process
    """    
    name: str
    elem_type: int
    shape: List[int]


class ONNXConverter(ModelConverter):
    """Concrete implementation of `ModelConverter` for ONNX models.
    """
    def __init__(self):
        """Initialize the converter and its operator quantizer.

        Initializes:
            self.op_quantizer (ONNXOpQuantizer): Dispatcher that quantizes
                individual ONNX ops and accumulates newly created initializers.
        """
        super().__init__()
        self.op_quantizer = ONNXOpQuantizer()

    # For saving and loading: https://onnx.ai/onnx/intro/python.html, larger models may require a different structure
    def save_model(self, file_path: str) -> None:
        """Serialize the ONNX model to file.

        Args:
            file_path (str): Destination path (e.g., ``"models/my_model.onnx"``).
        
        Note
        ----
        - For saving and loading: https://onnx.ai/onnx/intro/python.html, larger models may require a different structure
        """        
        onnx.save(self.model, file_path)
    
    def load_model(self, file_path: str, model_type = None) -> onnx.ModelProto:
        """Load an ONNX model from file and extract basic I/O metadata.

        Args:
            file_path (str): Path to the `.onnx` file.
            model_type (optional): Placeholder for future multi-format loaders. Defaults to None.

        Returns:
            onnx.ModelProto: The loaded onnx model.
        """        
        onnx_model = onnx.load(file_path)
        self.model = onnx_model

        self._extract_model_io_info(onnx_model)
        return self.model
    

    def save_quantized_model(self, file_path: str) -> None:
        """Serialize the quantized ONNX model to file.

        Args:
            file_path (str): Destination path for the quantized model.
        """
        onnx.save(self.quantized_model, file_path)

    # Not sure this is ideal
    def load_quantized_model(self, file_path: str) -> None:
        """Load a quantized ONNX model and create an inference session.

        Note
        ----
         - Uses the custom opset for the quantized layers

        Args:
            file_path (str): Path to the quantized ``.onnx`` file.
        """
        # May be able to remove next few lines...
        print("Loading quantized model from: ", file_path) #TODO: Change to logging
        onnx_model = onnx.load(file_path)
        custom_domain = onnx.helper.make_operatorsetid(domain="ai.onnx.contrib", version=1)
        onnx_model.opset_import.append(custom_domain)
        # Fix, can remove this next line 
        self.quantized_model = onnx_model
        self.ort_sess =  self._create_inference_session(file_path)
        self._extract_model_io_info(onnx_model)

        self.quantized_model_path = file_path

    def analyze_layers(self, output_name_to_shape: Dict[str, List[int]] = None) -> Tuple[List[ONNXLayer], List[ONNXLayer]]:
        """Analyze the onnx model graph into logical layers and parameter tensors.

        Args:
            output_name_to_shape (Dict[str, List[int]], optional): mapping of value name -> shape. If
                omitted, shapes are inferred via `onnx.shape_inference`. Defaults to None.

        Returns:
            Tuple[List[ONNXLayer], List[ONNXLayer]]: ``(architecture, w_and_b)`` where:
                - ``architecture`` is a list of `ONNXLayer` describing
                  the computational graph.
                - ``w_and_b`` is a list of `ONNXLayer` representing
                  constant tensors (initializers).
        """
        id_count = 0
        # Apply shape inference on the model
        if not output_name_to_shape:
            inferred_model = shape_inference.infer_shapes(self.model) 

            # Check the model and print Y"s shape information
            onnx.checker.check_model(inferred_model)
            output_name_to_shape = extract_shape_dict(inferred_model)
        domain_to_version = {opset.domain: opset.version for opset in self.model.opset_import}
        
        id_count = 0
        architecture = self.get_model_architecture(self.model, output_name_to_shape, id_count, domain_to_version)
        w_and_b = self.get_model_w_and_b(self.model, output_name_to_shape, id_count, domain_to_version)
        return (architecture, w_and_b)
    
    
    
    def run_model_onnx_runtime(self, path: str, input: torch.Tensor) -> Any:
        """Execute a model on CPU via ONNX Runtime and return its outputs.

        Creates a fresh inference session for the model at ``path``, feeds
        the provided tensor under the first input name, and returns the
        first output.

        Args:
            path (str): Path to the ONNX model to execute.
            input (torch.Tensor): Input tensor to feed into the model's first input.

        Returns:
            Any: The output(s) as returned by `InferenceSession.run`.
        """
        # Fix, can remove this next line        

        ort_sess =  self._create_inference_session(path)
        input_name = ort_sess.get_inputs()[0].name
        output_name = ort_sess.get_outputs()[0].name
        if ort_sess.get_inputs()[0].type == "tensor(double)":
            outputs = ort_sess.run([output_name], {input_name: np.asarray(input).astype(np.float64)})
        else:
            outputs = ort_sess.run([output_name], {input_name: np.asarray(input)})
        
        return outputs
    
    def get_model_architecture(self, model: onnx.ModelProto, output_name_to_shape: Dict[str, List[int]], id_count: int = 0, domain_to_version: dict[str, int] = None) -> List[ONNXLayer]:
        """Construct ONNXLayer objects for architecture graph nodes (not weights or biases).

        Args:
            model (onnx.ModelProto): The ONNX model to analyze.
            output_name_to_shape (Dict[str, List[int]]): Map of value name -> inferred shape.
            id_count (int, optional): Starting numeric ID for layers (incremented per node). Defaults to 0.
            domain_to_version (dict[str, int], optional): Map of opset domain -> version used. Defaults to None.

        Returns:
            List[ONNXLayer]: Model’s computational layers (excluding initializers) in the form of ONNXLayers.
        """
        layers = []
        constant_values = {}
        # First pass: collect constant nodes
        for node in model.graph.node:
            if node.op_type == "Constant":
                print(node) #TODO: Change to logging
                for attr in node.attribute:
                    if attr.name == "value":
                        tensor = attr.t
                        const_value = numpy_helper.to_array(tensor)
                        constant_values[node.output[0]] = const_value

        # Map output name to shape (assumed provided from previous analysis)
        layers = []
        id_count = 0

        # Second pass: analyze layers
        for (idx, node) in enumerate(model.graph.node):
            if node.op_type == "Constant":
                continue  # Already processed

            layer = self.analyze_layer(node, output_name_to_shape, id_count, domain_to_version)
            print(layer.shape) #TODO: Change to logging

            # Attach constant inputs as parameters
            for input_name in node.input:
                if input_name in constant_values:
                    print(layer.params) #TODO: Change to logging
                    if not hasattr(layer, 'params'):
                        layer.params = {}
                    result = constant_values[input_name]
                    if isinstance(result, np.ndarray) or isinstance(result, torch.Tensor):
                        layer.params[input_name] = result.tolist()
                    else:
                        layer.params[input_name] = constant_values[input_name]
                    print(layer.params) #TODO: Change to logging

            layers.append(layer)
            id_count += 1
        return layers
    
    def get_model_w_and_b(self, model: onnx.ModelProto, output_name_to_shape: Dict[str, List[int]], id_count: int = 0, domain_to_version: dict[str, int] = None) -> List[ONNXLayer]:
        """Extract constant initializers (weights/biases) as layers.

        Iterates through graph initializers and wraps each tensor
        into an ONNXLayers.

        Args:
            model (onnx.ModelProto): The ONNX model to analyze.
            output_name_to_shape (Dict[str, List[int]]): Map of value name -> inferred shape
            id_count (int, optional): Starting numeric ID for layers (incremented per tensor). Defaults to 0.
            domain_to_version (dict[str, int], optional): Map of opset domain -> version used (unused). Defaults to None.

        Returns:
            List[ONNXLayer]: ONNXLayers representing weights/biases found in the graph
        """
        layers = []
        # Check the model and print Y"s shape information
        for (idx, node) in enumerate(model.graph.initializer):
            layer = self.analyze_constant(node, output_name_to_shape, id_count, domain_to_version )
            layers.append(layer)
            id_count += 1

        return layers
    
    def _create_inference_session(self, model_path: str) -> InferenceSession:
        """Internal helper to create and configure an ONNX Runtime InferenceSession.
        Registers a custom ops shared library for use with the custom quantized operations.

        Args:
            model_path (str): Path to the ONNX model to load.

        Returns:
            InferenceSession: A configured InferenceSession.
        """
        opts = SessionOptions()
        opts.register_custom_ops_library(get_library_path())
        return InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        

    def analyze_layer(self, node: NodeProto, output_name_to_shape: Dict[str, List[int]], id_count: int = -1, domain_to_version: dict[str, int] = None) -> ONNXLayer:
        """Convert a non-constant ONNX node into a structured ONNXLayer.

        Args:
            node (NodeProto): The ONNX node to analyze.
            output_name_to_shape (Dict[str, List[int]]): Map of value name -> inferred shape.
            id_count (int, optional): Numeric ID to assign to this layer (increment handled by caller). Defaults to -1.
            domain_to_version (dict[str, int], optional): Map of opset domain -> version number. Defaults to None.

        Returns:
            ONNXLayer: ONNXLayer describing the node
        """
        name = node.name
        id = id_count
        id_count += 1
        op_type = node.op_type
        inputs = node.input
        outputs = node.output
        domain = node.domain if node.domain else "ai.onnx"
        opset_version = domain_to_version.get(node.domain, "unknown") if domain_to_version else -1
        params = parse_attributes(node.attribute)

        # 💡 Extract output shapes
        output_shapes = {
                out_name: output_name_to_shape.get(out_name, []) for out_name in outputs
            }
        layer = ONNXLayer(
                id = id, 
                name = name,
                op_type = op_type,
                inputs = list(inputs),
                outputs = list(outputs),
                shape = output_shapes,
                params = params,
                opset_version_number = opset_version,
                tensor = None,
            )
        return layer
    
    def analyze_constant(self, node: TensorProto, output_name_to_shape: Dict[str, List[int]], id_count: int = -1, domain_to_version: dict[str, int] = None) -> List[ONNXLayer]:
        """Convert a constant ONNX node (weights or bias) into a structured ONNXLayer.

        Args:
            node (NodeProto): The ONNX node to analyze.
            output_name_to_shape (Dict[str, List[int]]): Map of value name -> inferred shape.
            id_count (int, optional): Numeric ID to assign to this layer (increment handled by caller). Defaults to -1.
            domain_to_version (dict[str, int], optional): Map of opset domain -> version number. Defaults to None.

        Returns:
            ONNXLayer: ONNXLayer describing the node
        """        
        name = node.name
        id = id_count
        id_count += 1
        op_type = "Const"
        inputs = []
        outputs = []
        domain = "ai.onnx"
        opset_version = -1
        params = {}
        constant_dtype = node.data_type
        # Can do this step in rust potentially to keep file sizes low if needed
        np_data = onnx.numpy_helper.to_array(node, constant_dtype)
            # 💡 Extract output shapes
        output_shapes = {
                out_name: output_name_to_shape.get(out_name, []) for out_name in outputs
            }
        layer = ONNXLayer(
                id = id, 
                name = name,
                op_type = op_type,
                inputs = list(inputs),
                outputs = list(outputs),
                shape = output_shapes,
                params = params,
                opset_version_number = opset_version,
                tensor = np_data.tolist(),
            )
        return layer
    
    def quantize_model(self, unscaled_model: onnx.ModelProto, scale_base: int,  scale_exponent: int, rescale_config: dict = None) -> onnx.ModelProto:
        """Produce a quantized ONNX graph from a floating-point model.

        1. Read in the model and layers + analyze
        2. Look for layers that need quantizing 
        3. Convert layer to quantized version
        4. insert quantized version back into the model

        Args:
            unscaled_model (onnx.ModelProto): The original unscaled model.
            scale_base (int): Base for fixed-point scaling (e.g., 2).
            scale_exponent (int): Exponent for scaling (e.g., 18 would lead to a scale factor 2**18).
            rescale_config (dict, optional): mapping of node name -> bool to control
                whether a given node should apply a final rescale. Defaults to None.

        Returns:
            onnx.ModelProto: A new onnx model representation of the quantized model.
        """
        
        model = copy.deepcopy(unscaled_model)

        # Check that all of the layers in model are supported by our system
        self.op_quantizer.check_model(model)

        initializer_map = {init.name: init for init in model.graph.initializer}
        input_names = [inp.name for inp in unscaled_model.graph.input]



        new_nodes = []
        for i, name in enumerate(input_names):
            output_name, mul_node, floor_node, cast_to_int64 = self.quantize_input(input_name = name, op_quantizer = self.op_quantizer, scale_base = scale_base, scale_exponent = scale_exponent)
            new_nodes.append(mul_node)
            # new_nodes.append(floor_node)
            new_nodes.append(cast_to_int64)
            for node in model.graph.node:
                for idx, inp in enumerate(node.input):
                    if inp == name:
                        node.input[idx] = output_name
        for input_tensor in model.graph.input:
            tensor_type = input_tensor.type.tensor_type
            # Only change float32 (type = 1)
            if tensor_type.elem_type == TensorProto.FLOAT:
                tensor_type.elem_type = TensorProto.DOUBLE  # float64 is enum 11
        



        for node in model.graph.node:
            rescale = rescale_config.get(node.name, False) if rescale_config else True
            quant_nodes = self.quantize_layer(node = node, rescale = rescale, model = model, scale_exponent = scale_exponent, scale_base = scale_base, initializer_map = initializer_map)
            if isinstance(quant_nodes, list):
                new_nodes.extend(quant_nodes)
            else:
                new_nodes.append(quant_nodes)

        model.graph.ClearField("node")
        model.graph.node.extend(new_nodes)

        used_initializer_names = set()
        for node in model.graph.node:
            used_initializer_names.update(node.input)

        # Keep only initializers actually used
        kept_initializers = []
        for name in used_initializer_names:
            if name in initializer_map:
                orig_init = initializer_map[name]
                np_array = numpy_helper.to_array(orig_init)

                if np_array.dtype == np.float32:
                    # Convert to float64
                    np_array = np_array.astype(np.float64)
                    new_init = numpy_helper.from_array(np_array, name=name)
                    kept_initializers.append(new_init)
                else:
                    # Keep as-is
                    kept_initializers.append(orig_init)

        model.graph.ClearField("initializer")
        model.graph.initializer.extend(kept_initializers)
        model.graph.initializer.extend(self.op_quantizer.new_initializers)

        self.op_quantizer.new_initializers = []
        
        for layer in model.graph.node:
            #TODO: Change to logging
            print(layer.name, layer.op_type, layer.input, layer.output)
            

        for layer in model.graph.initializer:
            #TODO: Change to logging
            print(layer.name)

        for out in model.graph.output:
                out.type.tensor_type.elem_type = onnx.TensorProto.INT64
        # TODO This has not been extensively tested. May need to somehow include this when quantizing layers individually (Concern is that some layers shouldnt be converted into this type...)
        # Such as multiplying up scalers etc.
        for vi in model.graph.value_info:
            vi.type.tensor_type.elem_type = TensorProto.INT64
            
        custom_domain = helper.make_operatorsetid(domain="ai.onnx.contrib",version=1)
        domains = [op.domain for op in model.opset_import]
        if "ai.onnx.contrib" not in domains:
            model.opset_import.append(custom_domain)
        # onnx.checker.check_model(model)
        # onnx.save(model, "debug_test.onnx")
        return model
        

    def quantize_layer(self, node: onnx.NodeProto, rescale: bool, model: onnx.ModelProto, scale_exponent: int, scale_base: int, initializer_map: dict[str, onnx.TensorProto]) -> onnx.NodeProto:
        """Quantize a single ONNX node using the configured op quantizer.

        Args:
            node (onnx.NodeProto): The original onnx node to quantize.
            rescale (bool): Whether to apply output rescaling for this node.
            model (onnx.ModelProto): The original model used for context
            scale_exponent (int): Exponent for scaling (e.g., 18 would lead to a scale factor 2**18).
            scale_base (int): Base for fixed-point scaling (e.g., 2).
            initializer_map (dict[str, onnx.TensorProto]): Mapping from initializer name to tensor.

        Returns:
            onnx.NodeProto: A quantized node or list of nodes replacing the initial node.
        """
        quant_nodes = self.op_quantizer.quantize(node = node, rescale = rescale, graph = model.graph, scale_exponent = scale_exponent, scale_base = scale_base, initializer_map = initializer_map)
        return quant_nodes
    
    def quantize_input(self, input_name: str, op_quantizer: ONNXOpQuantizer, scale_base: int, scale_exponent: int) -> Tuple[str, onnx.NodeProto, onnx.NodeProto, onnx.NodeProto]:
        """Insert scaling and casting nodes to quantize a model input.

        Creates:
            - Mul: scales the input by scale_base ** scale.
            - Cast (to INT64): produces the final integer input tensor.

        Args:
            input_name (str): Name of the graph input to quantize.
            op_quantizer (ONNXOpQuantizer): The op quantizer whose ``new_initializers`` list is
                used to store the created scale constant.
            scale_base (int): Base for fixed-point scaling (e.g., 2).
            scale_exponent (int): Exponent for scaling (e.g., 18 would lead to a scale factor 2**18).

        Returns:
            Tuple[str, onnx.NodeProto, onnx.NodeProto, onnx.NodeProto]: A tuple ``(output_name, mul_node, floor_node, cast_node)`` where
            ``output_name`` is the name of the quantized input tensor and the nodes are nodes to add to the graph.
        """
        scale_value = scale_base ** scale_exponent

        # === Create scale constant ===
        scale_const_name = input_name + "_scale"
        scale_tensor = numpy_helper.from_array(
            np.array([scale_value], dtype=np.float64), name=scale_const_name
        )
        op_quantizer.new_initializers.append(scale_tensor)

        # === Add Mul node ===
        scaled_output_name = f"{input_name}_scaled"
        mul_node = helper.make_node(
            "Mul",
            inputs=[input_name, scale_const_name],
            outputs=[scaled_output_name],
            name=f"{input_name}_mul",
        )
        # === Floor node (simulate rounding) ===
        rounded_output_name = f"{input_name}_scaled_floor"
        floor_node = helper.make_node(
            "Floor",
            inputs=[scaled_output_name],
            outputs=[rounded_output_name],
            name=f"{scaled_output_name}",
        )
        output_name = f"{rounded_output_name}_int"
        cast_to_int64 = helper.make_node(
            "Cast",
            inputs=[scaled_output_name],
            outputs=[output_name],
            to=onnx.TensorProto.INT64,
            name = rounded_output_name
        )
        return output_name, mul_node, floor_node, cast_to_int64
    
    def _extract_model_io_info(self, onnx_model: onnx.ModelProto) -> None:
        """Populate input metadata from a loaded ONNX model.

        Args:
            onnx_model (onnx.ModelProto): Onnx model
        """
        self.required_keys = [input.name for input in onnx_model.graph.input]
        self.input_shape = get_input_shapes(onnx_model)
    

    # TODO JG suggestion - can maybe make the layers into a factory here, similar to how its done in Rust? Can refactor to this later imo.
    def get_weights(self, flatten: bool = False) -> Tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any]]:
        """Export architecture, weights, and circuit parameters for ECC.

        1. Analyze the model for architecture + w & b
        2. Put arch into format to be read by ECC circuit builder
        3. Put w + b into format to be read by ECC circuit builder

        Args:
            flatten (bool, optional): Currently unused; reserved for alternative layouts. Defaults to False.

        Returns:
            Tuple[dict[str, list[dict[str, Any]]], dict[str, list[dict[str, Any]]], dict[str, Any]]: A tuple ``(architecture, weights, circuit_params)``:
                - ``architecture``: Dict with serialized ``architecture`` layers.
                - ``weights``: Dict containing ``w_and_b`` (serialized tensors).
                - ``circuit_params``: Dict containing scaling parameters and
                  ``rescale_config``.
        """
        print(self.model.graph.node) #TODO: Change to logging
        inferred_model = shape_inference.infer_shapes(self.model) 

        # Check the model and print Y"s shape information
        # TODO ERRORS handle gently
        onnx.checker.check_model(inferred_model)
        output_name_to_shape = extract_shape_dict(inferred_model)
        (architecture, w_and_b) = self.analyze_layers(output_name_to_shape)
        for w in w_and_b:
            w_and_b_array = np.asarray(w.tensor)
            # VERY VERY TEMPORARY FIX
            if "bias" in w.name:
                w_and_b_scaled = w_and_b_array * (getattr(self, "scale_base", 2)**(getattr(self,"scale_exponent", 18)*2))
            else:
                w_and_b_scaled = w_and_b_array * (getattr(self, "scale_base", 2)**getattr(self,"scale_exponent", 18))
            w_and_b_out = w_and_b_scaled.astype(np.int64).tolist()
            w.tensor = w_and_b_out
        
        inputs = []
        outputs = []
        for input in self.model.graph.input:
            shape =  output_name_to_shape.get(input.name, [])
            elem_type = getattr(input, "elem_type", -1)
            inputs.append(ONNXIO(input.name, elem_type, shape))

        for output in self.model.graph.output:
            shape =  output_name_to_shape.get(output.name, [])
            elem_type = getattr(output, "elem_type", -1)
            outputs.append(ONNXIO(output.name, elem_type, shape))
        
        architecture = {
            "inputs": [asdict(i) for i in inputs],
            "outputs": [asdict(o) for o in outputs],
            "architecture": [asdict(a) for a in architecture],
        }
        weights = {
            "w_and_b": [asdict(w_b) for w_b in w_and_b]
        }
        circuit_params = {
            "scale_base": getattr(self, "scale_base", 2),
            "scale_exponent": getattr(self,"scale_exponent", 18),
            "rescale_config": getattr(self, "rescale_config", {})
        }
        self.save_quantized_model("test.onnx")
        return architecture, weights, circuit_params

    def get_model_and_quantize(self) -> None:
        """Load the configured model (by path) and build its quantized form.

        Expects the instance to define ``self.model_file_name`` beforehand.

        Raises:
            FileNotFoundError: If ``self.model_file_name`` is unset or invalid.
        """
        if hasattr(self, 'model_file_name'):
            self.load_model(self.model_file_name)
        else:
            raise FileNotFoundError("An ONNX model is required at the specified path")
        self.quantized_model = self.quantize_model(self.model, getattr(self,"scale_base", 2), getattr(self,"scale_exponent", 18), rescale_config=getattr(self,"rescale_config", {}))

    def get_outputs(self, inputs: Any) -> Any:
        """Run the currently loaded (quantized) model via ONNX Runtime.

        Args:
            inputs (Any): Input array/tensor matching the model’s first input.

        Returns:
            Any: The output of the onnxruntime inference.
        """
        input_name = self.ort_sess.get_inputs()[0].name
        output_name = self.ort_sess.get_outputs()[0].name

        # TODO This may cause some rounding errors at some point but works for now. Should be checked at some point
        inputs = torch.as_tensor(inputs)
        if inputs.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            inputs = inputs.double()
            inputs = inputs / (self.scale_base**self.scale_exponent)

        # TODO add for all inputs (we should be able to account for multiple inputs...)
        # TODO this may not be optimal or robust
        if self.ort_sess.get_inputs()[0].type == "tensor(double)":
            outputs = self.ort_sess.run([output_name], {input_name: np.asarray(inputs).astype(np.float64)})
        else:
            outputs = self.ort_sess.run([output_name], {input_name: np.asarray(inputs)})
        return outputs


if __name__ == "__main__":
    path  ="./models_onnx/doom.onnx"


    converter = ONNXConverter()
    converter.model_file_name, converter.quantized_model_file_name = path, "quantized_doom.onnx"
    converter.scale_base, converter.scale_exponent = 2,18

    converter.load_model(path)
    converter.get_model_and_quantize()

    converter.test_accuracy()
