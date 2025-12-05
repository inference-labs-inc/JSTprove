from __future__ import annotations

import copy
import logging
from dataclasses import asdict, dataclass
from importlib.metadata import version as get_version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from onnxruntime import NodeArg

import numpy as np
import onnx
import torch
from onnx import NodeProto, TensorProto, helper, numpy_helper, shape_inference

# Keep the ununused import below as it
# must remain due to 'SessionOptions' dependency.
from onnxruntime import InferenceSession, SessionOptions
from onnxruntime_extensions import get_library_path

import python.core.model_processing.onnx_custom_ops  # noqa: F401
from python.core import PACKAGE_NAME
from python.core.circuits.errors import CircuitConfigurationError
from python.core.model_processing.converters.base import ModelConverter, ModelType
from python.core.model_processing.errors import (
    InferenceError,
    InvalidModelError,
    IOInfoExtractionError,
    LayerAnalysisError,
    ModelConversionError,
    ModelLoadError,
    ModelSaveError,
    SerializationError,
)
from python.core.model_processing.onnx_custom_ops.onnx_helpers import (
    extract_shape_dict,
    get_input_shapes,
    parse_attributes,
)
from python.core.model_processing.onnx_quantizer.exceptions import QuantizationError
from python.core.model_processing.onnx_quantizer.layers.base import (
    BaseOpQuantizer,
    ScaleConfig,
)
from python.core.model_processing.onnx_quantizer.onnx_op_quantizer import (
    ONNXOpQuantizer,
)

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # noqa: F401

ONNXLayerDict = dict[
    str,
    int | str | list[str] | dict[str, list[int]] | list | None | dict,
]

ONNXIODict = dict[str, str | int | list[int]]

CircuitParamsDict = dict[str, int | dict[str, bool]]


@dataclass
class ONNXLayer:
    """
    A dataclass representing an ONNX layer in the form
    to be sent to the circuit building process.

    This class encapsulates the essential information
    about a layer in an ONNX model. It is designed to facilitate the
    conversion and processing of ONNX models for circuit building purposes.

    Attributes:
        id (int): A unique identifier for the layer.
        name (str): The name of the layer.
        op_type (str): The operation type of the layer,
            such as "Conv" for convolution layers.
        inputs (list[str]): A list of input names that this layer depends on.
        outputs (list[str]): A list of output names produced by this layer.
        shape (dict[str, list[int]]): A dictionary mapping output names
            to their corresponding shapes.
        tensor (Optional[list]): For constant nodes, this contains the
            tensor data (weights or biases) as a list. For other layers, empty.
        params (Optional[dict]): A dictionary of parameters specific to the
            layer's operation. For example, convolution layers may include parameters
            like dilation, kernel_shape, pad, strides, and group.
        opset_version_number (int): The version number of the ONNX opset
            used for this operation. This is included for infrastructure
            purposes and may not be actively used in all processing steps.
    """

    id: int
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    shape: dict[
        str,
        list[int],
    ]
    tensor: list | None
    params: dict | None
    opset_version_number: int


@dataclass
class ONNXIO:
    """
    A dataclass representing an ONNX input or output,
    in the form to be sent to the circuit building process
    """

    name: str
    elem_type: int
    shape: list[int]


class ONNXConverter(ModelConverter):
    """Concrete implementation of `ModelConverter` for ONNX models."""

    def __init__(self: ONNXConverter) -> None:
        """Initialize the converter and its operator quantizer.

        Initializes:
            self.op_quantizer (ONNXOpQuantizer): Dispatcher that quantizes
                individual ONNX ops and accumulates newly created initializers.
        """
        self.op_quantizer = ONNXOpQuantizer()
        self.model_type = ModelType.ONNX
        self.logger = logging.getLogger(__name__)

    def save_model(self: ONNXConverter, file_path: str) -> None:
        """Serialize the ONNX model to file.

        Args:
            file_path (str):
                Destination path (e.g., ``"models/my_model.onnx"``).

        Note
        ----
        - For saving and loading:
            https://onnx.ai/onnx/intro/python.html,
            larger models may require a different structure
        """
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            onnx.save(self.model, file_path)
        except Exception as e:
            raise ModelSaveError(
                file_path,
                model_type=self.model_type,
                reason=str(e),
            ) from e

    def load_model(self: ONNXConverter, file_path: str) -> onnx.ModelProto:
        """Load an ONNX model from file and extract basic I/O metadata.

        Args:
            file_path (str): Path to the `.onnx` file.

        Returns:
            onnx.ModelProto: The loaded onnx model.

        Raises:
            ModelLoadError: If the model cannot be loaded or validated.
        """
        try:
            onnx_model = onnx.load(file_path)
        except Exception as e:
            raise ModelLoadError(
                file_path,
                model_type=self.model_type,
                reason=str(e),
            ) from e

        self.model = onnx_model

        try:
            self._extract_model_io_info(onnx_model)
        except Exception as e:
            raise IOInfoExtractionError(
                model_path=file_path,
                model_type=self.model_type,
                reason=str(e),
            ) from e
        return self.model

    def save_quantized_model(self: ONNXConverter, file_path: str) -> None:
        """Serialize the quantized ONNX model to file.

        Args:
            file_path (str): Destination path for the quantized model.
        """
        try:
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            onnx.save(self.quantized_model, file_path)
        except Exception as e:
            raise ModelSaveError(
                file_path,
                model_type=self.model_type,
                reason=str(e),
            ) from e

    # Not sure this is ideal
    def load_quantized_model(self: ONNXConverter, file_path: str) -> None:
        """Load a quantized ONNX model and create an inference session.

        Note
        ----
          - Uses the custom opset for the quantized layers

        Args:
            file_path (str): Path to the quantized ``.onnx`` file.

        Raises:
            FileNotFoundError: If the file does not exist.
            ModelLoadError: If loading or validation fails.
        """
        if not Path(file_path).exists():
            msg = f"Quantized model file not found: {file_path}"
            raise FileNotFoundError(msg)
        self.logger.info("Loading quantized model from", extra={"file_path": file_path})
        onnx_model = onnx.load(file_path)
        custom_domain = onnx.helper.make_operatorsetid(
            domain="ai.onnx.contrib",
            version=1,
        )
        onnx_model.opset_import.append(custom_domain)
        # Fix, can remove this next line
        self.quantized_model = onnx_model
        self.ort_sess = self._create_inference_session(file_path)
        self._extract_model_io_info(onnx_model)

        self.quantized_model_path = file_path

    def _onnx_check_model_safely(self: ONNXConverter, model: onnx.ModelProto) -> None:
        try:
            onnx.checker.check_model(model)
        except Exception as e:
            raise InvalidModelError(
                model_path=getattr(self, "model_file_name", None),
                reason=f"Model validation failed: {e!s}",
            ) from e

    def analyze_layers(
        self: ONNXConverter,
        model: onnx.ModelProto,
        output_name_to_shape: dict[str, list[int]] | None = None,
    ) -> tuple[list[ONNXLayer], list[ONNXLayer]]:
        """Analyze the onnx model graph into
        logical layers and parameter tensors.

        Args:
            output_name_to_shape (dict[str, list[int]], optional):
                mapping of value name -> shape. If omitted,
                shapes are inferred via `onnx.shape_inference`. Defaults to None.

        Returns:
            Tuple[list[ONNXLayer], list[ONNXLayer]]: ``(architecture, w_and_b)`` where:
                - ``architecture`` is a list of `ONNXLayer` describing
                  the computational graph.
                - ``w_and_b`` is a list of `ONNXLayer` representing
                  constant tensors (initializers).
        """
        try:
            id_count = 0
            # Apply shape inference on the model
            if not output_name_to_shape:
                inferred_model = shape_inference.infer_shapes(model)
                self._onnx_check_model_safely(inferred_model)

                output_name_to_shape = extract_shape_dict(inferred_model)
            domain_to_version = {
                opset.domain: opset.version for opset in model.opset_import
            }

            id_count = 0
            architecture = self.get_model_architecture(
                model,
                output_name_to_shape,
                domain_to_version,
            )
            w_and_b = self.get_model_w_and_b(
                model,
                output_name_to_shape,
                id_count,
                domain_to_version,
            )
        except InvalidModelError:
            raise
        except (ValueError, TypeError, RuntimeError, OSError) as e:
            raise LayerAnalysisError(model_type=self.model_type, reason=str(e)) from e
        except Exception as e:
            raise LayerAnalysisError(model_type=self.model_type, reason=str(e)) from e
        else:
            return (architecture, w_and_b)

    def run_model_onnx_runtime(
        self: ONNXConverter,
        path: str,
        inputs: torch.Tensor,
    ) -> list[np.ndarray]:
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
        try:
            ort_sess = self._create_inference_session(path)
            input_name = ort_sess.get_inputs()[0].name
            output_name = ort_sess.get_outputs()[0].name
            if ort_sess.get_inputs()[0].type == "tensor(double)":
                outputs = ort_sess.run(
                    [output_name],
                    {input_name: np.asarray(inputs).astype(np.float64)},
                )
            else:
                outputs = ort_sess.run([output_name], {input_name: np.asarray(inputs)})

        except (
            ModelConversionError,
            RuntimeError,
            ValueError,
            TypeError,
            OSError,
            Exception,
        ) as e:
            raise InferenceError(
                model_path=path,
                model_type=self.model_type,
                reason=str(e),
            ) from e
        else:
            return outputs

    def _collect_constant_values(
        self: ONNXConverter,
        model: onnx.ModelProto,
    ) -> dict[str, np.ndarray]:
        """Collect constant values from Constant nodes in the model.

        Args:
            model (onnx.ModelProto): The ONNX model to analyze.

        Returns:
            dict[str, np.ndarray]: Mapping of output name to constant value.
        """
        constant_values = {}
        for node in model.graph.node:
            if node.op_type == "Constant":
                self.logger.debug("Constant node", extra={"node": node})
                for attr in node.attribute:
                    if attr.name == "value":
                        tensor = attr.t
                        const_value = numpy_helper.to_array(tensor)
                        constant_values[node.output[0]] = const_value
        return constant_values

    def _attach_constant_parameters(
        self: ONNXConverter,
        layer: ONNXLayer,
        node: NodeProto,
        constant_values: dict[str, np.ndarray],
    ) -> None:
        """Attach constant inputs as parameters to a layer.

        Args:
            layer (ONNXLayer): The layer to modify.
            node (NodeProto): The ONNX node being processed.
            constant_values (dict[str, np.ndarray]): Constant values mapping.
        """
        for input_name in node.input:
            if input_name in constant_values:
                self.logger.debug(
                    "Layer params before:",
                    extra={"layer_params": layer.params},
                )
                if not hasattr(layer, "params") or layer.params is None:
                    layer.params = {}
                result = constant_values[input_name]
                if isinstance(result, (np.ndarray, torch.Tensor)):
                    layer.params[input_name] = result.tolist()
                else:
                    layer.params[input_name] = constant_values[input_name]
                self.logger.debug(
                    "Updated layer params",
                    extra={"layer_params": layer.params},
                )

    def get_model_architecture(
        self: ONNXConverter,
        model: onnx.ModelProto,
        output_name_to_shape: dict[str, list[int]],
        domain_to_version: dict[str, int] | None = None,
    ) -> list[ONNXLayer]:
        """Construct ONNXLayer objects for architecture graph nodes
        (not weights or biases).

        Args:
            model (onnx.ModelProto): The ONNX model to analyze.
            output_name_to_shape (dict[str, list[int]]):
                Map of value name -> inferred shape.
            id_count (int, optional):
                Starting numeric ID for layers (incremented per node).
                Defaults to 0.
            domain_to_version (dict[str, int], optional):
                Map of opset domain -> version used. Defaults to None.

        Returns:
            list[ONNXLayer]:
                Models computational layers
                (excluding initializers) in the form of ONNXLayers.
        """
        _ = domain_to_version
        constant_values = self._collect_constant_values(model)
        layers = []
        current_id = 0

        for node in model.graph.node:
            if node.op_type == "Constant":
                continue  # Skip constant nodes

            layer = self.analyze_layer(
                node,
                output_name_to_shape,
                current_id,
                domain_to_version,
            )
            self.logger.debug(
                "Layer",
                extra={
                    "layer_name": layer.name,
                    "layer_op": layer.op_type,
                    "layer_shape": layer.shape,
                },
            )

            self._attach_constant_parameters(layer, node, constant_values)
            layers.append(layer)
            current_id += 1

        return layers

    def get_model_w_and_b(
        self: ONNXConverter,
        model: onnx.ModelProto,
        output_name_to_shape: dict[str, list[int]],
        id_count: int = 0,
        domain_to_version: dict[str, int] | None = None,
    ) -> list[ONNXLayer]:
        """Extract constant initializers (weights/biases) as layers.

        Iterates through graph initializers and wraps each tensor
        into an ONNXLayers.

        Args:
            model (onnx.ModelProto): The ONNX model to analyze.
            output_name_to_shape (dict[str, list[int]]):
                Map of value name -> inferred shape
            id_count (int, optional):
                Starting numeric ID for layers (incremented per tensor).
                Defaults to 0.
            domain_to_version (dict[str, int], optional):
                Map of opset domain -> version used (unused). Defaults to None.

        Returns:
            list[ONNXLayer]: ONNXLayers representing weights/biases found in the graph
        """
        layers = []
        # Check the model and print Y"s shape information
        for _, node in enumerate(model.graph.initializer):
            layer = self.analyze_constant(
                node,
                output_name_to_shape,
                id_count,
                domain_to_version,
            )
            layers.append(layer)
            id_count += 1

        return layers

    def _create_inference_session(
        self: ONNXConverter,
        model_path: str,
    ) -> InferenceSession:
        """Internal helper to create and configure an ONNX Runtime InferenceSession.
        Registers a custom ops shared library for use with the
        custom quantized operations.

        Args:
            model_path (str): Path to the ONNX model to load.

        Returns:
            InferenceSession: A configured InferenceSession.
        """
        try:
            opts = SessionOptions()
            opts.register_custom_ops_library(get_library_path())
            return InferenceSession(
                model_path,
                opts,
                providers=["CPUExecutionProvider"],
            )
        except (OSError, RuntimeError, Exception) as e:
            raise InferenceError(
                model_path=model_path,
                model_type=self.model_type,
                reason=str(e),
            ) from e

    def analyze_layer(
        self: ONNXConverter,
        node: NodeProto,
        output_name_to_shape: dict[str, list[int]],
        id_count: int = -1,
        domain_to_version: dict[str, int] | None = None,
    ) -> ONNXLayer:
        """Convert a non-constant ONNX node into a structured ONNXLayer.

        Args:
            node (NodeProto): The ONNX node to analyze.
            output_name_to_shape (dict[str, list[int]]):
            Map of value name -> inferred shape.
            id_count (int, optional):
                Numeric ID to assign to this layer (increment handled by caller).
                Defaults to -1.
            domain_to_version (dict[str, int], optional):
                Map of opset domain -> version number. Defaults to None.

        Returns:
            ONNXLayer: ONNXLayer describing the node
        """
        name = node.name
        layer_id = id_count
        id_count += 1
        op_type = node.op_type
        inputs = node.input
        outputs = node.output
        opset_version = (
            domain_to_version.get(node.domain, "unknown") if domain_to_version else -1
        )
        params = parse_attributes(node.attribute)

        # Extract output shapes
        output_shapes = {
            out_name: output_name_to_shape.get(out_name, []) for out_name in outputs
        }

        return ONNXLayer(
            id=layer_id,
            name=name,
            op_type=op_type,
            inputs=list(inputs),
            outputs=list(outputs),
            shape=output_shapes,
            params=params,
            opset_version_number=opset_version,
            tensor=None,
        )

    def analyze_constant(
        self: ONNXConverter,
        node: TensorProto,
        output_name_to_shape: dict[str, list[int]],
        id_count: int = -1,
        domain_to_version: dict[str, int] | None = None,
    ) -> list[ONNXLayer]:
        """Convert a constant ONNX node (weights or bias) into a structured ONNXLayer.

        Args:
            node (NodeProto): The ONNX node to analyze.
            output_name_to_shape (dict[str, list[int]]):
                 Map of value name -> inferred shape.
            id_count (int, optional):
                Numeric ID to assign to this layer (increment handled by caller).
                Defaults to -1.
            domain_to_version (dict[str, int], optional):
                Map of opset domain -> version number. Defaults to None.

        Returns:
            ONNXLayer: ONNXLayer describing the node
        """
        _ = domain_to_version
        name = node.name
        id_count += 1
        op_type = "Const"
        inputs = []
        outputs = []
        opset_version = -1
        params = {}
        constant_dtype = node.data_type
        # Can do this step in rust potentially to keep file sizes low if needed
        try:
            np_data = onnx.numpy_helper.to_array(node, constant_dtype)
        except (ValueError, TypeError, onnx.ONNXException, Exception) as e:
            raise SerializationError(
                model_type=self.model_type,
                tensor_name=node.name,
                reason=f"Failed to convert tensor: {e!s}",
            ) from e
        # 💡 Extract output shapes
        output_shapes = {
            out_name: output_name_to_shape.get(out_name, []) for out_name in outputs
        }
        return ONNXLayer(
            id=id_count,
            name=name,
            op_type=op_type,
            inputs=list(inputs),
            outputs=list(outputs),
            shape=output_shapes,
            params=params,
            opset_version_number=opset_version,
            tensor=np_data.tolist(),
        )

    def _prepare_model_for_quantization(
        self: ONNXConverter,
        unscaled_model: onnx.ModelProto,
    ) -> tuple[onnx.ModelProto, dict[str, onnx.TensorProto], list[str]]:
        """Prepare the model for quantization by creating a copy and necessary mappings.

        Args:
            unscaled_model (onnx.ModelProto): The original unscaled model.

        Returns:
            tuple[onnx.ModelProto, dict[str, onnx.TensorProto], list[str]]:
                Model copy, initializer map, and input names.
        """
        model = copy.deepcopy(unscaled_model)
        self.op_quantizer.check_model(model)
        initializer_map = {init.name: init for init in model.graph.initializer}
        input_names = [inp.name for inp in unscaled_model.graph.input]
        return model, initializer_map, input_names

    def _quantize_inputs(
        self: ONNXConverter,
        model: onnx.ModelProto,
        input_names: list[str],
        scale_base: int,
        scale_exponent: int,
    ) -> list[onnx.NodeProto]:
        """Quantize model inputs and update node connections.

        Args:
            model (onnx.ModelProto): The model being quantized.
            input_names (list[str]): Names of input tensors.
            scale_base (int): Base for scaling.
            scale_exponent (int): Exponent for scaling.

        Returns:
            list[onnx.NodeProto]: New nodes created for input quantization.
        """
        new_nodes = []
        for name in input_names:
            output_name, mul_node, _, cast_to_int64 = self.quantize_input(
                input_name=name,
                op_quantizer=self.op_quantizer,
                scale_base=scale_base,
                scale_exponent=scale_exponent,
            )
            new_nodes.extend([mul_node, cast_to_int64])

            # Update references to this input in all nodes
            for node in model.graph.node:
                for idx, inp in enumerate(node.input):
                    if inp == name:
                        node.input[idx] = output_name
        return new_nodes

    def _update_input_types(self: ONNXConverter, model: onnx.ModelProto) -> None:
        """Update input tensor types from float32 to float64.

        Args:
            model (onnx.ModelProto): The model to update.
        """
        for input_tensor in model.graph.input:
            tensor_type = input_tensor.type.tensor_type
            if tensor_type.elem_type == TensorProto.FLOAT:
                tensor_type.elem_type = TensorProto.DOUBLE

    def _quantize_nodes(
        self: ONNXConverter,
        model: onnx.ModelProto,
        scale_config: ScaleConfig,
        rescale_config: dict | None,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.NodeProto]:
        """Quantize all nodes in the model.

        Args:
            model (onnx.ModelProto): The model being quantized.
            scale_base (int): Base for scaling.
            scale_exponent (int): Exponent for scaling.
            rescale_config (dict, optional): Rescale configuration.
            initializer_map (dict[str, onnx.TensorProto]): Initializer mapping.

        Returns:
            list[onnx.NodeProto]: Quantized nodes.
        """
        quantized_nodes = []
        for node in model.graph.node:
            rescale = rescale_config.get(node.name, False) if rescale_config else True
            quant_nodes = self.quantize_layer(
                node=node,
                model=model,
                scale_config=ScaleConfig(
                    exponent=scale_config.exponent,
                    base=scale_config.base,
                    rescale=rescale,
                ),
                initializer_map=initializer_map,
            )
            if isinstance(quant_nodes, list):
                quantized_nodes.extend(quant_nodes)
            else:
                quantized_nodes.append(quant_nodes)
        return quantized_nodes

    def _process_initializers(
        self: ONNXConverter,
        model: onnx.ModelProto,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> list[onnx.TensorProto]:
        """Process and filter initializers, converting types as needed.

        Args:
            model (onnx.ModelProto): The quantized model.
            initializer_map (dict[str, onnx.TensorProto]): Original initializer map.

        Returns:
            list[onnx.TensorProto]: Processed initializers to keep.
        """
        used_initializer_names = set()
        for node in model.graph.node:
            used_initializer_names.update(node.input)

        kept_initializers = []
        for name in used_initializer_names:
            if name in initializer_map:
                orig_init = initializer_map[name]
                np_array = numpy_helper.to_array(orig_init)

                if np_array.dtype == np.float32:
                    np_array = np_array.astype(np.float64)
                    new_init = numpy_helper.from_array(np_array, name=name)
                    kept_initializers.append(new_init)
                else:
                    kept_initializers.append(orig_init)

        return kept_initializers

    def _update_graph_types(self: ONNXConverter, model: onnx.ModelProto) -> None:
        """Update output and value_info types to INT64.

        Args:
            model (onnx.ModelProto): The model to update.
        """
        for out in model.graph.output:
            out.type.tensor_type.elem_type = onnx.TensorProto.INT64

        for vi in model.graph.value_info:
            vi.type.tensor_type.elem_type = TensorProto.INT64

    def _add_custom_domain(self: ONNXConverter, model: onnx.ModelProto) -> None:
        """Add custom opset domain if not present.

        Args:
            model (onnx.ModelProto): The model to update.
        """
        custom_domain = helper.make_operatorsetid(
            domain="ai.onnx.contrib",
            version=1,
        )
        domains = [op.domain for op in model.opset_import]
        if "ai.onnx.contrib" not in domains:
            model.opset_import.append(custom_domain)

    def _log_quantization_results(self: ONNXConverter, model: onnx.ModelProto) -> None:
        """Log quantization results for debugging.

        Args:
            model (onnx.ModelProto): The quantized model.
        """
        for layer in model.graph.node:
            self.logger.debug(
                "Node",
                extra={
                    "layer_name": layer.name,
                    "op_type": layer.op_type,
                    "input": layer.input,
                    "output": layer.output,
                },
            )

        for layer in model.graph.initializer:
            self.logger.debug("Initializer", extra={"layer_name": layer.name})

    def quantize_model(
        self: ONNXConverter,
        unscaled_model: onnx.ModelProto,
        scale_base: int,
        scale_exponent: int,
        rescale_config: dict | None = None,
    ) -> onnx.ModelProto:
        """Produce a quantized ONNX graph from a floating-point model.

        Args:
            unscaled_model (onnx.ModelProto): The original unscaled model.
            scale_base (int): Base for fixed-point scaling (e.g., 2).
            scale_exponent (int):
                Exponent for scaling (e.g., 18 would lead to a scale factor 2**18).
            rescale_config (dict, optional): mapping of node name -> bool to control
                whether a given node should apply a final rescale. Defaults to None.

        Returns:
            onnx.ModelProto: A new onnx model representation of the quantized model.
        """
        try:
            # Prepare model and mappings
            model, initializer_map, input_names = self._prepare_model_for_quantization(
                unscaled_model,
            )

            # Quantize inputs and collect new nodes
            new_nodes = self._quantize_inputs(
                model,
                input_names,
                scale_base,
                scale_exponent,
            )

            # Update input types
            self._update_input_types(model)

            # Quantize all nodes
            quantized_nodes = self._quantize_nodes(
                model,
                ScaleConfig(scale_exponent, scale_base, rescale=True),
                rescale_config,
                initializer_map,
            )
            new_nodes.extend(quantized_nodes)

            # Update graph with new nodes
            model.graph.ClearField("node")
            model.graph.node.extend(new_nodes)

            # Process initializers
            kept_initializers = self._process_initializers(model, initializer_map)

            # Update graph initializers
            model.graph.ClearField("initializer")
            model.graph.initializer.extend(kept_initializers)
            model.graph.initializer.extend(self.op_quantizer.new_initializers)
            self.op_quantizer.new_initializers = []

            # Update types and add custom domain
            self._update_graph_types(model)
            self._add_custom_domain(model)

            # Log results
            self._log_quantization_results(model)

        except (QuantizationError, ModelConversionError):
            raise
        except (
            onnx.ONNXException,
            ValueError,
            TypeError,
            RuntimeError,
            OSError,
            Exception,
        ) as e:
            msg = (
                "Quantization failed for model"
                f" '{getattr(self, 'model_file_name', 'unknown')}': {e!s}"
            )
            raise ModelConversionError(
                msg,
                model_type=self.model_type,
            ) from e
        else:
            return model

    def quantize_layer(
        self: ONNXConverter,
        node: onnx.NodeProto,
        model: onnx.ModelProto,
        scale_config: ScaleConfig,
        initializer_map: dict[str, onnx.TensorProto],
    ) -> onnx.NodeProto:
        """Quantize a single ONNX node using the configured op quantizer.

        Args:
            node (onnx.NodeProto): The original onnx node to quantize.
            model (onnx.ModelProto): The original model used for context
            scale_config (ScaleConfig): Contains the following:
                - rescale (bool): Whether to apply output rescaling for this node.
                - scale_exponent (int):
                Exponent for scaling (e.g., 18 would lead to a scale factor 2**18).
                - scale_base (int): Base for fixed-point scaling (e.g., 2).
            initializer_map (dict[str, onnx.TensorProto]):
                Mapping from initializer name to tensor.

        Returns:
            onnx.NodeProto:
                A quantized node or list of nodes replacing the initial node.
        """
        try:
            return self.op_quantizer.quantize(
                node=node,
                rescale=scale_config.rescale,
                graph=model.graph,
                scale_exponent=scale_config.exponent,
                scale_base=scale_config.base,
                initializer_map=initializer_map,
            )
        except QuantizationError:
            raise
        except (RuntimeError, ValueError, TypeError, Exception) as e:
            raise ModelConversionError(str(e), model_type=self.model_type) from e

    def quantize_input(
        self: ONNXConverter,
        input_name: str,
        op_quantizer: ONNXOpQuantizer,
        scale_base: int,
        scale_exponent: int,
    ) -> tuple[str, onnx.NodeProto, onnx.NodeProto, onnx.NodeProto]:
        """Insert scaling and casting nodes to quantize a model input.

        Creates:
            - Mul: scales the input by scale_base ** scale.
            - Cast (to INT64): produces the final integer input tensor.

        Args:
            input_name (str): Name of the graph input to quantize.
            op_quantizer (ONNXOpQuantizer): The op quantizer whose
            ``new_initializers`` list is used to store the created scale constant.
            scale_base (int): Base for fixed-point scaling (e.g., 2).
            scale_exponent (int):
                Exponent for scaling (e.g., 18 would lead to a scale factor 2**18).

        Returns:
            tuple[str, onnx.NodeProto, onnx.NodeProto, onnx.NodeProto]:
            A tuple ``(output_name, mul_node, floor_node, cast_node)`` where
            ``output_name`` is the name of the quantized input tensor
            and the nodes are nodes to add to the graph.
        """
        try:
            scale_value = BaseOpQuantizer.get_scaling(
                scale_base=scale_base,
                scale_exponent=scale_exponent,
            )

            # === Create scale constant ===
            scale_const_name = input_name + "_scale"
            scale_tensor = numpy_helper.from_array(
                np.array([scale_value], dtype=np.float64),
                name=scale_const_name,
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
                name=rounded_output_name,
            )
        except (ValueError, TypeError, RuntimeError, OSError, Exception) as e:
            msg = f"Error quantizing inputs: {e}"
            raise ModelConversionError(
                msg,
                self.model_type,
            ) from e
        else:
            return output_name, mul_node, floor_node, cast_to_int64

    def _extract_model_io_info(
        self: ONNXConverter,
        onnx_model: onnx.ModelProto,
    ) -> None:
        """Populate input metadata from a loaded ONNX model.

        Args:
            onnx_model (onnx.ModelProto): Onnx model
        """
        self.required_keys = [
            graph_input.name for graph_input in onnx_model.graph.input
        ]
        self.input_shape = get_input_shapes(onnx_model)

    def get_weights(self: ONNXConverter) -> tuple[
        dict[str, list[ONNXLayerDict]],
        dict[str, list[ONNXLayerDict]],
        CircuitParamsDict,
    ]:
        """Export architecture, weights, and circuit parameters for ECC.

        1. Analyze the model for architecture + w & b
        2. Put arch into format to be read by ECC circuit builder
        3. Put w + b into format to be read by ECC circuit builder

        Returns:
            tuple[dict[str, list[dict[str, Any]]],
            dict[str, list[dict[str, Any]]], dict[str, Any]]:
                 A tuple ``(architecture, weights, circuit_params)``:
                - ``architecture``: dict with serialized ``architecture`` layers.
                - ``weights``: dict containing ``w_and_b`` (serialized tensors).
                - ``circuit_params``: dict containing scaling parameters and
                  ``rescale_config``.
        """
        inferred_model = shape_inference.infer_shapes(self.model)
        scale_base = getattr(self, "scale_base", 2)
        scale_exponent = getattr(self, "scale_exponent", 18)

        # Check the model and print Y"s shape information
        self._onnx_check_model_safely(inferred_model)
        output_name_to_shape = extract_shape_dict(inferred_model)
        scaled_and_transformed_model = self.op_quantizer.apply_pre_analysis_transforms(
            inferred_model,
            scale_exponent=scale_exponent,
            scale_base=scale_base,
        )
        # Get layers in correct format
        (architecture, w_and_b) = self.analyze_layers(
            scaled_and_transformed_model,
            output_name_to_shape,
        )

        def _convert_tensor_to_int_list(w: ONNXLayer) -> list:
            try:
                arr = np.asarray(w.tensor).astype(np.int64)
                return arr.tolist()
            except Exception as e:
                raise SerializationError(
                    tensor_name=getattr(w, "name", None),
                    model_type=self.model_type,
                    reason=f"cannot convert to ndarray: {e}",
                ) from e

        for w in w_and_b:
            w.tensor = _convert_tensor_to_int_list(w)

        inputs = []
        outputs = []
        for graph_input in self.model.graph.input:
            shape = output_name_to_shape.get(graph_input.name, [])
            elem_type = getattr(graph_input, "elem_type", -1)
            inputs.append(ONNXIO(graph_input.name, elem_type, shape))

        for output in self.model.graph.output:
            shape = output_name_to_shape.get(output.name, [])
            elem_type = getattr(output, "elem_type", -1)
            outputs.append(ONNXIO(output.name, elem_type, shape))

        # Get version from package metadata
        try:
            version = get_version(PACKAGE_NAME)
        except Exception:
            version = "0.0.0"

        architecture = {
            "architecture": [asdict(a) for a in architecture],
        }
        weights = {"w_and_b": [asdict(w_b) for w_b in w_and_b]}
        circuit_params = {
            "scale_base": getattr(self, "scale_base", 2),
            "scale_exponent": getattr(self, "scale_exponent", 18),
            "rescale_config": getattr(self, "rescale_config", {}),
            "inputs": [asdict(i) for i in inputs],
            "outputs": [asdict(o) for o in outputs],
            "version": version,
        }
        return architecture, weights, circuit_params

    def get_model_and_quantize(self: ONNXConverter) -> None:
        """Load the configured model (by path) and build its quantized form.

        Expects the instance to define ``self.model_file_name`` beforehand.

        Raises:
            FileNotFoundError: If ``self.model_file_name`` is unset or invalid.
        """
        if hasattr(self, "model_file_name"):
            self.load_model(self.model_file_name)
        else:
            msg = "An ONNX model is required at the specified path"
            raise FileNotFoundError(msg)
        self.quantized_model = self.quantize_model(
            self.model,
            getattr(self, "scale_base", 2),
            getattr(self, "scale_exponent", 18),
            rescale_config=getattr(self, "rescale_config", {}),
        )

    def _process_single_input_for_get_outputs(
        self: ONNXConverter,
        value: np.ndarray | torch.Tensor,
        input_def: NodeArg,
    ) -> np.ndarray:
        """Process a single input tensor according to dtype and scale settings."""
        value = torch.as_tensor(value)

        if value.dtype in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            value = value.double()
            value = value / BaseOpQuantizer.get_scaling(
                scale_base=self.scale_base,
                scale_exponent=self.scale_exponent,
            )

        if input_def.type == "tensor(double)":
            return np.asarray(value).astype(np.float64)
        return np.asarray(value)

    def get_outputs(
        self: ONNXConverter,
        inputs: np.ndarray | torch.Tensor | dict[str, np.ndarray | torch.Tensor],
    ) -> list[np.ndarray]:
        """Run the currently loaded (quantized) model via ONNX Runtime.

        Args:
            inputs: Single tensor/array or a dict of named inputs.

        Returns:
            list[np.ndarray]: List of output arrays from ONNX Runtime inference.
        """

        def _raise_type_error(inputs: np.ndarray | torch.Tensor) -> None:
            msg = (
                "Expected np.ndarray, torch.Tensor, or dict "
                f"for inputs, got {type(inputs)}"
            )
            raise TypeError(msg)

        def _raise_value_error(msg: str) -> None:
            raise ValueError(msg)

        def _raise_no_scale_configs() -> None:
            raise CircuitConfigurationError(
                missing_attributes=["scale_base", "scale_exponent"],
            )

        scale_base = getattr(self, "scale_base", None)
        scale_exponent = getattr(self, "scale_exponent", None)

        try:
            input_defs = self.ort_sess.get_inputs()
            output_defs = self.ort_sess.get_outputs()
            output_names = [out.name for out in output_defs]

            if scale_base is None or scale_exponent is None:
                _raise_no_scale_configs()

            # Normalize inputs into a dict
            if isinstance(inputs, (np.ndarray, torch.Tensor)):
                input_name = input_defs[0].name
                inputs = {input_name: inputs}
            elif not isinstance(inputs, dict):
                _raise_type_error(inputs)

            # Process inputs
            processed_inputs = {}
            for input_def in input_defs:
                name = input_def.name
                if name not in inputs:
                    _raise_value_error(
                        f"Missing required input '{name}' in provided inputs",
                    )
                processed_inputs[name] = self._process_single_input_for_get_outputs(
                    inputs[name],
                    input_def,
                )

            return self.ort_sess.run(output_names, processed_inputs)

        except (RuntimeError, ValueError, TypeError, Exception) as e:
            raise InferenceError(
                model_path=getattr(self, "quantized_model_path", None),
                model_type=self.model_type,
                reason=str(e),
            ) from e


if __name__ == "__main__":
    path = "./models_onnx/doom.onnx"

    converter = ONNXConverter()
    converter.model_file_name, converter.quantized_model_file_name = (
        path,
        "quantized_doom.onnx",
    )
    converter.scale_base, converter.scale_exponent = 2, 18

    converter.load_model(path)
    converter.get_model_and_quantize()

    converter.test_accuracy()
