import pytest
import numpy as np
import torch
import onnx

from onnx import TensorProto, shape_inference, helper, numpy_helper

from python.core.model_processing.converters.onnx_converter import ONNXConverter
from python.core.model_processing.onnx_custom_ops.onnx_helpers import extract_shape_dict
from python.core.model_processing.onnx_quantizer.onnx_op_quantizer import ONNXOpQuantizer

from onnxruntime import InferenceSession, SessionOptions
from onnxruntime_extensions import get_library_path, OrtPyFunction
from python.core.model_processing.onnx_custom_ops import conv

from python.core.model_processing.onnx_custom_ops.conv import int64_conv
from python.core.model_processing.onnx_custom_ops.gemm import int64_gemm7


@pytest.fixture
def tiny_conv_model_path(tmp_path):
    # Create input and output tensor info
    input_tensor = helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 1, 4, 4])
    output_tensor = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 1, 2, 2])

    # Kernel weights (3x3 ones)
    W_init = helper.make_tensor(
        name='W',
        data_type=TensorProto.FLOAT,
        dims=[1, 1, 3, 3],
        vals=np.ones((1 * 1 * 3 * 3), dtype=np.float32).tolist()
    )
    Z_init = helper.make_tensor(
        name='Z',
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=np.ones(( 1), dtype=np.float32).tolist()
    )

    # Conv node with no padding, stride 1
    conv_node = helper.make_node(
        'Conv',
        inputs=['X', 'W', 'Z'],
        outputs=['Y'],
        kernel_shape=[3, 3],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
        dilations = [1,1]
    )

    # Build graph and model
    graph = helper.make_graph(
        nodes=[conv_node],
        name='TinyConvGraph',
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[W_init, Z_init]
    )

    model = helper.make_model(graph, producer_name='tiny-conv-example')

    # Save to a temporary file
    model_path = tmp_path / "tiny_conv.onnx"
    onnx.save(model, str(model_path))

    return str(model_path)

@pytest.mark.integration
def test_tiny_conv(tiny_conv_model_path):
    path = tiny_conv_model_path

    converter = ONNXConverter()

    X_input = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    id_count = 0
    model = onnx.load(path)
    # Fix, can remove this next line 
    onnx.checker.check_model(model)

    # Check the model and print Y"s shape information
    onnx.checker.check_model(model)
    print(f"Before shape inference, the shape info of Y is:\n{model.graph.value_info}")

    # Apply shape inference on the model
    inferred_model = shape_inference.infer_shapes(model)

    # Check the model and print Y"s shape information
    onnx.checker.check_model(inferred_model)
    # print(f"After shape inference, the shape info of Y is:\n{inferred_model.graph.value_info}")
    

    domain_to_version = {opset.domain: opset.version for opset in model.opset_import}
    
    inferred_model = shape_inference.infer_shapes(model)
    output_name_to_shape = extract_shape_dict(inferred_model)
    id_count = 0

    new_model = converter.quantize_model(model, 2, 21)
    custom_domain = onnx.helper.make_operatorsetid(domain="ai.onnx.contrib", version=1)
    new_model.opset_import.append(custom_domain)
    onnx.checker.check_model(new_model)

    with open("model.onnx", "wb") as f:
        f.write(new_model.SerializeToString())

    model = onnx.load("model.onnx")
    onnx.checker.check_model(model)  # This throws a descriptive error

    inputs = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    outputs_true = converter.run_model_onnx_runtime(path, inputs)

    outputs_quant = converter.run_model_onnx_runtime("model.onnx", inputs)
    true = torch.tensor(np.array(outputs_true), dtype=torch.float32)
    quant = torch.tensor(np.array(outputs_quant), dtype=torch.float32) / (2**21)

    assert torch.allclose(true, quant, rtol=1e-3, atol=1e-5), "Outputs do not match"
