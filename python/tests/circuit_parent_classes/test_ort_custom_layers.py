from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, shape_inference

from python.core.model_processing.converters.onnx_converter import ONNXConverter


@pytest.fixture
def tiny_conv_model_path(tmp_path: Path) -> Path:
    # Create input and output tensor info
    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 1, 2, 2])

    # Kernel weights (3x3 ones)
    w_init = helper.make_tensor(
        name="W",
        data_type=TensorProto.FLOAT,
        dims=[1, 1, 3, 3],
        vals=np.ones((1 * 1 * 3 * 3), dtype=np.float32).tolist(),
    )
    z_init = helper.make_tensor(
        name="Z",
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=np.ones((1), dtype=np.float32).tolist(),
    )

    # Conv node with no padding, stride 1
    conv_node = helper.make_node(
        "Conv",
        inputs=["X", "W", "Z"],
        outputs=["Y"],
        kernel_shape=[3, 3],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
        dilations=[1, 1],
    )

    # Build graph and model
    graph = helper.make_graph(
        nodes=[conv_node],
        name="TinyConvGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[w_init, z_init],
    )

    model = helper.make_model(graph, producer_name="tiny-conv-example")

    # Save to a temporary file
    model_path = tmp_path / "tiny_conv.onnx"
    onnx.save(model, str(model_path))

    return model_path


@pytest.mark.integration
def test_tiny_conv(tiny_conv_model_path: Path, tmp_path: Path) -> None:
    path = tiny_conv_model_path

    converter = ONNXConverter()

    # Load and validate original model
    model = onnx.load(path)
    onnx.checker.check_model(model)

    # Apply shape inference and validate
    inferred_model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(inferred_model)

    # Quantize and add custom domain
    new_model = converter.quantize_model(model, 2, 21)
    custom_domain = onnx.helper.make_operatorsetid(domain="ai.onnx.contrib", version=1)
    new_model.opset_import.append(custom_domain)
    onnx.checker.check_model(new_model)

    # Save quantized model
    out_path = tmp_path / "model_quant.onnx"
    with out_path.open("wb") as f:
        f.write(new_model.SerializeToString())

    # Reload quantized model to ensure it is valid
    model_quant = onnx.load(str(out_path))
    onnx.checker.check_model(model_quant)

    # Prepare inputs and compare outputs
    inputs = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    outputs_true = converter.run_model_onnx_runtime(path, inputs)
    outputs_quant = converter.run_model_onnx_runtime(out_path, inputs)

    true = torch.tensor(np.array(outputs_true), dtype=torch.float32)
    quant = torch.tensor(np.array(outputs_quant), dtype=torch.float32) / (2**21)

    assert torch.allclose(true, quant, rtol=1e-3, atol=1e-5), "Outputs do not match"
