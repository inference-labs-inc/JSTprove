# test_converter.py
import pytest
from unittest.mock import MagicMock, patch

import torch

from python_testing.utils.onnx_converter import ONNXConverter

import onnx
import tempfile
from onnx import helper, TensorProto
import onnxruntime as ort

@pytest.fixture
def converter():
    conv = ONNXConverter()
    conv.model = MagicMock(name="model")
    conv.quantized_model = MagicMock(name="quantized_model")
    return conv

@patch("python_testing.utils.onnx_converter.onnx.save")
def test_save_model(mock_save, converter):
    path = "model.onnx"
    converter.save_model(path)
    mock_save.assert_called_once_with(converter.model, path)

@patch("python_testing.utils.onnx_converter.onnx.load")
@patch("python_testing.utils.onnx_converter.onnx.checker.check_model")
def test_load_model(mock_check, mock_load, converter):
    fake_model = MagicMock(name="onnx_model")
    mock_load.return_value = fake_model

    path = "model.onnx"
    converter.load_model(path)

    mock_load.assert_called_once_with(path)
    mock_check.assert_called_once_with(fake_model)
    assert converter.model == fake_model

@patch("python_testing.utils.onnx_converter.onnx.save")
def test_save_quantized_model(mock_save, converter):
    path = "quantized_model.onnx"
    converter.save_quantized_model(path)
    mock_save.assert_called_once_with(converter.quantized_model, path)

@patch("python_testing.utils.onnx_converter.onnx.load")
@patch("python_testing.utils.onnx_converter.onnx.checker.check_model")
@patch("python_testing.utils.onnx_converter.ort.InferenceSession")
def test_load_quantized_model(mock_ort_sess, mock_check, mock_load, converter):
    fake_model = MagicMock(name="onnx_model")
    mock_load.return_value = fake_model

    path = "quantized_model.onnx"
    converter.load_quantized_model(path)

    mock_load.assert_called_once_with(path)
    mock_check.assert_called_once_with(fake_model)
    mock_ort_sess.assert_called_once_with(path, providers=["CPUExecutionProvider"])

    assert converter.quantized_model == fake_model


def test_get_outputs_with_mocked_session(converter):
    dummy_input = [[1.0]]
    dummy_output = [[2.0]]

    mock_sess = MagicMock()

    # Mock .get_inputs()[0].name => "input"
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_sess.get_inputs.return_value = [mock_input]

    # Mock .get_outputs()[0].name => "output"
    mock_output = MagicMock()
    mock_output.name = "output"
    mock_sess.get_outputs.return_value = [mock_output]

    # Mock .run() output
    mock_sess.run.return_value = dummy_output

    converter.ort_sess = mock_sess

    result = converter.get_outputs(dummy_input)

    mock_sess.run.assert_called_once_with(["output"], {"input": dummy_input})
    assert result == dummy_output


# Integration test


def create_dummy_model():
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([node], "test-graph", [input_tensor], [output_tensor])
    
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])
    return model

def test_save_and_load_real_model():
    converter = ONNXConverter()
    model = create_dummy_model()
    converter.model = model
    converter.quantized_model = model

    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        # Save model
        converter.save_model(tmp.name)

        # Load model
        converter.load_model(tmp.name)

        # Validate loaded model
        assert isinstance(converter.model, onnx.ModelProto)
        assert converter.model.graph.name == model.graph.name
        assert len(converter.model.graph.node) == 1
        assert converter.model.graph.node[0].op_type == "Identity"

        # Save model
        converter.save_quantized_model(tmp.name)

        # Load model
        converter.load_quantized_model(tmp.name)

        # Validate loaded model
        assert isinstance(converter.model, onnx.ModelProto)
        assert converter.model.graph.name == model.graph.name
        assert len(converter.model.graph.node) == 1
        assert converter.model.graph.node[0].op_type == "Identity"


# def test_save_and_load_large_model():
#     pass

def test_real_inference_from_onnx():
    converter = ONNXConverter()
    converter.model = create_dummy_model()

    # Save and load into onnxruntime
    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        onnx.save(converter.model, tmp.name)
        converter.ort_sess = ort.InferenceSession(tmp.name, providers=["CPUExecutionProvider"])

        dummy_input = torch.tensor([1.0], dtype=torch.float32).numpy()
        result = converter.get_outputs(dummy_input)

        assert isinstance(result, list)
        print(result) # Identity op should return input


def test_analyze_layers():
    converter = ONNXConverter()
    converter.model = create_dummy_model()
    converter.analyze_layers(converter.model)

    assert False
    
    # # Save and load into onnxruntime
    # with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
    #     onnx.save(converter.model, tmp.name)
    #     converter.ort_sess = ort.InferenceSession(tmp.name, providers=["CPUExecutionProvider"])

    #     dummy_input = torch.tensor([1.0], dtype=torch.float32).numpy()
    #     result = converter.get_outputs(dummy_input)

    #     assert isinstance(result, list)
    #     print(result) # Identity op should return input
    #     # assert False