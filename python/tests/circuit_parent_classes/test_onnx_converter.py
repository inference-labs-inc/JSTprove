# test_converter.py
import os
import pytest
from unittest.mock import MagicMock, patch

import torch

from python.core.model_processing.converters.onnx_converter import ONNXConverter

import onnx
import tempfile
from onnx import helper, TensorProto
import onnxruntime as ort


@pytest.fixture
def temp_model_path(tmp_path):
    model_path = tmp_path / "temp_model.onnx"
    # Give it to the test
    yield model_path

    # After the test is done, remove it
    if os.path.exists(model_path):
        model_path.unlink()

@pytest.fixture
def temp_quant_model_path(tmp_path):
    model_path = tmp_path / "temp_quantized_model.onnx"
    # Give it to the test
    yield model_path

    # After the test is done, remove it
    if os.path.exists(model_path):
        model_path.unlink()

@pytest.fixture
def converter(temp_model_path, temp_quant_model_path):
    conv = ONNXConverter()
    conv.model = MagicMock(name="model")
    conv.quantized_model = MagicMock(name="quantized_model")
    return conv

@pytest.mark.unit
@patch("python.core.model_processing.converters.onnx_converter.onnx.save")
def test_save_model(mock_save, converter):
    path = "model.onnx"
    converter.save_model(path)
    mock_save.assert_called_once_with(converter.model, path)

@pytest.mark.unit
@patch("python.core.model_processing.converters.onnx_converter.onnx.load")
@patch("python.core.model_processing.converters.onnx_converter.onnx.checker.check_model")
def test_load_model(mock_check, mock_load, converter):
    fake_model = MagicMock(name="onnx_model")
    mock_load.return_value = fake_model

    path = "model.onnx"
    converter.load_model(path)

    mock_load.assert_called_once_with(path)
    # mock_check.assert_called_once_with(fake_model)
    assert converter.model == fake_model

@pytest.mark.unit
@patch("python.core.model_processing.converters.onnx_converter.onnx.save")
def test_save_quantized_model(mock_save, converter):
    path = "quantized_model.onnx"
    converter.save_quantized_model(path)
    mock_save.assert_called_once_with(converter.quantized_model, path)

@pytest.mark.unit
@patch("python.core.model_processing.converters.onnx_converter.SessionOptions")
@patch("python.core.model_processing.converters.onnx_converter.InferenceSession")
@patch("python.core.model_processing.converters.onnx_converter.onnx.checker.check_model")
@patch("python.core.model_processing.converters.onnx_converter.onnx.load")
def test_load_quantized_model(mock_load, mock_check, mock_ort_sess, mock_session_opts, converter):

    fake_model = MagicMock(name="onnx_model")
    mock_load.return_value = fake_model

    mock_opts_instance = MagicMock(name="session_options")
    mock_session_opts.return_value = mock_opts_instance

    path = "quantized_model.onnx"
    converter.load_quantized_model(path)

    mock_load.assert_called_once_with(path)
    # mock_check.assert_called_once_with(fake_model)
    mock_ort_sess.assert_called_once_with(path, mock_opts_instance, providers=["CPUExecutionProvider"])
    assert converter.quantized_model == fake_model

@pytest.mark.unit
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

@pytest.mark.integration
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
@pytest.mark.integration
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
