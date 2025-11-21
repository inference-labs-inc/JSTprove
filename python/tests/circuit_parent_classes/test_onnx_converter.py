# test_converter.py
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
from onnx import TensorProto, helper

from python.core.model_processing.converters.onnx_converter import ONNXConverter


@pytest.fixture
def temp_model_path(
    tmp_path: Generator[Path, None, None],
) -> Generator[Path, Any, None]:
    model_path = tmp_path / "temp_model.onnx"
    # Give it to the test
    yield model_path

    # After the test is done, remove it
    if Path.exists(model_path):
        model_path.unlink()


@pytest.fixture
def temp_quant_model_path(
    tmp_path: Generator[Path, None, None],
) -> Generator[Path, Any, None]:
    model_path = tmp_path / "temp_quantized_model.onnx"
    # Give it to the test
    yield model_path

    # After the test is done, remove it
    if Path.exists(model_path):
        model_path.unlink()


@pytest.fixture
def converter() -> ONNXConverter:
    conv = ONNXConverter()
    conv.model = MagicMock(name="model")
    conv.quantized_model = MagicMock(name="quantized_model")
    return conv


@pytest.mark.unit
@patch("python.core.model_processing.converters.onnx_converter.onnx.save")
def test_save_model(mock_save: MagicMock, converter: ONNXConverter) -> None:
    path = "model.onnx"
    converter.save_model(path)
    mock_save.assert_called_once_with(converter.model, path)


@pytest.mark.unit
@patch("python.core.model_processing.converters.onnx_converter.onnx.load")
def test_load_model(mock_load: MagicMock, converter: ONNXConverter) -> None:
    fake_model = MagicMock(name="onnx_model")
    mock_load.return_value = fake_model

    path = "model.onnx"
    converter.load_model(path)

    mock_load.assert_called_once_with(path)
    assert converter.model == fake_model


@pytest.mark.unit
@patch("python.core.model_processing.converters.onnx_converter.onnx.save")
def test_save_quantized_model(mock_save: MagicMock, converter: ONNXConverter) -> None:
    path = "quantized_model.onnx"
    converter.save_quantized_model(path)
    mock_save.assert_called_once_with(converter.quantized_model, path)


@pytest.mark.unit
@patch("python.core.model_processing.converters.onnx_converter.Path.exists")
@patch("python.core.model_processing.converters.onnx_converter.SessionOptions")
@patch("python.core.model_processing.converters.onnx_converter.InferenceSession")
@patch("python.core.model_processing.converters.onnx_converter.onnx.load")
def test_load_quantized_model(
    mock_load: MagicMock,
    mock_ort_sess: MagicMock,
    mock_session_opts: MagicMock,
    mock_exists: MagicMock,
    converter: ONNXConverter,
) -> None:

    fake_model = MagicMock(name="onnx_model")
    mock_load.return_value = fake_model
    mock_exists.return_value = True  # Mock os.path.exists to return True

    mock_opts_instance = MagicMock(name="session_options")
    mock_session_opts.return_value = mock_opts_instance

    path = "quantized_model.onnx"
    converter.load_quantized_model(path)

    mock_load.assert_called_once_with(path)
    mock_ort_sess.assert_called_once_with(
        path,
        mock_opts_instance,
        providers=["CPUExecutionProvider"],
    )
    assert converter.quantized_model == fake_model


@pytest.mark.unit
def test_get_outputs_with_mocked_session(converter: ONNXConverter) -> None:
    dummy_input = np.array([[1.0]])  # Use np.ndarray, not list
    dummy_output = [[2.0]]
    converter.scale_base = 2
    converter.scale_exponent = 10

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

    # Expect NumPy array to be passed into ort_sess.run()
    expected_call_inputs = {"input": np.asarray(dummy_input)}
    mock_sess.run.assert_called_once_with(["output"], expected_call_inputs)

    assert result == dummy_output


# Integration test


def create_dummy_model() -> onnx.ModelProto:
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
    graph = helper.make_graph([node], "test-graph", [input_tensor], [output_tensor])

    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 21)])


@pytest.mark.integration
def test_save_and_load_real_model() -> None:
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


@pytest.mark.integration
def test_real_inference_from_onnx() -> None:
    converter = ONNXConverter()
    converter.model = create_dummy_model()
    converter.scale_base = 2
    converter.scale_exponent = 10

    # Save and load into onnxruntime
    with tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        onnx.save(converter.model, tmp.name)
        converter.ort_sess = ort.InferenceSession(
            tmp.name,
            providers=["CPUExecutionProvider"],
        )

        dummy_input = torch.tensor([1.0], dtype=torch.float32).numpy()
        result = converter.get_outputs(dummy_input)

        assert isinstance(result, list)
        print(result)  # Identity op should return input
