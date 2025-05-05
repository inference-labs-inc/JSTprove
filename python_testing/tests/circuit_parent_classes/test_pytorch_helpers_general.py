import pytest
import torch
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, mock_open
from python_testing.utils.pytorch_helpers import GeneralLayerFunctions


def test_check_4d_eq_pass():
    g = GeneralLayerFunctions()
    t1 = torch.ones((2, 2, 2, 2))
    t2 = torch.ones((2, 2, 2, 2)) + 0.5
    g.check_4d_eq(t1, t2)  # Should not raise


def test_check_4d_eq_fail():
    g = GeneralLayerFunctions()
    t1 = torch.ones((1, 1, 1, 1))
    t2 = torch.zeros((1, 1, 1, 1))
    with pytest.raises(AssertionError):
        g.check_4d_eq(t1, t2)


def test_check_2d_eq_pass():
    g = GeneralLayerFunctions()
    t1 = torch.tensor([[5.0, 5.5]])
    t2 = torch.tensor([[5.4, 5.9]])
    g.check_2d_eq(t1, t2)


def test_check_2d_eq_fail():
    g = GeneralLayerFunctions()
    t1 = torch.tensor([[1.0]])
    t2 = torch.tensor([[3.0]])
    with pytest.raises(AssertionError):
        g.check_2d_eq(t1, t2)

# --------- ONNX to Torch Format ---------

@patch("python_testing.utils.pytorch_helpers.torch.tensor")
@patch("python_testing.utils.pytorch_helpers.onnx.numpy_helper.to_array")
def test_weights_onnx_to_torch_format(mock_to_array, mock_tensor):
    g = GeneralLayerFunctions()

    mock_initializer = [MagicMock(name="conv1.weight"), MagicMock(name="conv1.bias")]
    mock_initializer[0].name = "conv1.weight"
    mock_initializer[1].name = "conv1.bias"
    mock_model = MagicMock()
    mock_model.graph.initializer = mock_initializer

    mock_to_array.side_effect = [1, 2]
    mock_tensor.side_effect = lambda x: f"tensor({x})"

    result = g.weights_onnx_to_torch_format(mock_model)
    assert result["conv1"].weight == "tensor(1)"
    assert result["conv1"].bias == "tensor(2)"

# --------- File I/O ---------

@patch("builtins.open", new_callable=mock_open, read_data="1.0 2.0 3.0")
def test_read_tensor_from_file(mock_file):
    g = GeneralLayerFunctions()
    tensor = g.read_tensor_from_file("dummy.txt")
    assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0]))


@patch("builtins.open", new_callable=mock_open, read_data='{"input": [1, 2, 3]}')
def test_read_input(mock_file):
    g = GeneralLayerFunctions()
    result = g.read_input("dummy.json")
    assert result == [1, 2, 3]

# --------- Output Reading ---------


def test_read_output_torch():
    g = GeneralLayerFunctions()
    model = MagicMock()
    model.return_value = "mocked_output"
    input_data = [1, 2, 3]

    result = g.read_output(model, input_data, is_torch=True)
    assert result == "mocked_output"


@patch("python_testing.utils.pytorch_helpers.ort.InferenceSession")
def test_read_output_onnx(mock_session_cls):
    g = GeneralLayerFunctions()
    mock_session = MagicMock()
    mock_session_cls.return_value = mock_session
    mock_session.get_inputs.return_value = [MagicMock(name="input")]
    mock_session.get_inputs()[0].name = "input_name"
    mock_session.run.return_value = ["onnx_output"]

    result = g.read_output("model.onnx", [1, 2, 3], is_torch=False)
    assert result == ["onnx_output"]

# --------- Input Handling ---------

@patch.object(GeneralLayerFunctions, 'read_input', return_value=[1, 2])
def test_get_inputs_from_file_scaled(mock_read_input):
    g = GeneralLayerFunctions()
    g.input_shape = (2,)
    g.scale_base = 10
    g.scaling = 2
    tensor = g.get_inputs_from_file("file", is_scaled=False)
    assert torch.equal(tensor, torch.tensor([1, 2]) * 100)
    assert tensor.shape == (2,)


@patch.object(GeneralLayerFunctions, 'read_input', return_value=[3, 4])
def test_get_inputs_from_file_long(mock_read_input):
    g = GeneralLayerFunctions()
    g.input_shape = (2,)
    tensor = g.get_inputs_from_file("file", is_scaled=True)
    assert torch.equal(tensor, torch.tensor([3, 4]))
    assert tensor.shape == (2,)


def test_create_new_inputs_shape_and_type():
    g = GeneralLayerFunctions()
    g.input_shape = (2, 3)
    g.scale_base = 10
    g.scaling = 1
    result = g.create_new_inputs()
    assert result.shape == (2, 3)
    assert result.dtype == torch.long
    assert result.max() < 10
    assert result.min() > -10


def test_get_inputs_with_shape(monkeypatch):
    g = GeneralLayerFunctions()
    g.input_shape = (1, 2)
    g.get_inputs_from_file = lambda f, is_scaled: torch.tensor([1, 2])
    result = g.get_inputs("dummy.txt")
    assert torch.equal(result, torch.tensor([1, 2]).reshape(1, 2))


def test_get_inputs_raises_if_no_shape():
    g = GeneralLayerFunctions()
    with pytest.raises(NotImplementedError, match = "Must define attribute input_shape"):
        g.get_inputs("dummy.txt")


# --------- Formatters ---------

def test_format_inputs():
    g = GeneralLayerFunctions()
    result = g.format_inputs(torch.tensor([1, 2]))
    assert result == {"input": [1, 2]}


def test_format_outputs():
    g = GeneralLayerFunctions()
    result = g.format_outputs(torch.tensor([5, 6]))
    assert result == {"output": [5, 6]}


def test_format_inputs_outputs():
    g = GeneralLayerFunctions()
    inputs = torch.tensor([1])
    outputs = torch.tensor([9])
    i, o = g.format_inputs_outputs(inputs, outputs)
    assert i == {"input": [1]}
    assert o == {"output": [9]}
