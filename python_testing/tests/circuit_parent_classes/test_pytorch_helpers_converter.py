import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock, mock_open
from types import SimpleNamespace
import inspect

from python_testing.utils.pytorch_helpers import PytorchConverter, QuantizedConv2d, QuantizedLinear

# python_testing/utils/pytorch_helpers

# ---------- Save & Load Models ----------
@pytest.mark.unit
@patch("torch.save")
def test_save_model(mock_save):
    c = PytorchConverter()
    c.model = MagicMock()
    c.model.state_dict.return_value = {"weights": 123}
    c.save_model("model.pt")
    mock_save.assert_called_once_with({"weights": 123}, "model.pt")

@pytest.mark.unit
@patch("torch.load", return_value={"weights": 123})
def test_load_model(mock_load):
    c = PytorchConverter()
    c.model = MagicMock()
    c.load_model("model.pt")
    c.model.load_state_dict.assert_called_once_with({"weights": 123})

@pytest.mark.unit
@patch("torch.save")
def test_save_quantized_model(mock_save):
    c = PytorchConverter()
    c.quantized_model = "quantized_model_obj"
    c.save_quantized_model("quant.pt")
    mock_save.assert_called_once_with("quantized_model_obj", "quant.pt")

@pytest.mark.unit
@patch("torch.load", return_value="quantized_model_obj")
def test_load_quantized_model(mock_load):
    c = PytorchConverter()
    c.load_quantized_model("quant.pt")
    mock_load.assert_called_once_with("quant.pt", weights_only=False)
    assert c.quantized_model == "quantized_model_obj"

# ---------- Padding Expansion ----------
@pytest.mark.unit
def test_expand_padding():
    c = PytorchConverter()
    assert c.expand_padding((1, 2)) == (2, 2, 1, 1)

@pytest.mark.unit
def test_expand_padding():
    c = PytorchConverter()
    with pytest.raises(ValueError, match = "Expand padding requires initial padding of dimension 2"):
        c.expand_padding((1,2,2))

# ---------- get_used_layers ----------
@pytest.mark.unit
def test_get_used_layers():
    c = PytorchConverter()
    layer = nn.Linear(4, 4)
    model = nn.Sequential(layer)
    used = c.get_used_layers(model, (1, 4))
    assert used[0][0] == "0"
    assert isinstance(used[0][1], nn.Linear)

    layer = nn.Linear(4, 4)
    model = nn.Sequential(layer, layer)
    used = c.get_used_layers(model, (1, 4))
    print(used)
    assert used[0][0] == "0"
    assert used[1][0] == "0"
    assert isinstance(used[0][1], nn.Linear)
    assert isinstance(used[1][1], nn.Linear)

# ---------- get_input_shapes_by_layer ----------
@pytest.mark.unit
def test_get_input_shapes_by_layer():
    c = PytorchConverter()
    model = nn.Sequential(nn.Conv2d(1, 1, 3, padding=1), nn.ReLU())
    shape_map, out_shape_map = c.get_input_and_output_shapes_by_layer(model, (1, 1, 5, 5))
    print(out_shape_map)
    assert "0_0" in shape_map
    assert shape_map["0_0"] == torch.Size([1, 1, 5, 5])

    assert "0_0" in out_shape_map
    assert out_shape_map["0_0"] == torch.Size([1, 1, 5, 5])

# ---------- clone_model_with_same_args ----------
@pytest.mark.unit
def test_clone_model_with_init_args():
    class Dummy(nn.Module):
        def __init__(self, a, b=5):
            super().__init__()
            self.a = a
            self.b = b

    c = PytorchConverter()
    model = Dummy(a=1, b=2)
    cloned = c.clone_model_with_same_args(model)
    assert isinstance(cloned, Dummy)
    assert cloned.a == 1
    assert cloned.b == 2

# ---------- quantize_model ----------
@pytest.mark.unit
def test_quantize_model():
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.conv1 = nn.Conv2d(1, 1, 3)

    model = Dummy()
    c = PytorchConverter()
    quantized = c.quantize_model(model, scale=100, rescale_config={"fc1": True})
    assert isinstance(quantized, Dummy)
    assert isinstance(getattr(quantized, "fc1"), QuantizedLinear)
    assert isinstance(getattr(quantized, "conv1"), QuantizedConv2d)

# ---------- get_weights ----------
@pytest.mark.integration
@patch.object(PytorchConverter, 'get_input_and_output_shapes_by_layer')
@patch.object(PytorchConverter, 'get_used_layers')
def test_get_weights(mock_used_layers, mock_shapes):
    class Dummy(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.conv1 = nn.Conv2d(1, 1, 3)
            self.pool = nn.MaxPool2d(3)
    model = Dummy()

    c = PytorchConverter()
    c.scale_base = 10
    c.scaling = 2
    c.input_shape = (1, 1, 5, 5)

    conv = nn.Conv2d(1, 1, 3, padding=1)
    linear = nn.Linear(4, 4)

    c.quantized_model = c.quantize_model(model, scale=100, rescale_config={"fc1": True})


    mock_used_layers.return_value = [("conv", conv), ("fc", linear)]
    mock_shapes.return_value = ({"conv_0": (1, 1, 5, 5), "fc_0": (1, 2)}, {"conv_0": (1, 1, 5, 5), "fc_0": (1, 2)})

    weights = c.get_weights()
    assert "conv_weights" in weights
    assert "fc_weights" in weights

@pytest.mark.integration
def test_get_weights_values():
    class DummyConv(nn.Conv2d):
        def __init__(self):
            super().__init__(1, 1, 1)
            self.weight.data.fill_(2.0)
            self.bias.data.fill_(3.0)

    class DummyLinear(nn.Linear):
        def __init__(self):
            super().__init__(2, 2)
            self.weight.data.fill_(4.0)
            self.bias.data.fill_(5.0)

    c = PytorchConverter()
    c.input_shape = (1, 1, 1, 1)
    c.scale_base = 10
    c.scaling = 2

    conv = DummyConv()
    linear = DummyLinear()
    c.quantized_model = c.quantize_model(conv, 1)


    with patch.object(PytorchConverter, 'get_used_layers', return_value=[("conv", conv), ("fc", linear)]), \
         patch.object(PytorchConverter, 'get_input_and_output_shapes_by_layer', return_value=({"conv_0": (1, 1, 1, 1), "fc_0": (1, 2)}, {"conv_0": (1, 1, 1, 1), "fc_0": (1, 2)})):

        weights = c.get_weights()

        assert weights["conv_weights"][0] == [[[ [2.0] ]]]
        assert weights["conv_bias"][0] == [3.0]
        assert weights["fc_weights"][0] == [[4.0, 4.0], [4.0, 4.0]]
        assert weights["fc_bias"][0] == [[5.0, 5.0]]

# ---------- get_model ----------
@pytest.mark.unit
def test_get_model_success():
    class DummyModel(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.hidden_size = hidden_size

    c = PytorchConverter()
    c.model_type = DummyModel
    c.model_params = {"hidden_size": 10}
    model = c.get_model("cpu")
    assert isinstance(model, DummyModel)
    assert model.hidden_size == 10

@pytest.mark.unit
def test_get_model_failure():
    c = PytorchConverter()
    with pytest.raises(NotImplementedError):
        c.get_model("cpu")

# ---------- get_model_and_quantize ----------
@pytest.mark.unit
@patch.object(PytorchConverter, "get_model")
@patch("torch.load")
@patch.object(PytorchConverter, "quantize_model")
def test_get_model_and_quantize(mock_quantize, mock_load, mock_get_model):
    c = PytorchConverter()
    c.model_file_name = "file.pt"
    c.scale_base = 10
    c.scaling = 2
    mock_get_model.return_value = MagicMock()
    mock_load.return_value = {"model_state_dict": "weights"}
    mock_quantize.return_value = MagicMock()

    c.get_model_and_quantize()
    assert hasattr(c, "model")
    assert hasattr(c, "quantized_model")

# ---------- test_accuracy ----------
@pytest.mark.unit
def test_test_accuracy_prints(capsys):
    c = PytorchConverter()
    c.scale_base = 10
    c.scaling = 1
    c.input_shape = (1, 4)
    c.model = MagicMock(return_value="original_out")
    c.quantized_model = MagicMock(return_value=torch.tensor([100.0]))

    c.test_accuracy()

    captured = capsys.readouterr()
    assert "original_out" in captured.out

@pytest.mark.unit
def test_model_save_and_load_equivalence(tmp_path):
    class Simple(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(2, 2)

    c = PytorchConverter()
    model = Simple()
    model.fc.weight.data.fill_(42.0)
    model.fc.bias.data.fill_(7.0)
    c.model = model

    file_path = tmp_path / "model.pt"
    c.save_model(str(file_path))

    # Load into a fresh instance
    c2 = PytorchConverter()
    c2.model = Simple()
    c2.load_model(str(file_path))

    for p1, p2 in zip(c.model.parameters(), c2.model.parameters()):
        assert torch.allclose(p1, p2)