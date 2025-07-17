import pytest
import torch
import torch.nn as nn
from python.testing.python_testing.utils.pytorch_partial_models import QuantizedLinear, QuantizedConv2d
from python.testing.python_testing.utils.pytorch_partial_models import (
    Conv2DModel, Conv2DModelReLU,
    MatrixMultiplicationModel, MatrixMultiplicationReLUModel
)

# ---------- QuantizedLinear ----------
@pytest.mark.unit
def test_quantized_linear_forward_matches_float():
    scale = 16
    linear = nn.Linear(4, 2, bias=True)
    linear.weight.data.fill_(0.5)
    linear.bias.data.fill_(0.25)

    q = QuantizedLinear(linear, scale)

    # Float model
    x_float = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    y_float = linear(x_float)

    # Quantized input
    x_q = (x_float * scale).long()
    y_q = q(x_q)
    y_deq = y_q.float() / scale  # Re-scale down

    assert y_float.shape == y_deq.shape
    torch.testing.assert_close(y_float, y_deq)

@pytest.mark.unit
def test_quantized_linear_no_rescale():
    scale = 100
    linear = nn.Linear(2, 2)
    linear.weight.data.fill_(0.05)
    linear.bias.data.fill_(0.025)
    q = QuantizedLinear(linear, scale, rescale_output=False)
    

    # Float model
    x_float = torch.tensor([[0.05, 0.1]])
    y_float = linear(x_float)

    # Quantized input
    x_q = (x_float * scale).long()
    y_q = q(x_q)
    y_deq = y_q.float() / (scale*scale)  # Re-scale down

    assert y_q.dtype == torch.long
    torch.testing.assert_close(y_float, y_deq)



# ---------- QuantizedConv2d ----------
@pytest.mark.unit
def test_quantized_conv2d_forward_matches_float():
    conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    conv.weight.data.fill_(1.0)
    conv.bias.data.fill_(1.0)

    scale = 16
    qconv = QuantizedConv2d(conv, scale)

    x_float = torch.ones((1, 1, 3, 3))
    y_float = conv(x_float)

    x_q = (x_float * scale).long()
    y_q = qconv(x_q)
    y_deq = y_q.float() / scale  # rescale output

    assert y_float.shape == y_deq.shape
    torch.testing.assert_close(y_float, y_deq)

@pytest.mark.unit
def test_quantized_conv2d_without_bias():
    conv = nn.Conv2d(1, 1, kernel_size=1, bias=False)
    qconv = QuantizedConv2d(conv, scale=16)
    x = torch.ones((1, 1, 1, 1)).long()
    y = qconv(x)
    assert y.shape == (1, 1, 1, 1)
    assert y.dtype == torch.long

@pytest.mark.unit
def test_quantized_conv2d_no_rescale():
    scale = 16
    conv = nn.Conv2d(1, 1, kernel_size=1)
    conv.weight.data.fill_(0.5)
    conv.bias.data.fill_(0.25)
    qconv = QuantizedConv2d(conv, scale=scale, rescale_output=False)
    x_float = torch.ones((1, 1, 1, 1))
    # Float model
    y_float = conv(x_float)

    # Quantized input
    x_q = (x_float * scale).long()
    y_q = qconv(x_q)
    y_deq = y_q.float() / (scale*scale)  # Re-scale down

    assert y_q.dtype == torch.long
    print(y_float, y_deq)
    torch.testing.assert_close(y_float, y_deq)


# ---------- Simple Model Wrappers ----------
@pytest.mark.unit
def test_conv2d_model_output():
    model = Conv2DModel(1, 1)
    x = torch.ones((1, 1, 5, 5))
    y = model(x)
    assert y.shape[1] == 1

@pytest.mark.unit
def test_conv2d_model_relu_output():
    model = Conv2DModelReLU(1, 1, bias = True)

    model.conv.weight.data.fill_(0.5)
    model.conv.bias.data.fill_(0.25)
    x = -torch.ones((1, 1, 5, 5))
    y = model(x)

    assert torch.all(y == 0)  # relu zero-out

@pytest.mark.unit
def test_matmul_model_output_shape():
    model = MatrixMultiplicationModel(4, 2)
    x = torch.ones((1, 4))
    y = model(x)
    assert y.shape == (1, 2)

@pytest.mark.unit
def test_matmul_relu_model_non_negative():
    model = MatrixMultiplicationReLUModel(4, 2)
    model.fc1.weight.data.fill_(0.5)
    x = -torch.ones((1, 4))
    y = model(x)
    assert (y >= 0).all()
