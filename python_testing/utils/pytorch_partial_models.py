import torch
import torch.nn.functional as F
import torch.nn as nn


class QuantizedLinear(nn.Module):
    def __init__(self, original_linear: nn.Linear, scale: int, rescale_output: bool = True):
        super().__init__()
        self.scale = scale
        self.shift = int(scale).bit_length() - 1  # assumes power of 2 scale
        self.rescale_output = rescale_output
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        
        # Quantize weights and biases to integers
        self.weight = nn.Parameter((original_linear.weight.data * scale).long(), requires_grad=False)
        if not original_linear.bias is None:
            bias = original_linear.bias.data * scale * scale
        else:
            bias = torch.tensor(0)
        self.bias = nn.Parameter(bias.long(), requires_grad=False)

    def forward(self, x):
        # Assume x is already scaled (long), do matmul in int domain
        x = x.long()
        
        out = torch.matmul(x, self.weight.t())
        out += self.bias
        if self.rescale_output:
            out = out >> self.shift  # scale down


        return out
    

class QuantizedConv2d(nn.Module):
    def __init__(self, original_conv: nn.Conv2d, scale: int, rescale_output: bool = True):
        super().__init__()
        self.scale = scale
        self.shift = int(scale).bit_length() - 1  # assumes scale is a power of 2
        self.rescale_output = rescale_output

        self.stride = original_conv.stride
        self.padding = original_conv.padding
        self.dilation = original_conv.dilation
        self.groups = original_conv.groups
        self.in_channels = original_conv.in_channels
        self.kernel_size = original_conv.kernel_size

        # Convert weights and biases to long after scaling
        weight = original_conv.weight.data * scale
        self.weight = nn.Parameter(weight.long(), requires_grad=False)

        if original_conv.bias is not None:
            bias = original_conv.bias.data * scale * scale
            self.bias = nn.Parameter(bias.long(), requires_grad=False)
        else:
            self.bias = None

    def forward(self, x):
        x = x.long()  # ensure input is long
        out = F.conv2d(
            x,
            self.weight,
            bias=None,  # do bias separately to handle shift
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups
        )
        
        if self.bias is not None:
            out += self.bias.view(1, -1, 1, 1)
        
        if self.rescale_output:
            out = out >> self.shift
        return out
    

class Conv2DModel(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(Conv2DModel, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

    def forward(self, x):
        return self.conv(x)
    
class Conv2DModelReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super(Conv2DModelReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias = bias

    def forward(self, x):
        return F.relu(self.conv(x))
    

class MatrixMultiplicationModel(nn.Module):
    def __init__(self, in_channels, out_channels, bias = False):
        super(MatrixMultiplicationModel, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels, bias)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return self.fc1(x)
    
class MatrixMultiplicationReLUModel(nn.Module):
    def __init__(self, in_channels, out_channels, bias = False):
        super(MatrixMultiplicationReLUModel, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels, bias = False)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        return F.relu(self.fc1(x))
    

class MaxPooling2DModel(nn.Module):
    def __init__(self, kernel_size, stride, padding = 0 , dilation = 1, return_indeces = False, ceil_mode = False):
        super(MaxPooling2DModel, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indeces, ceil_mode)
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return self.pool(x)