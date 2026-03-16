import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    def forward(self, x):
        identity = x
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + identity
        out = torch.relu(out)
        return out


class MiniResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1, bias=True)
        self.block1 = ResidualBlock(8)
        self.block2 = ResidualBlock(8)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(8 * 14 * 14, 10, bias=True)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


model = MiniResNet()
model.eval()

dummy = torch.randn(1, 1, 28, 28)

torch.onnx.export(
    model,
    dummy,
    "rust/jstprove_remainder/models/mini_resnet.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    dynamo=False,
)
print("Exported mini_resnet.onnx")
